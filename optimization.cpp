#include "optimization.h"
#include "dual.h"


// cpu pso code to initialize the array of N * DIM
template<typename FuncEval>
void hostPSOInit(
    FuncEval        FunctionEval,  // e.g. [&](const double* x){ return Function::evaluate(x); }
    double          lower,
    double          upper,
    double*         hostPsoArray,  // output: length = N*DIM
    int             N,
    int             DIM,
    int             PSO_ITERS = 10)
{
    // simple 64‑bit LCG for host randomness
    uint64_t state = 1234;
    auto rand01 = [&](){
      state = state * 6364136223846793005ULL + 1;
      return double((state >> 11) & ((1ULL<<53)-1)) / double(1ULL<<53);
    };
    auto randR = [&](double lo, double hi){
            return lo + (hi - lo) * rand01();
    };
    //printf("before allocation\n");
    // allocate arrays
    double* X        = new double[N*DIM];
    double* V        = new double[N*DIM];
    double* pBestX   = new double[N*DIM];
    double* pBestVal = new double[N];
    double* gBestX   = new double[DIM];
    double  gBestVal;
    //printf("initialize positions, velocities, and pbs.\n");
    // initialize positions, velocities, and personal bests
    double vel_range = (upper - lower) * 0.1;
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < DIM; ++d) {
            X[i*DIM + d]      = randR(lower, upper);
            V[i*DIM + d]      = randR(-vel_range, vel_range);
            pBestX[i*DIM + d] = X[i*DIM + d];
        }
        pBestVal[i] = FunctionEval(&X[i*DIM]);
    if (i < 3) {  // print first 3 particles
      printf("init particle %2d: X = [", i);
      for (int d = 0; d < DIM; ++d)
        printf(" %8.4f", X[i*DIM + d]);
      printf(" ]  V = [");
      for (int d = 0; d < DIM; ++d)
        printf(" %8.4f", V[i*DIM + d]);
      printf(" ]  f(pi)=%.4e\n", pBestVal[i]);
    } }
    // find initial global best
    gBestVal = pBestVal[0];
    for (int d = 0; d < DIM; ++d)
        gBestX[d] = pBestX[d];
    for (int i = 1; i < N; ++i) {
        if (pBestVal[i] < gBestVal) {
            gBestVal = pBestVal[i];
            for (int d = 0; d < DIM; ++d)
                gBestX[d] = pBestX[i*DIM + d];
        }
    }
    printf(" initial gBestVal = %.4e at position [", gBestVal);
    for (int d = 0; d < DIM; ++d)
      printf(" %8.4f", gBestX[d]);
    printf(" ]\n");
    // pso  main loop
    const double w = 0.7, c1 = 1.4, c2 = 1.4;
    for (int it = 0; it < PSO_ITERS; ++it) {
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < DIM; ++d) {
                double r1 = rand01(), r2 = rand01();
                V[i*DIM + d] = w * V[i*DIM + d]
                              + c1 * r1 * (pBestX[i*DIM + d] - X[i*DIM + d])
                              + c2 * r2 * (gBestX[d]       - X[i*DIM + d]);
                X[i*DIM + d] = X[i*DIM + d] + V[i*DIM + d];
            }
            double f = FunctionEval(&X[i*DIM]);
            // personal best
            if (f < pBestVal[i]) {
                pBestVal[i] = f;
                for (int d = 0; d < DIM; ++d)
                    pBestX[i*DIM + d] = X[i*DIM + d];
            }
            // global best
            if (f < gBestVal) {
                gBestVal = f;
                for (int d = 0; d < DIM; ++d)
                    gBestX[d] = X[i*DIM + d];
            }
        }
    }
    // print the best‐ever solution found
    printf(" final gBestVal = %.6e  at gBestX = [", gBestVal);
    for (int d = 0; d < DIM; ++d)
        printf(" %8.4f", gBestX[d]);
    printf(" ]\n");

    //  write final swarm positions back to hostPsoArray
    for (int i = 0; i < N*DIM; ++i) {
        hostPsoArray[i] = X[i];
    }
    // clean up 
    delete[] X;
    delete[] V;
    delete[] pBestX;
    delete[] pBestVal;
    delete[] gBestX;
}


/**
 * Back-tracking line search (Armijo condition).
 *
 * @param func   Objective: R^n → R, accepts x by const-ref.
 * @param f0     f(x) at the current iterate.
 * @param x      Current point (size n).
 * @param p      Search direction (size n).
 * @param g      Gradient at x (size n).
 * @returns      Step length α ∈ (0,1] satisfying f(x+αp) ≤ f0 + c1 α gᵀp.
 */
double line_search(std::function<double(std::vector<double>&)>& func,
                   double                                   f0,
                   const std::vector<double>&               x,
                   const std::vector<double>&               p,
                   const std::vector<double>&               g)
{
    const double c1    = 0.3;
    double       alpha = 1.0;
    double       ddir  = dot_product(g, p);

    std::vector<double> xTemp(x.size());
    // limit to max 20 halving steps
    for (int iter = 0; iter < 20; ++iter) {
        // xTemp = x + alpha * p
        for (size_t j = 0; j < x.size(); ++j) {
            xTemp[j] = x[j] + alpha * p[j];
        }
        double f1 = func(xTemp);
        // Armijo condition
        if (f1 <= f0 + c1 * alpha * ddir) {
            break;
        }
        alpha *= 0.5;
    }
    return alpha;
}

double safe_divide(double numerator, double denominator, double default_value = std::numeric_limits<double>::max()) {
    if (std::abs(denominator) < 1e-10) {
        return default_value;
    }
    return numerator / denominator;
}


/* The Broyden–Fletcher–Goldfarb–Shanno update
void bfgs_update(std::vector<std::vector<double>>& H,
                 std::vector<double> delta_x,
                 std::vector<double> delta_g,
                 double delta_dot) {
    int n = delta_x.size();
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; i++) {
        I[i][i] = 1; // Identity Matrix
    }

    std::vector<std::vector<double>> term1 = outer_product(delta_x, delta_g);
    for (int row = 0; row < term1.size(); row++) {
        for (int col = 0; col < term1[row].size(); col++) {
            term1[row][col] /= delta_dot;
        }
    }

    std::vector<std::vector<double>> term1H = matmul(term1, H);
    std::vector<std::vector<double>> termHterm1 = matmul(H, term1);

    std::vector<std::vector<double>> term2 = outer_product(delta_x, delta_x);
    for (int row = 0; row < term2.size(); row++) {
        for (int col = 0; col < term2[row].size(); col++) {
            term2[row][col] /= delta_dot;
        }
    }

    for (int row = 0; row < H.size(); row++) {
        for (int col = 0; col < H[row].size(); col++) {
            H[row][col] = (I[row][col] - term1H[row][col]) * H[row][col] * (I[row][col] - termHterm1[row][col]) + term2[row][col];
        }
    }
}
*/


// BFGS update: sequential, dynamic-dimension
void bfgs_update_seq(std::vector<std::vector<double>>& H,
                 std::vector<double> s,
                 std::vector<double> y,
                 double sTy)
{
    int n = s.size();
    if (std::fabs(sTy) < 1e-14) return;      // skip if denominator too small
    double rho = 1.0 / sTy;

    // allocate new Hessian approximation
    std::vector<std::vector<double>> H_new(n, std::vector<double>(n, 0.0));

    // Compute H_new = (I - rho·s·yᵀ) · H · (I - rho·y·sᵀ) + rho·s·sᵀ
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // First term: (I - rho·s·yᵀ) * H * (I - rho·y·sᵀ)
            double accum = 0.0;
            for (int k = 0; k < n; ++k) {
                // A_{ik} = δ_{ik} - rho * s[i] * y[k]
                double Aik = ((i == k) ? 1.0 : 0.0) - rho * s[i] * y[k];

                // sum over m: H[k][m] * B_{mj}, where B_{mj} = δ_{mj} - rho * y[m] * s[j]
                double inner = 0.0;
                for (int m = 0; m < n; ++m) {
                    double Bmj = ((m == j) ? 1.0 : 0.0) - rho * y[m] * s[j];
                    inner += H[k][m] * Bmj;
                }

                accum += Aik * inner;
            }

            // Second term: + rho * s[i] * s[j]
            H_new[i][j] = accum + rho * s[i] * s[j];
        }
    }

    // Copy back into H
    H.swap(H_new);
}


/* Davidon-Fletcher-Powell optimization algorithm */
void dfp_update(std::vector<std::vector<double>>& H,
                std::vector<double> delta_x,
                std::vector<double> delta_g) {

    // term1 is the outer product of delta_x with itself.
    std::vector<std::vector<double>> term1 = outer_product(delta_x, delta_x);

    // term2 is the dot product of delta_x and delta_g
    // delgam in Minuit code
    double term2 = dot_product(delta_x, delta_g);

    // term3 is the matrix multiplication of H and the outer product of delta_g with itself.
    std::vector<std::vector<double>> term3 = matmul(H, outer_product(delta_g, delta_g));
    // term4 is the result of term3 being matrix multiplied with H again.
    std::vector<std::vector<double>> term4 = matmul(term3, H);

    // term5 is the dot product of delta_g and the result of H matrix multiplied with delta_g.
    // in Minuit code it is gvg
    double term5 = dot_product(delta_g, matvec_product(H, delta_g));

    // The approximation of the inverse Hessian is updated element by element.
    for (int row = 0; row < H.size(); row++) {
        for (int col = 0; col < H[row].size(); col++) {
            H[row][col] = H[row][col] + term1[row][col] / safe_divide(term1[row][col], term2) - term4[row][col] / safe_divide(term4[row][col], term5);
        }// end inner for
    }// end outer for
    /*std::cout<<"\nHessian:\n";
    for (int row = 0; row < H.size(); row++) {
        for (int col = 0; col < H[row].size(); col++) {
            std::cout << H[row][col] << "  ";
        }// end inner for
        std::cout << std::endl;
    }// end outer for
    */
}

double optimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string algorithm,  double tol, int max_iter, std::pair<std::vector<double>,std::vector<double>> bounds) {
    double min_value = std::numeric_limits<double>::max();
    std::vector<double> x = x0;

    // L-BFGS-B init
    std::deque<std::vector<double>> s_history, y_history;
    std::deque<double> rho_history;
    //double gamma_k = 1.0;  // initial scaling factor
    //int m = 3;  // history size

    // Initialize the Hessian matrix to identity matrix
    std::vector<std::vector<double>> H(x0.size(), std::vector<double>(x0.size(), 0));
    for (int i = 0; i < x0.size(); i++) {
        H[i][i] = 1; // along the diagonals, place 1's to create Identity Matrix
    }//end for
    
    // Main loop
    for (int i = 0; i < max_iter; i++) {
        // Compute Gradient
        std::vector<double> g = gradient(func, x, 1e-7);

        // Check if the length of gradient vector is less than our tolerance
        if (norm(g) < 1e-6) { 
        	std::cout << "converged" << std::endl;
            min_value = std::min(min_value, func(x));
            if (min_value < global_min) {
                global_min = min_value;
                //std::cout << "\nnorm(g): New Global Minimum: " << global_min << " with parameters:" <<std::endl;
                best_params = {};
                for (int i=0;i<=x.size();i++){
                    best_params.push_back(x[i]);
                //    std::cout<< "x["<<i<<"]: " << best_params[i]<<std::endl;
                }
                return min_value;
            }//end if
            return global_min;
        }// end if

        /*if(algorithm.find("lbfgs") != std::string::npos) {
            //std::pair<std::vector<double>, std::vector<double>> result = lbfgsb_step(func,algorithm,x0,g,bounds,s_history, y_history, rho_history, gamma_k, m);
            //auto [x_new, g_new] = lbfgsb_step(func, x, g, s_history, y_history, rho_history, gamma_k, m);
            //std::vector<double> x_new = result.first;
            //std::vector<double> g_new = result.second;
            //x = x_new;
            //g = g_new;
        } else {*/
        // Compute Search Direction
        std::vector<double> p = matvec_product(H, g);
        for (auto &val : p) {
            val = -val; // opposite of greatest increase
        }// end for

        /*** Calculate optimal step size in the search direction p ***/
        // Simple Inexact Line Search
        //double alpha = line_search_simple(func, x, p, 0.5, 0.5);
        //double alpha = 0.01;
        // Armijo condition,. alpha, c, tau
        //double alpha = line_search(func, x, p, 0.5, 0.5, 0.5);
        //double alpha = simple_backtracking(func, x,p, 0.5, 0.5);
        // Cubic interpolation
        //double alpha = cubicInterpolationLineSearch(func, x, p, x[0]);
        // Quadratic interpolation
        double f0 = func(x);
        double alpha = line_search(func, f0, x, p, g); //
        // quadratic_line_search(func, x, p);
        // Quadratic interpolation same as MnLineSearch in Minuit2
        //double alpha = mnLineSearch(func, x, p);

        // Update the current point x by taking a step of size alpha in the direction p.
        std::vector<double> x_new = x;
        for (int j = 0; j < x.size(); j++) { x_new[j] += alpha * p[j];}// end for

        // Compute the difference between the new point and the old point, delta_x
        std::vector<double> delta_x = x_new;
        for (int j = 0; j < x.size(); j++) { delta_x[j] -= x[j];}// end for

        // Compute the difference in the gradient at the new point and the old point, delta_g.
        std::vector<double> delta_g = gradient(func, x_new, 1e-7);
        for (int j = 0; j < g.size(); j++) { delta_g[j] -= g[j];}// end for


        if (algorithm == "bfgs") {
            // Update the inverse Hessian approximation using BFGS
            double delta_dot = dot_product(delta_x, delta_g);

            bfgs_update_seq(H, delta_x, delta_g, delta_dot);
        } else {
            // Update the approximation of the inverse Hessian using DFP
            dfp_update(H, delta_x, delta_g);
        }
        x = x_new;
        min_value = std::min(min_value, func(x));
        //}//end else
    }// end main loop

    //std::cout << "Maximum iterations reached without convergence.\nOptimized parameters:\n";
    //for (double param : x) {
    //    std::cout << param << "\n";
    //}// end for
    return min_value; 
}// end dfp


long minimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string name, 
              int pop_size, int dim, std::string algorithm, std::pair<std::vector<double>, std::vector<double>> bounds) {
    global_min = std::numeric_limits<double>::max();
    auto start = std::chrono::high_resolution_clock::now();
    //auto final_population = genetic_algo(func, max_gens, pop_size, dim, x0, algorithm, bounds);
    double minima = optimize(func, x0, algorithm, 1e-12, 2500, bounds);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    long time = duration.count();
    std::cout << "\ntime: " << time << " ms" << std::endl;
    std::cout << "Predicted Global minimum for " << name << ": " << minima <<std::endl;
    std::cout << std::endl;
    return time;
}
