#include "optimization.h"
#include "dual.h"


// cpu pso code to initialize the array of N * DIM
//template<typename FuncEval>
std::vector<double>
hostPSOInit(const std::function<double(const double*)>& f_eval,
    double          lower,
    double          upper,
    int             N,
    int             DIM,
    int             PSO_ITERS = 10)
{
    //printf("before allocation\n");
    // allocate arrays
    std::vector<double> X(N*DIM);
    //double* X        = new double[N*DIM];
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
            X[i*DIM + d]      = uniform_rand(lower, upper);
            V[i*DIM + d]      = uniform_rand(-vel_range, vel_range);
            pBestX[i*DIM + d] = X[i*DIM + d];
        }
        pBestVal[i] = f_eval(&X[i*DIM]);
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
                double r1 = uniform_rand(0.0, 1.0), r2 = uniform_rand(0.0, 1.0);;
                V[i*DIM + d] = w * V[i*DIM + d]
                              + c1 * r1 * (pBestX[i*DIM + d] - X[i*DIM + d])
                              + c2 * r2 * (gBestX[d]       - X[i*DIM + d]);
                X[i*DIM + d] = X[i*DIM + d] + V[i*DIM + d];
            }
            double f = f_eval(&X[i*DIM]);
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
    /*for (int i = 0; i < N*DIM; ++i) {
        hostPsoArray[i] = X[i];
    }*/

    // clean up 
    //delete[] X;
    delete[] V;
    delete[] pBestX;
    delete[] pBestVal;
    delete[] gBestX;
    return X;
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
double line_search(const std::function<double(const std::vector<double>&)>& func,
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

Result optimize(const ADFunc &f_ad,
    std::vector<double>       x0,
    const std::string &       algorithm,
    double                    tol,
    int                       max_iter) {
    /* real‐valued wrapper for line‐search
    auto f_real = [&](std::vector<double> &xx){
        int n = xx.size();
        std::vector<dual::DualNumber> tmp(n);
        for (int i = 0; i < n; ++i) tmp[i] = {xx[i], 0.0};
        return f_ad(tmp).real;
    };*/
    auto f_real = [&](const std::vector<double>& xx) {
	    int n = xx.size();
	    std::vector<dual::DualNumber> tmp(n);
	    for (int i = 0; i < n; ++i) tmp[i] = {xx[i], 0.0};
	    return f_ad(tmp).real;
    };

	//const ADFunc & f_ad, std::vector<double> x0, std::string algorithm,  double tol, int max_iter, std::pair<std::vector<double>,std::vector<double>> bounds) {
    double min_value = std::numeric_limits<double>::max();
    std::vector<double> x = x0;
    double global_min = std::numeric_limits<double>::max();

    Result result;
    result.status       = -1;     // assume “not converged” by default
    result.fval         = 333777.0;
    result.gradientNorm = 69.0;
    result.coordinates.resize(x.size());
    for (int d = 0; d < x.size(); ++d) {
        result.coordinates[d] = 0.0;
    }
    result.iter = -1;
   

    // Initialize the Hessian matrix to identity matrix
    std::vector<std::vector<double>> H(x0.size(), std::vector<double>(x0.size(), 0));
    for (int i = 0; i < x0.size(); i++) {
        H[i][i] = 1; // along the diagonals, place 1's to create Identity Matrix
    }//end for
    std::vector<double> g;
    // Main loop
    int i=0;
    for (i = 0; i < max_iter; i++) {
        // Compute Gradient
        g = gradientAD(f_ad, x);//gradient(func, x, 1e-7);

        // Check if the length of gradient vector is less than our tolerance
        if (norm(g) < 1e-8) { 
        	std::cout << "converged" << std::endl;
            min_value = std::min(min_value, f_real(x));
            if (min_value < global_min) {
                global_min = min_value;
                //std::cout << "\nnorm(g): New Global Minimum: " << global_min << " with parameters:" <<std::endl;
                //best_params = {};
                for (int i=0;i<x.size();i++){
                    //best_params.push_back(x[i]);
                    result.coordinates[i] = x[i];
                //    std::cout<< "x["<<i<<"]: " << best_params[i]<<std::endl;
                }
                //return result;
            }//end if
            result.iter = i;
            result.fval = min_value;
            result.status = 1;
            return result;
        }// end if

        // Compute Search Direction
        std::vector<double> p = matvec_product(H, g);
        for (auto &val : p) {
            val = -val; // opposite of greatest increase
        }// end for

        /*** Calculate optimal step size in the search direction p ***/
        double f0 = f_real(x);

        double alpha = line_search(f_real, f0, x, p, g); //

        // Update the current point x by taking a step of size alpha in the direction p.
        std::vector<double> x_new = x;
        for (int j = 0; j < x.size(); j++) { x_new[j] += alpha * p[j];}// end for

        // Compute the difference between the new point and the old point, delta_x
        std::vector<double> delta_x = x_new;
        for (int j = 0; j < x.size(); j++) { delta_x[j] -= x[j];}// end for

        // Compute the difference in the gradient at the new point and the old point, delta_g.
        std::vector<double> delta_g = gradientAD(f_ad, x);//gradient(func, x_new, 1e-7);
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
        min_value = std::min(min_value, f_real(x));
        //}//end else
    }// end main loop

    result.iter = i;
    result.status = 0;
    result.gradientNorm = norm(g);
    for (int i=0;i<x.size();i++){
        result.coordinates[i] = x[i];
        //    std::cout<< "x["<<i<<"]: " << best_params[i]<<std::endl;
    }
    result.fval = min_value;
    return result; 
}// end dfp

Result run_minimizers(const ADFunc &f_ad, std::string name, int pso_iter, int bfgs_iter, 
    int pop_size, int dim,int seed, int converged, double tolerance, std::string algorithm,double lower,double upper) {
    global_min = std::numeric_limits<double>::max();
    
    // wrap f_ad into a simple double(const double*) function:
    auto f_eval = [&](const double* x) {
        std::vector<dual::DualNumber> xx(dim);
        for (int d = 0; d < dim; ++d) xx[d] = { x[d], 0.0 };
        return f_ad(xx).real;
    };

    long total_time = 0;
    int converged_counter = 0;
    Result global_best;
    global_best.fval = global_min;
    std::vector<double> swarm;
    if (pso_iter > 0) {
    	auto start = std::chrono::high_resolution_clock::now();
        swarm = hostPSOInit(f_eval,lower,upper,pop_size, dim,pso_iter);
        auto stop = std::chrono::high_resolution_clock::now();
    	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    	total_time += duration.count();
    }

    for(int i=0;i<pop_size;i++) {
    	std::vector<double> x0(dim);
    	if (!swarm.empty()) {
    		for(int d=0;d<dim;++d) 
    			x0[d] = swarm[i*dim + d];
    	} else {
    		for (int d=0; d<dim; ++d) 
    			x0[d] = uniform_rand(lower, upper);
    	}
    	auto start = std::chrono::high_resolution_clock::now();
    	Result result = optimize(f_ad,x0,algorithm,tolerance, 10000);
    	auto stop = std::chrono::high_resolution_clock::now();
    	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    	result.time = duration.count();
    	total_time += result.time;
    	if (result.fval < global_min) {
    		global_best = result;
    	}
    	if(result.status == 1) {
			converged_counter+= 1;
			if(converged_counter == converged) {
				std::cout << "\nLast particle converged!" << std::endl;
				break;
			}
    	}
    }
    //auto final_population = genetic_algo(func, max_gens, pop_size, dim, x0, algorithm, bounds);

    std::cout << "\ntotal time: " << total_time << " ms" << std::endl;
    std::cout << "Best Particle" << name << ": " << global_best.fval <<std::endl;
    std::cout << "at the coordinates: \n";
    for(int i=0;i<global_best.coordinates.size();i++) {
    	std::cout << "x["<<i<<"]: "<<std::scientific<<global_best.coordinates[i] << "\n";
    }
    std::cout << "\nin " << global_best.iter << " iterations." << std::endl;
    return global_best;
}
