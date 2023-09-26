#include "optimization.h"

/*
 * Simple Inexact Line Search
 */
double line_search_simple(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha, double tau) {
    double fx = func(x);
    std::vector<double> x_alpha_p = x;  // new proposed point in the parameter space

    // Adjust x_alpha_p along the search direction p by the initial step size of alpha
    for (int i = 0; i < x.size(); i++) {
        x_alpha_p[i] += alpha * p[i];
    }// end for

    // If the function value at the new proposed point is not less than at the current point,
    //    alpha is reduced by a factor of tau to backtrack along the search direction p
    while (func(x_alpha_p) >= fx) {
        alpha *= tau;

        // Adjust x_alpha_p to the new value of alpha
        for (int i = 0; i < x.size(); i++) {
            x_alpha_p[i] = x[i] + alpha * p[i];
        }// end for
    }// end while
    return alpha;
}// end line_search_simple


/* Armijo Backtracking Line Search to find an appropriate step size alpha 
 * along the direction p at point x for the function func.
 */
double line_search(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha = 0.5, double c = 0.5, double tau = 0.5) {
    std::vector<double> g = gradient(func, x, 1e-7);
    double fx = func(x);
    std::vector<double> x_alpha_p = x; // new proposed point in the parameter space

    // adds alpha * p[i] to x_alpha_p[i] for each i
    // moving x_alpha_p along the search direction p by a step size of alpha.
    for (int i = 0; i < x.size(); i++) {
        x_alpha_p[i] += alpha * p[i];
    }// end for

    // If the Armijo condition is not met,
    //    alpha is reduced by a factor of tau to backtrack along the search direction p
    while (func(x_alpha_p) > fx + c * alpha * dot_product(g, p)) { // Armijo Rule
        alpha *= tau;

        // adjusts x_alpha_p to the new value of alpha.
        for (int i = 0; i < x.size(); i++) {
            x_alpha_p[i] = x[i] + alpha * p[i];
        }// end for
    }// end while
    return alpha;
}// end line_search

/* Line Minimization from the original paper 
 * that use cubic inteprolation
 */
double cubicInterpolationLineSearch(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> s, double f0) {
    double f = func(x);
    std::vector<double> g = gradient(func, x, 1e-7);
    double q = dot_product(g, s);

    // calculate lambda
    double lambda = std::min(1.0, std::abs(-2 * (f - f0) / q));
    std::vector<double> x_prime = x;
    for (auto& val : x_prime) {
        val += lambda;
    }// end for

    double f_prime = func(x_prime);
    std::vector<double> g_prime = gradient(func, x_prime, 1e-7);
    double q_prime = dot_product(g_prime, s);

    double z = 3 * (f - f_prime) / lambda + q + q_prime;
    double a =  z * z - q * q_prime;
    double w;
    if (a > 0) {
        w = std::sqrt(z * z - q * q_prime);    
    } else {
        std::cout << "can't take square root";
        w = 0;
    }
    //double w = std::sqrt(z * z - q * q_prime);
    //std::cout <<"\n\nlambda = " << lambda << std::endl;
    //std::cout << "q = " << q << "\nq'= "<<q_prime << "\nz * z - q * q_prime = " << a;
    if (z * z < q * q_prime){// || q > 0 || q_prime < 0) {
        throw std::invalid_argument("Conditions not met for line minimization.");
    }// end if

    double alpha = (z + w - q) / (q_prime + 2 * w - q);
    //std::cout << "\nalpha = " << alpha;
    return alpha;
}// end cubicInterpolationLineSearch

double simple_backtracking(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha, double tau) {
    double fx = func(x);
    std::vector<double> x_alpha_p = x;  // new proposed point in the parameter space

    // Adjust x_alpha_p along the search direction p by the initial step size of alpha
    for (int i = 0; i < x.size(); i++) {
        x_alpha_p[i] += alpha * p[i];
    }// end for

    // If the function value at the new proposed point is not less than at the current point,
    //    alpha is reduced by a factor of tau to backtrack along the search direction p
    while (func(x_alpha_p) >= fx) {
        alpha *= tau;

        // Adjust x_alpha_p to the new value of alpha
        for (int i = 0; i < x.size(); i++) {
            x_alpha_p[i] = x[i] + alpha * p[i];
        }// end for
    }// end while
    return alpha;
}

/* An attempt to Implementation of MnLineSearch */
double mnLineSearch(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double gdel = 0.0) {
    double overal = 1000., toler = 0.05, slamin = 0., slambg = 5.;
    //double undral = -100., alpha = 2.;
    int maxiter = 12, niter = 1;
    
    for (unsigned int i = 0; i < p.size(); i++) {
        if (p[i] == 0)
            continue;
        double ratio = std::fabs(x[i] / p[i]);
        if (slamin == 0)
            slamin = ratio;
        if (ratio < slamin)
            slamin = ratio;
    }

    double f0 = func(x);

    std::vector<double> modified_vec(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        modified_vec[i] = x[i] + p[i];
    }
    double f1 = func(modified_vec);
    double fvmin = f0;
    double xvmin = 0.;

    if (f1 < f0) {
        fvmin = f1;
        xvmin = 1.;
    }
    double toler8 = toler, slamax = slambg, flast = f1, slam = 1.;

    do {
        double denom = 2. * (flast - f0 - gdel * slam) / (slam * slam);
        if (denom != 0) {
            slam = -gdel / denom;
        } else {
            denom = -0.1 * gdel;
            slam = 1.;
        }

        if (slam > slamax)
            slam = slamax;
        if (slam < toler8)
            slam = toler8;
        if (slam < slamin)
            return xvmin;
        if (std::fabs(slam - 1.) < toler8)
            slam = 1. + toler8;
        
        std::vector<double> vec(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            vec[i] = x[i] + slam * p[i];
        }
        double f2 = func(vec);
       //double f2 = func(x + slam * p);
        niter++;

        if (f2 < fvmin) {
            fvmin = f2;
            xvmin = slam;
        }
        
        if (std::fabs(f0 - fvmin) < 1e-8) {
            flast = f2;
            toler8 = toler * slam;
            overal = slam - toler8;
            slamax = overal;
            f0 = f1;
            f1 = f2;
        }
    } while (niter < maxiter);

    return xvmin;
}

double quadratic_line_search(std::function<double(std::vector<double> &)> func,
                             std::vector<double> x,
                             std::vector<double> p,
                             double alpha = 1.0,
                             double c = 0.5,
                             double tau = 0.5,
                             int maxiter = 12) {
    // Constants for the search
    const double SLAMBG = 5.;
    //const double ALPHA = 2.;
    const double TOLER = 0.05;

    int niter = 0;
    double slam = 1.;
    double toler8 = TOLER;
    double slamax = SLAMBG;
    double fvmin = func(x);
    double xvmin = 0.;

    while (niter < maxiter) {
        std::vector<double> new_point = x;
        for (size_t i = 0; i < x.size(); i++) {
            new_point[i] += slam * p[i];
        }

        double f_new = func(new_point);

        if (f_new < fvmin) {
            fvmin = f_new;
            xvmin = slam;
        }

        // Check the Armijo condition
        if (f_new > fvmin + c * slam * dot_product(gradient(func, x, 1e-7), p)) {
            slam *= tau;
        } else {
            // If it's not met, use the quadratic model to update slam
            double a = (f_new - fvmin) / (slam * slam);
            slam = -dot_product(gradient(func, x, 1e-7), p) / (2 * a);
        }

        // Enforce step size bounds
        if (slam < toler8) slam = toler8;
        if (slam > slamax) slam = slamax;

        niter++;
    }

    return xvmin;
}

double safe_divide(double numerator, double denominator, double default_value = std::numeric_limits<double>::max()) {
    if (std::abs(denominator) < 1e-10) {
        return default_value;
    }
    return numerator / denominator;
}

double armijoCurvature(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, std::vector<double> grad, double alpha, double tau) {
    double c1 = 1e-3;
    double c2 = 0.1;
    double fx = func(x);
    std::vector<double> x_alpha_p = x;  // new proposed point in the parameter space

    // Adjust x_alpha_p along the search direction p by the initial step size of alpha
    for (int i = 0; i < x.size(); i++) {
        x_alpha_p[i] += alpha * p[i];
    }// end for

    // While the Armijo condition or the Curvature condition is not met,
    //    alpha is reduced by a factor of tau to backtrack along the search direction p
    while (func(x_alpha_p) > fx + c1 * alpha * dot_product(grad, p) || dot_product(grad, p) < c2 * dot_product(grad, p)) {
        alpha *= tau;

        // Adjust x_alpha_p to the new value of alpha
        for (int i = 0; i < x.size(); i++) {
            x_alpha_p[i] = x[i] + alpha * p[i];
        }// end for
    }// end while
    return alpha;
}// end armijoCurvature


/* The Broyden–Fletcher–Goldfarb–Shanno update */
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

double optimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, bool use_bfgs, double tol = 1e-8, int max_iter = 2500) {
    double min_value = std::numeric_limits<double>::max();
    std::vector<double> x = x0;

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
        if (norm(g) < 1e-10) { 
            min_value = std::min(min_value, func(x));
            if (min_value < global_min) {
                global_min = min_value;
                std::cout << "\nnorm(g): New Global Minimum: " << global_min << " with parameters:" <<std::endl;
                best_params = {};
                for (int i=0;i<=x.size();i++){
                    best_params.push_back(x[i]);
                    std::cout<< "x["<<i<<"]: " << best_params[i]<<std::endl;
                }
                return min_value;
            }//end if
            return global_min;
        }// end if

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

        // Armijo + Curvature

        // Cubic interpolation
        //double alpha = cubicInterpolationLineSearch(func, x, p, x[0]);

        // Quadratic interpolation
        double alpha = quadratic_line_search(func, x, p);

        // Quadratic interpolation same as MnLineSearch in Minuit2
        //double alpha = mnLineSearch(func, x, p);
        //std::cout << "\nalpha = "<<alpha;

        // Update the current point x by taking a step of size alpha in the direction p.
        std::vector<double> x_new = x;
        //std::cout << "x_new loop" << std::endl;
        for (int j = 0; j < x.size(); j++) {
            x_new[j] += alpha * p[j];
            //std::cout << x_new[j] << " " << " SD = " << p[j] << " ";
        }// end for


        // Compute the difference between the new point and the old point, delta_x
        std::vector<double> delta_x = x_new;
        for (int j = 0; j < x.size(); j++) {
            delta_x[j] -= x[j];
        }// end for

        // Compute the difference in the gradient at the new point and the old point, delta_g.
        std::vector<double> delta_g = gradient(func, x_new, 1e-7);
        for (int j = 0; j < g.size(); j++) {
            delta_g[j] -= g[j];
        }// end for

        //
        double delta_dot = dot_product(delta_x, delta_g);

        if (use_bfgs) {
            // Update the inverse Hessian approximation using BFGS
            bfgs_update(H, delta_x, delta_g, delta_dot);
        } else {
            // Update the approximation of the inverse Hessian using DFP
            dfp_update(H, delta_x, delta_g);
        }
        /* bool valid = false;
        for(auto& param : x_new) {
            if(abs(param) > 512 || abs(std::min(min_value, func(x))) > 960) {
                valid = false;
            } else {
                valid = true;
            }
        }
        if(valid == true) {
            x = x_new;
        }*/
        x = x_new;
        min_value = std::min(min_value, func(x));
    }// end main loop

    //std::cout << "Maximum iterations reached without convergence.\nOptimized parameters:\n";
    //for (double param : x) {
    //    std::cout << param << "\n";
    //}// end for
    return min_value; 
}// end dfp


double minimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string name, 
              int pop_size, int max_gens, int dim, bool use_bfgs) {
                
    global_min = std::numeric_limits<double>::max();
    auto start = std::chrono::high_resolution_clock::now();
    auto final_population = genetic_algo(func, max_gens, pop_size, dim, x0, use_bfgs);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\ntime: " << duration.count() << " ms" << std::endl;
    std::cout << "Predicted Global minimum for " << name << " = " << global_min << "\nOptimized Parameters:" <<std::endl;
    for (int i = 0; i < best_params.size(); i++) {
        std::cout << "x" << i  << ": "<< best_params[i] << "\n";
    }
    return global_min;
}
