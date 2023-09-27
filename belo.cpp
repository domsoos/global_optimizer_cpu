
double optimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string algorithm, double tol = 1e-8, int max_iter = 2500) {
    double min_value = std::numeric_limits<double>::max();
    std::vector<double> x = x0;
    std::vector<double> x_new;

    // L-BFGS-B init
    std::deque<std::vector<double>> s_history, y_history;
    std::deque<double> rho_history;
    double gamma_k = 1.0;  // initial scaling factor
    int m = 10;  // history size

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
        if(algorithm == "lbfgsb") {
            std::pair<std::vector<double>, std::vector<double>> result = lbfgsb_step(func,x, g, s_history, y_history, rho_history, gamma_k, m);
            //auto [x_new, g_new] = lbfgsb_step(func, x, g, s_history, y_history, rho_history, gamma_k, m);
            std::vector<double> x_new = result.first;
            std::vector<double> g_new = result.second;
            x = x_new;
            g = g_new;
        } else {
            // Compute Search Direction
            std::vector<double> p = matvec_product(H, g);
            for (auto &val : p) {
                val = -val; // opposite of greatest increase
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
            for (int j = 0; j < x.size(); j++) {
                x_new[j] += alpha * p[j];
            }// end for

            if(algorithm == "bfgs") {
                double delta_dot = dot_product(delta_x, delta_g);
                // Update the inverse Hessian approximation using BFGS
                bfgs_update(H, delta_x, delta_g, delta_dot);
            } else {
                // Update the approximation of the inverse Hessian using DFP
                dfp_update(H, delta_x, delta_g);
            }
            x = x_new;
            min_value = std::min(min_value, func(x));
        }//end else-if

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
        //double alpha = quadratic_line_search(func, x, p);

        // Quadratic interpolation same as MnLineSearch in Minuit2
        //double alpha = mnLineSearch(func, x, p);
        //std::cout << "\nalpha = "<<alpha;

        // Update the current point x by taking a step of size alpha in the direction p.
        //std::vector<double> x_new = x;
        //std::cout << "x_new loop" << std::endl;
        //for (int j = 0; j < x.size(); j++) {
        //    x_new[j] += alpha * p[j];
        //    //std::cout << x_new[j] << " " << " SD = " << p[j] << " ";
        //}// end for


        //if (use_bfgs) {
        //    // Update the inverse Hessian approximation using BFGS
        //    bfgs_update(H, delta_x, delta_g, delta_dot);
        //} else {
        //    // Update the approximation of the inverse Hessian using DFP
        //    dfp_update(H, delta_x, delta_g);
        //}
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
        //x = x_new;
        //min_value = std::min(min_value, func(x));
    }// end main loop

    //std::cout << "Maximum iterations reached without convergence.\nOptimized parameters:\n";
    //for (double param : x) {
    //    std::cout << param << "\n";
    //}// end for
    return min_value; 
}// end dfp