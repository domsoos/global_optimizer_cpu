#include "utility.h"
#include "test_functions.h"
#include "optimization.h"

#include <sys/resource.h>
#include <fstream>


int main() {
    int pop_size, max_gens;
    long before, after, result;
    double error;
    char answer, going;
    bool done = false;

    double lower, upper;
    std::vector<double> lower_bounds, upper_bounds;
    std::pair<std::vector<double>, std::vector<double>> bounds;

    std::string algorithm;
    std::ofstream rast("memory_rastrigin5.csv");
    std::ofstream heli("memory_helical5.csv");
    rast << "Algorithm,Time,Error,MemoryKB,MemoryMB\n";
    heli << "Algorithm,Time,Error,MemoryKB,MemoryMB\n";

    std::cout << "\nEnter population size: ";
    std::cin >> pop_size;
    std::cout << "Enter maximum generations: ";  
    std::cin >> max_gens;

    while(!done) {
        std::cout << "Use BFGS? ";
        std::cin >> answer;
        if (std::tolower(answer) == 'y') {
            char lbfgsb;
            std::cout << "Limited memory variant - L-BFGS-B? ";
            std::cin >> lbfgsb;
            if (std::tolower(lbfgsb) == 'y') {
                algorithm = "lbfgsb";
            } else {
                algorithm = "bfgs";
            }
        }
        else { algorithm = "dfp";}

        before = measure_memory();
        std::vector<double> ra0 = {3.0, 3.0};
        if (algorithm  == "lbfgsb"){
            //for (int i=0; i<dim; ++i) {
            //    std::cout << "\nEnter the lower bound for dimension " << i + 1 << ": ";
            //    std::cin >> lower_bounds[i];  
            //    std::cout << "\nEnter the upper bound for dimension " << i + 1 << ": ";
            //    std::cin >> upper_bounds[i];
            //}
            //bounds = {lower_bounds, upper_bounds};
            std::cout<<"Enter lower bound: ";
            std::cin>>lower;
            std::cout<<"Enter upper bound: ";
            std::cin>>upper;
            for(int i=0;i<2;i++){ // generate bound for each dimension
                bounds.first.push_back(lower);
                bounds.second.push_back(upper);
            }//end for
        }// end if lbfgsb
        std::cout << "\n\nrastrigin(3.0, 3.0) = " << rastrigin(ra0);
        auto time = minimize(rastrigin, ra0, "Rastrigin", pop_size, max_gens, 2, algorithm, bounds);
        after = measure_memory();
        result = after - before;
        error = fabs(global_min);
        std::cout << "Memory usage during Rastrigin Optimization: " << result << " KB";
        rast << algorithm << "," << time << "," << error << "," << result << "," << result/1024 << "\n";
        std::cout << "\nGlobal Minimum(0.0, 0.0) = 0\n Error = " <<error;

        if (algorithm  == "lbfgsb"){
            //for (int i=0; i<dim; ++i) {
            //    std::cout << "\nEnter the lower bound for dimension " << i + 1 << ": ";
            //    std::cin >> lower_bounds[i];  
            //    std::cout << "\nEnter the upper bound for dimension " << i + 1 << ": ";
            //    std::cin >> upper_bounds[i];
            //}
            //bounds = {lower_bounds, upper_bounds};
            std::cout<<"\n\nEnter lower bound: ";
            std::cin>>lower;
            std::cout<<"Enter upper bound: ";
            std::cin>>upper;
            for(int i=0;i<3;i++){// generate bound for each dimension
                bounds.first.push_back(lower);
                bounds.second.push_back(upper);
            }//end for
        }// end if lbfgsb
        before = measure_memory();
        std::vector<double> hv0 = {-1.0, 0.0, 0.0};
        std::cout << "\n\nhelical_valley(-1.0, 0.0, 0.0) = " << helical_valley(hv0);
        time = minimize(helical_valley, hv0, "Helical Valley", pop_size, max_gens, 3, algorithm, bounds);
        after = measure_memory();
        result = (after - before)/1024;
        std::cout << "Memory usage during Helical Valley Optimization: " << result << " KB";

        heli << algorithm << "," << time << "," << error << "," << result << "," << result/1024 << "\n";

        error = fabs(global_min);
        std::cout << "\nGlobal Minimum(1.0, 0.0, 0.0) = 0\n Error = "<<error<< "\n" << std::endl;

        std::cout << "\n\nKeep going? ";
        std::cin >> going;
        if(std::tolower(going) != 'y'){
            std::cout << "Goodbye!"<<std::endl;
            done = true;
        }
    }//end while
    rast.close();
    heli.close();
    return 0;
    /*
    std::vector<double> r0 = {-1.2, 2.0};
    std::cout << "\n\nrosenbrock(-1.2, 2.0) = " << rosenbrock(r0);
    minimum = minimize(rosenbrock, r0, "Rosenbrock", pop_size, max_gens, 2, algorithm);
    error = fabs(minimum);
    std::cout << "\nGlobal Minimum(1.0, 1.0) = 0\n Error = " <<error;

    std::vector<double> gsp = {1.0, 1.0};
    std::cout << "\n\ngoldstein-price(0.0, -1.0) = " << goldstein_price(gsp);
    minimum = minimize(goldstein_price, gsp, "Gold-Stein Price", pop_size, max_gens, 2, algorithm);
    error = 3 - minimum;
    std::cout << "\nGlobal Minimum(0.0, -1.0) = 3\nError = "<<error;


    std::vector<double> easy = {1.1, 1.2, 1.3, 1.4};
    std::vector<double> w0 = {-3.0, -1.0, -3.0, -1.0};
    std::cout << "\n\nwoord(-3.0, -1.0, -3.0, -1.0) = " << woods(w0);
    minimize(woods, w0, "Woods", pop_size, max_gens, 4, algorithm);

    std::vector<double> p0= {-3.0, -1.0, 0.0, -1.0};
    std::cout << "\n\npowell(x0) = " << powell_quartic(p0);
    minimize(powell_quartic, p0, "Powell Quartic", pop_size, max_gens, 4, algorithm);

    std::vector<double> fp0 = {1.0, -1.0}; // ?? 
    std::cout << "\n\nfletcher_powell_trig(x0) = " << fletcher_powell_trig(fp0);
    minimize(fletcher_powell_trig, fp0, "Fletcher Powell Trig", pop_size, max_gens, 2, algorithm);

    std::vector<double> t0 = {0.02, 4000.0, 250.0};
    std::cout << "\n\nthermister(x0) = " << thermister(t0);
    minimize(thermister, t0, "Thermister", pop_size, max_gens, 3, algorithm);

    std::vector<double> s0 = {0.0, 20.0};
    std::cout << "\n\nstwo_exponentials(x0) = " << two_exponentials(s0);
    minimize(two_exponentials, s0, "Sum of Two Exponentials", pop_size, max_gens, 2, algorithm);

    std::vector<double> c0 = {0.1, 0.1, 0.1};
    std::cout << "\n\nchemical_equilibrium(x0) = " << chemical_equilibrium(c0);
    minimize(chemical_equilibrium,c0,"Chemical Equilibrium", pop_size, max_gens, 3, algorithm);

    std::vector<double> h0 = {4.7, 6.1, 6.5, 8.0};
    std::cout << "\n\nheat_conduction(x0) = " << heat_conduction(h0);
    minimize(heat_conduction, h0, "Heat Conduction", pop_size, max_gens, 4, algorithm);
    */
    //return 0;
}