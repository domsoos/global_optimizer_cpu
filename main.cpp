#include "utility.h"
#include "test_functions.h"
#include "optimization.h"

int main() {
    int pop_size, max_gens;
    bool use_bfgs;
    char answer; 
    std::cout << "\nEnter population size: ";
    std::cin >> pop_size;
    std::cout << "Enter maximum generations: ";  
    std::cin >> max_gens;

    std::cout << "Would you like to use BFGS? (Y/N) ";
    std::cin >> answer;
    if (std::tolower(answer) == 'y') {use_bfgs = true;}
    else { use_bfgs = false;}

    double minimum, error;

    /**/

    std::vector<double> ra0 = {3.0, 3.0};
    std::cout << "\n\nrastrigin(1.0, 2.0) = " << rastrigin(ra0);
    minimum = minimize(rastrigin, ra0, "Rastrigin", pop_size, max_gens, 4, use_bfgs);
    error = fabs(minimum);
    std::cout << "\nGlobal Minimum(0.0, 0.0) = 0\n Error = " <<error;

    std::vector<double> hv0 = {-1.0, 0.0, 0.0};
    std::cout << "\n\nhelical_valley(-1.0, 0.0, 0.0) = " << helical_valley(hv0);
    minimum= minimize(helical_valley, hv0, "Helical Valley", pop_size, max_gens, 3, use_bfgs);
    error = fabs(minimum);
    std::cout << "\nGlobal Minimum(1.0, 0.0, 0.0) = 0\n Error = "<<error;


    /*
    std::vector<double> r0 = {-1.2, 2.0};
    std::cout << "\n\nrosenbrock(-1.2, 2.0) = " << rosenbrock(r0);
    minimum = minimize(rosenbrock, r0, "Rosenbrock", pop_size, max_gens, 2, use_bfgs);
    error = fabs(minimum);
    std::cout << "\nGlobal Minimum(1.0, 1.0) = 0\n Error = " <<error;

    std::vector<double> gsp = {1.0, 1.0};
    std::cout << "\n\ngoldstein-price(0.0, -1.0) = " << goldstein_price(gsp);
    minimum = minimize(goldstein_price, gsp, "Gold-Stein Price", pop_size, max_gens, 2, use_bfgs);
    error = 3 - minimum;
    std::cout << "\nGlobal Minimum(0.0, -1.0) = 3\nError = "<<error;


    std::vector<double> easy = {1.1, 1.2, 1.3, 1.4};
    std::vector<double> w0 = {-3.0, -1.0, -3.0, -1.0};
    std::cout << "\n\nwoord(-3.0, -1.0, -3.0, -1.0) = " << woods(w0);
    minimize(woods, w0, "Woods", pop_size, max_gens, 4, use_bfgs);

    std::vector<double> p0= {-3.0, -1.0, 0.0, -1.0};
    std::cout << "\n\npowell(x0) = " << powell_quartic(p0);
    minimize(powell_quartic, p0, "Powell Quartic", pop_size, max_gens, 4, use_bfgs);

    std::vector<double> fp0 = {1.0, -1.0}; // ?? 
    std::cout << "\n\nfletcher_powell_trig(x0) = " << fletcher_powell_trig(fp0);
    minimize(fletcher_powell_trig, fp0, "Fletcher Powell Trig", pop_size, max_gens, 2, use_bfgs);

    std::vector<double> t0 = {0.02, 4000.0, 250.0};
    std::cout << "\n\nthermister(x0) = " << thermister(t0);
    minimize(thermister, t0, "Thermister", pop_size, max_gens, 3, use_bfgs);

    std::vector<double> s0 = {0.0, 20.0};
    std::cout << "\n\nstwo_exponentials(x0) = " << two_exponentials(s0);
    minimize(two_exponentials, s0, "Sum of Two Exponentials", pop_size, max_gens, 2, use_bfgs);

    std::vector<double> c0 = {0.1, 0.1, 0.1};
    std::cout << "\n\nchemical_equilibrium(x0) = " << chemical_equilibrium(c0);
    minimize(chemical_equilibrium,c0,"Chemical Equilibrium", pop_size, max_gens, 3, use_bfgs);

    std::vector<double> h0 = {4.7, 6.1, 6.5, 8.0};
    std::cout << "\n\nheat_conduction(x0) = " << heat_conduction(h0);
    minimize(heat_conduction, h0, "Heat Conduction", pop_size, max_gens, 4, use_bfgs);
    */

    return 0;
}