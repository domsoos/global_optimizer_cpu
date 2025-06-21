#include "utility.h"
#include "test_functions.h"
#include "optimization.h"
#include "dual.h"
#include "fun.h"

#include <sys/resource.h>
#include <fstream>
#include <sstream>

ADFunc makeTestFunc(const std::string &name, int dim) {
    using util::Rosenbrock;
    using util::Rastrigin;
    using util::Ackley;
    using util::GoldsteinPrice;

    if (name == "rosenbrock") {
        switch (dim) {
          case 2:  return [](const std::vector<dual::DualNumber>& vx){ return Rosenbrock<2>::evaluate(vx.data()); };
          case 5:  return [](const std::vector<dual::DualNumber>& vx){ return Rosenbrock<5>::evaluate(vx.data()); };
          case 10: return [](const std::vector<dual::DualNumber>& vx){ return Rosenbrock<10>::evaluate(vx.data()); };
          default: throw std::runtime_error("rosenbrock supports dims 2,5,10");
        }
    }
    if (name == "rastrigin") {
        switch (dim) {
          case 2:  return [](const std::vector<dual::DualNumber>& vx){ return Rastrigin<2>::evaluate(vx.data()); };
          case 5:  return [](const std::vector<dual::DualNumber>& vx){ return Rastrigin<5>::evaluate(vx.data()); };
          case 10: return [](const std::vector<dual::DualNumber>& vx){ return Rastrigin<10>::evaluate(vx.data()); };
          default: throw std::runtime_error("rastrigin supports dims 2,5,10");
        }
    }
    if (name == "ackley") {
        switch (dim) {
          case 2:  return [](const std::vector<dual::DualNumber>& vx){ return Ackley<2>::evaluate(vx.data()); };
          case 5:  return [](const std::vector<dual::DualNumber>& vx){ return Ackley<5>::evaluate(vx.data()); };
          case 10: return [](const std::vector<dual::DualNumber>& vx){ return Ackley<10>::evaluate(vx.data()); };
          default: throw std::runtime_error("ackley supports dims 2,5,10");
        }
    }
    if (name == "goldstein") {
        if (dim != 2) throw std::runtime_error("Goldstein–Price only 2-D");
        return [](const std::vector<dual::DualNumber>& vx){ return GoldsteinPrice<2>::evaluate(vx.data()); };
    }
    throw std::runtime_error("Unknown function: " + name);
}


int main() {
	std::cout << "Which function? [rosenbrock|rastrigin|ackley|goldstein]: ";
    std::string fname; std::cin >> fname;
    std::cout << "Dimension: "; int dim; std::cin >> dim;
    ADFunc f_ad = makeTestFunc(fname, dim);

    long before, after, memory;
    double error;
    char answer, going;
    bool done = false;

    double lower, upper;
    std::vector<double> lower_bounds, upper_bounds;
    std::pair<std::vector<double>, std::vector<double>> bounds;

    std::string algorithm;

    std::cout << "\n\nMultidimensional Optimization" << std::endl;

    std::ostringstream filenameStream;
    filenameStream << "memory_rosenbrock" << dim << ".csv";
    std::string filename = filenameStream.str();
    std::ofstream rosen(filename);

    //ADFunc f_ad = makeRosenbAd(dim);
    
    if(dim >= 100) {
        rosen << "Algorithm,Time,Error,MemoryMB,MemoryGB\n";
    }else {
        rosen << "Algorithm,Time,Error,MemoryKB,MemoryMB\n";
    }

    while(!done) {
        std::cout << "\nChoose the algorithm"<<std::endl;;
        std::cout << "Use BFGS? ";
        std::cin >> answer;
        if (std::tolower(answer) == 'y') {
            char lbfgs, box;
            std::cout << "Limited memory variant - L-BFGS? ";
            std::cin >> lbfgs;
            if (std::tolower(lbfgs) == 'y') {
                std::cout << "bounded? ";
                std::cin >> box;
                if(std::tolower(box) == 'y') {
                    algorithm = "lbfgsb";
                } else {
                    algorithm = "lbfgs";
                }
            } else {
                algorithm = "bfgs";
            }
        }
        else { algorithm = "dfp";}

        if (algorithm  == "lbfgsb"){
            std::cout<<"Enter lower bound: ";
            std::cin>>lower;
            std::cout<<"Enter upper bound: ";
            std::cin>>upper;
            for(int i=0;i<2;i++){ // generate bound for each dimension
                bounds.first.push_back(lower);
                bounds.second.push_back(upper);
            }//end for
        }// end if lbfgsb

        std::vector<double> x0;
        for(int i=0;i<dim;i++){
            x0.push_back(uniform_rand(-5.0, 5.0));
        }
        before = measure_memory();
        Result result = run_minimizers(f_ad,x0,"Rosenbrock",0,dim, algorithm, bounds);
        //  rosenbrock_multi, x0,algorithm,1e-12,2500,bounds);
        after = measure_memory();
        memory = after - before;
        error = result.fval;

        std::cout << "Memory usage during " << dim  <<  "D Rosenbrock Optimization: " << memory << " KB";
        long mb = memory/1024;
        if(dim >= 100) {
            rosen << algorithm << "," << result.time << "," << error << "," << mb << "," << mb/1024 << "\n";
        }else {
            rosen << algorithm << "," << result.time << "," << error << "," << memory << "," << mb << "\n";
        }
        //rosen << algorithm << "," << time << "," << error << "," << result << "," << result/1024 << "\n";
        std::cout << "\nGlobal Minimum(1.0, 1.0,...,1.0) = 0\n Error = " <<error;

        std::cout << "\n\nKeep going? ";
        std::cin >> going;
        if(std::tolower(going) != 'y'){
            std::cout << "Goodbye!"<<std::endl;
            done = true;
        }
    }//end while
    rosen.close();
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