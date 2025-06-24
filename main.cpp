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


int main(int argc, char* argv[]) {
	if (argc != 10) {
	 std::cerr << "Usage: " << argv[0] << " <lower_bound> <upper_bound> <max_iter> <pso_iters> <converged> <number_of_optimizations> <tolerance> <seed> <run>\n";
        return 1;
    }
    double lower = std::atof(argv[1]);
    double upper = std::atof(argv[2]);   	
    int bfgs_iter = std::stoi(argv[3]);
    int pso_iters = std::stoi(argv[4]);
    int requiredConverged = std::stoi(argv[5]);
    int swarm_size = std::stoi(argv[6]);
    double tolerance = std::stod(argv[7]);
    int seed = std::stoi(argv[8]);
    int run = std::stoi(argv[9]);

	std::cout << "Which function? [rosenbrock|rastrigin|ackley|goldstein]: ";
    std::string fname; std::cin >> fname;
    std::cout << "Dimension: "; int dim; std::cin >> dim;
    ADFunc f_ad = makeTestFunc(fname, dim);

    long before, after, memory;
    char going;
    bool done = false;

    std::string algorithm = "bfgs";

    std::cout << "\n\nMultidimensional Optimization" << std::endl;

    while(!done) {
        before = measure_memory();
        Result result = run_minimizers(f_ad,fname,pso_iters,bfgs_iter,swarm_size,dim,seed, requiredConverged, tolerance, algorithm, lower, upper, run);
        //  rosenbrock_multi, x0,algorithm,1e-12,2500,bounds);
        after = measure_memory();
        memory = after - before;
        //error = result.fval;
        //error = util::calculate_euclidean(result.coordinates, fname);
        std::cout << "Memory usage during " << dim  <<  "D " << fname <<" Optimization: " << memory << " KB";
        //long mb = memory/1024;

        std::cout << "\n\nKeep going? ";
        std::cin >> going;
        if(std::tolower(going) != 'y'){
            std::cout << "Goodbye!"<<std::endl;
            done = true;
        }
    }//end while
    return 0;
}