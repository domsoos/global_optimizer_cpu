#pragma once
#include <deque>
#include "utility.h"
#include "test_functions.h"
#include "genetic.h"
#include "ad_gradient.h"

struct Result {
	int idx;
	long time;
	//int points_we_need;
    int status; // 1 if converged, else if stopped_bc_someone_flipped_the_flag: 2, else 0
    double fval; // function value
    double gradientNorm;
    std::vector<double> coordinates;
    int iter;
};

using RealFunc = std::function<double(const std::vector<double>&)>;

// Line Search method
double line_search(
    const std::function<double(const std::vector<double>&)>& func,
    double f0,
    const std::vector<double>& x,
    const std::vector<double>& p,
    const std::vector<double>& g
);


void bfgs_update_seq(std::vector<std::vector<double>>& H, std::vector<double> delta_x, std::vector<double> delta_g, double delta_dot);
void dfp_update(std::vector<std::vector<double>>& H, std::vector<double> delta_x, std::vector<double> delta_g);

//double optimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string algorithm, double tol, int max_iter, std::pair<std::vector<double>, std::vector<double>> bounds);
//long minimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string name, int pop_size, int max_gens, int dim, std::string algorithm, std::pair<std::vector<double>, std::vector<double>> bounds);
Result optimize(
    const ADFunc &f_ad,
    std::vector<double> x0,
    const std::string& algorithm,
    double tolerance,
    int max_iter
);

Result run_minimizers(
    const ADFunc &f_ad,
    std::string const& name, int pso_iter,int bfgs_iter,
    int pop_size,
    int dim,int seed,
    int converged, double tolerance,
    std::string const& algorithm,double lower, double upper, const int run
);