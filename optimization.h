#pragma once
#include "utility.h"
#include "test_functions.h"
#include "genetic.h"

// Line Search methods
double simple_backtracking(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha, double tau);
double line_search_simple(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha, double tau);
double line_search(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha, double c, double tau);
double cubicInterpolationLineSearch(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> s, double f0);
double mnLineSearch(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double gdel);
double quadratic_line_search(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, double alpha, double c, double tau, int maxiter);
double safe_divide(double numerator, double denominator, double default_value);
double armijoCurvature(std::function<double(std::vector<double> &)> func, std::vector<double> x, std::vector<double> p, std::vector<double> grad, double alpha, double tau);

void bfgs_update(std::vector<std::vector<double>>& H, std::vector<double> delta_x, std::vector<double> delta_g, double delta_dot);
void dfp_update(std::vector<std::vector<double>>& H, std::vector<double> delta_x, std::vector<double> delta_g);
double optimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, bool use_bfgs, double tol, int max_iter);
double minimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string name, int pop_size, int max_gens, int dim, bool use_bfgs);