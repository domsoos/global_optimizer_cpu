#pragma once
#include <deque>
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

std::pair<std::vector<double>, std::vector<double>> lbfgsb_step(std::function<double(std::vector<double> &)> func,
    const std::vector<double>& x, const std::vector<double>& g,
    const std::pair<std::vector<double>, std::vector<double>>& bounds, // bounds.first = lower else upper
    std::deque<std::vector<double>>& s_history, std::deque<std::vector<double>>& y_history, // histories
    std::deque<double>& rho_history, double& gamma_k, int m); // m = history size, gamma_k = scaling factor
std::vector<double> lbfgsb_update(const std::vector<double> &g, std::deque<std::vector<double>> &s_history, 
    std::deque<std::vector<double>> &y_history, std::deque<double> &rho_history,
    double gamma_k, int m);

void bfgs_update(std::vector<std::vector<double>>& H, std::vector<double> delta_x, std::vector<double> delta_g, double delta_dot);
void dfp_update(std::vector<std::vector<double>>& H, std::vector<double> delta_x, std::vector<double> delta_g);

double optimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string algorithm, double tol, int max_iter, std::pair<std::vector<double>, std::vector<double>> bounds);
long minimize(std::function<double(std::vector<double> &)> func, std::vector<double> x0, std::string name, int pop_size, int max_gens, int dim, std::string algorithm, std::pair<std::vector<double>, std::vector<double>> bounds);