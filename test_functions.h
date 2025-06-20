#pragma once
#include <vector>
#include "dual.h"
#include "fun.h"

template<int DIM>
dual::DualNumber rosenbrock_ad(const dual::DualNumber* x);


double rosenbrock(std::vector<double>& x);
double rosenbrock_multi(std::vector<double>& x);
double rastrigin(std::vector<double>& x);
double ackley(std::vector<double>& x);
double eggholder(std::vector<double>& x);
double goldstein_price(std::vector<double>& x);
double woods(std::vector<double>& x);
double powell_quartic(std::vector<double>& x);
double helical_valley(std::vector<double>& x);
double fletcher_powell_trig(std::vector<double>& x0);
double randomValue(double lower, double upper);
double thermister(std::vector<double>& x);
double two_exponentials(std::vector<double>& x);
double chemical_equilibrium(std::vector<double>& x);
double heat_conduction(std::vector<double>& x);
