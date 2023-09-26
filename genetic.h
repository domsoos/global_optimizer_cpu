#pragma once
#include <iostream>
#include <vector>
#include <functional>

#include "optimization.h"

struct Individual {
    std::vector<double> genes; 
    double fitness;
};

std::vector<Individual> init_population(std::function<double(std::vector<double> &)> func, int dim, std::vector<double> x0, int pop_size, bool use_bfgs);
Individual tournament_selection(std::vector<Individual> population);
std::vector<Individual> crossover(std::function<double(std::vector<double> &)> func, Individual ind1, Individual ind2, bool use_bfgs);
void mutate(Individual &ind);
std::vector<Individual> genetic_algo(std::function<double(std::vector<double> &)> func, int max_gens, int pop_size, int dim, std::vector<double> x0, bool use_bfgs);
