#pragma once
#include <iostream>
#include <vector>
#include <functional>

#include "optimization.h"

struct Individual {
    std::vector<double> genes; 
    double fitness;
};

std::vector<Individual> init_population(std::function<double(std::vector<double> &)> func, int dim, std::vector<double> x0, int pop_size, std::string algorithm, std::pair<std::vector<double>,std::vector<double>> bounds);
Individual tournament_selection(std::vector<Individual> population);
std::vector<Individual> crossover(std::function<double(std::vector<double> &)> func, Individual ind1, Individual ind2, std::string algorithm);
void mutate(Individual &ind);
std::vector<Individual> genetic_algo(std::function<double(std::vector<double> &)> func, int max_gens, int pop_size, int dim, std::vector<double> x0, std::string algorithm, std::pair<std::vector<double>,std::vector<double>> bounds);
