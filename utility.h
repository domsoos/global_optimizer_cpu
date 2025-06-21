#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <filesystem>

#include "dual.h"

long measure_memory();

extern std::random_device rd;
extern std::mt19937      rng; 
double uniform_rand(double min, double max);

namespace util {
	double calculate_euclidean(std::vector<double> coordinates,std::string fname);
    void append_results_2_tsv(const int dim,const int N, const std::string fun_name,float ms_init, float ms_pso,float ms_opt,float ms_rand, const int max_iter, const int pso_iter,const double error,const double globalMin, std::vector<double> hostCoordinates, const int idx, const int status, const double norm);
}

extern double global_min;
extern std::vector<double> best_params;

std::vector<double> scale_vector(const std::vector<double> &v1, double scalar);
std::vector<double> add_vectors(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> subtract_vectors(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> matvec_product(const std::vector<std::vector<double>>& m, const std::vector<double>& v);
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2);
double norm(const std::vector<double>& v);
std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2);
std::vector<std::vector<double>> outer_product(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> gradient(std::function<double(std::vector<double> &)> func, std::vector<double> x, double h);
