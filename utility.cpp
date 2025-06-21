#include "utility.h"

std::random_device rd;
std::mt19937      rng(rd());

double uniform_rand(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

long measure_memory() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return r_usage.ru_maxrss;
}

namespace util {
	double calculate_euclidean(std::vector<double> coordinates,std::string fname) {

    double sum_sq = 0.0;

    if (fname == "rosenbrock") {
        // global minimizer at x_i = 1 for all i
        for (double xi : coordinates) {
            double d = xi - 1.0;
            sum_sq += d * d;
        }
    } else if (fname == "goldstein") {
        // Goldstein–Price has its global minimum at (0, -1) in 2D
        if (coordinates.size() < 2)
            throw std::invalid_argument("Goldstein–Price requires at least 2 dims");
        double d0 = coordinates[0] - 0.0;
        double d1 = coordinates[1] - (-1.0);
        sum_sq += d0*d0 + d1*d1;
        // if more dims are passed, assume their minimizers are at 0:
        for (size_t i = 2; i < coordinates.size(); ++i) {
            sum_sq += coordinates[i] * coordinates[i];
        }
    } else if (fname == "rastrigin" || fname == "ackley") {
        // both have global minimizer at the origin
        for (double xi : coordinates) {
            sum_sq += xi * xi;
        }
    } else {
        throw std::invalid_argument("Unknown function name: " + fname);
    }
    return std::sqrt(sum_sq);
}


void append_results_2_tsv(const int dim,const int N, const std::string fun_name,float ms_init, float ms_pso,float ms_opt,float ms_rand, const int max_iter, const int pso_iter,const double error,const double globalMin, std::vector<double> hostCoordinates, const int idx, const int status, const double norm) {
        std::string filename = "zeus_" + std::to_string(dim) + "d_results.tsv";
        std::ofstream outfile(filename, std::ios::app);
        
        bool file_exists = std::filesystem::exists(filename);
        bool file_empty = file_exists ? (std::filesystem::file_size(filename) == 0) : true;
        //std::ofstream outfile(filename, std::ios::app);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // if file is new or empty, let us write the header
        if (file_empty) {
            outfile << "fun\tN\tidx\tstatus\tbfgs_iter\tpso_iter\ttime\terror\tfval\tnorm\t";
            for (int i = 0; i < dim; i++)
                outfile << "\tcoord_" << i;
            outfile << std::endl;
        }// end if file is empty
        
        double time_seconds = std::numeric_limits<double>::infinity();
        if (pso_iter > 0) {
            time_seconds = (ms_init+ms_pso+ms_opt+ms_rand);
            //printf("total time = pso + bfgs = total time = %0.4f ms\n", time_seconds);
        } else {
            time_seconds = (ms_opt+ms_rand);
            //printf("bfgs time = total time = %.4f ms\n", time_seconds);
        }
        outfile << fun_name << "\t" << N << "\t"<<idx<<"\t"<<status <<"\t" << max_iter << "\t" << pso_iter << "\t"
            << time_seconds << "\t"
            << std::scientific << error << "\t" << globalMin << "\t" << norm <<"\t" ;
        for (int i = 0; i < dim; i++) {
            outfile << hostCoordinates[i];
            if (i < dim - 1)
                outfile << "\t";
        }
        outfile << "\n";
        outfile.close();
        //printf("results are saved to %s", filename.c_str());
}// end append_results_2_tsv

}// end of util namespace

double global_min = std::numeric_limits<double>::max();
std::vector<double> best_params;

/*** Utility Functions for Matrix-Vector-Scalar operations  ***/
// Scale a vector
std::vector<double> scale_vector(const std::vector<double> &v1, double scalar) {
    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] * scalar;
    }//end for
    return result;
}
// Vector addition
std::vector<double> add_vectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector sizes do not match for addition");
    }// end if

    std::vector<double> result(v1.size(), 0.0);
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }// end for
    return result;
}// end add_vectors 

// Vector subtraction
std::vector<double> subtract_vectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector sizes do not match for subtraction");
    }// end if

    std::vector<double> result(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }// end for

    return result;
}// end subtract_vectors

// Matrix-vector multiplication
std::vector<double> matvec_product(const std::vector<std::vector<double>>& m, const std::vector<double>& v) {
    int rows = m.size();
    int cols = m[0].size();

    if (cols != v.size()) {
        throw std::runtime_error("Invalid dimensions for matrix-vector multiplication");
    }// end if

    std::vector<double> result(rows, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += m[i][j] * v[j];
        }// end inner for
    }// end outer for
    return result;
} // end matvec_product

// Dot product of two vectors
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
    double result = 0.0;
    for (int i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }// end for
    return result;
}// end dot_product

// Norm of a vector
double norm(const std::vector<double>& v) {
    return std::sqrt(dot_product(v, v));
}// end norm

// Matrix multiplication
std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    int rows1 = m1.size();
    int cols1 = m1[0].size();
    int rows2 = m2.size();
    int cols2 = m2[0].size();

    if (cols1 != rows2) {
        throw std::runtime_error("Invalid dimensions for matrix multiplication");
    }// end if

    std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2, 0));
    // 6 x 6 matrix multiplication
    for (int i = 0; i < rows1; i++) { // for each row in the matrix
        for (int j = 0; j < cols2; j++) { // for each column in the 2nd matrix
            // For each row in the first matrix and column in the second matrix, calculate the dot product of the corresponding row and column
            // The dot product is calculated by multiplying the corresponding elements together and then summing the result
            for (int k = 0; k < cols1; k++) {
                result[i][j] += m1[i][k] * m2[k][j];
            }// end innnner
        }// end inner
    }// end outer
    return result;
}// end matmul

/* Outer product of two vectors
 * The outer product of two vectors is a matrix where each element is the product of an element from the first vector and an element from the second vector.
 */
std::vector<std::vector<double>> outer_product(const std::vector<double>& v1, const std::vector<double>& v2) {
    int size1 = v1.size();
    int size2 = v2.size();

    std::vector<std::vector<double>> result(size1, std::vector<double>(size2, 0));
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            result[i][j] = v1[i] * v2[j];
        }// end inner
    }// end outer
    return result;
}// end outer_product

// Finite difference gradient calculation
std::vector<double> gradientFD(std::function<double(std::vector<double> &)> func, std::vector<double> x, double h) {
    std::vector<double> grad(x.size(), 0);
    for (int i = 0; i < x.size(); i++) {
        double temp = x[i];
        x[i] = temp + h;
        double fp = func(x);
        x[i] = temp - h;
        double fm = func(x);
        grad[i] = (fp - fm) / (2 * h);
        x[i] = temp;
    }// end for
    return grad;
}// end gradient

/*
template<typename Function, int DIM>
void calculateGradientUsingAD(double *x, double *gradient) {
    dual::DualNumber xDual[DIM];

    for (int i = 0; i < DIM; ++i) { // // iterate through each dimension (vairbale)
        xDual[i] = dual::DualNumber(x[i], 0.0);
    }

    // calculate the partial derivative of  each dimension
    for (int i = 0; i < DIM; ++i) {
        xDual[i].dual = 1.0; // derivative w.r.t. dimension i
        dual::DualNumber result = Function::evaluate(xDual); // evaluate the function using AD
        gradient[i] = result.dual; // store derivative
        //printf("\nxDual[%d]: %f, grad[%d]: %f ",i,xDual[i].real,i,gradient[i]);
        xDual[i].dual = 0.0;
    }
}
*/


