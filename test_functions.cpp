#include "test_functions.h"

double square(double x)
{
  return x * x;
}


double rosenbrock(std::vector<double>& x) {
  return square(1 - x[0]) + 100 * square(x[1] - square(x[0]));
}


double rastrigin(std::vector<double>& x) {
  double sum = 0;

  for (int i = 0; i < x.size(); i++) {
    sum += x[i] * x[i] - 10 * cos(2 * M_PI * x[i]); 
  }
  return 10 * x.size() + sum;
}

double ackley(std::vector<double>& x) {
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < x.size(); i++) {
        sum1 += x[i] * x[i];
        sum2 += cos(2 * M_PI * x[i]);
    }
    return -20 * exp(-0.2 * sqrt(sum1 / x.size())) - exp(sum2 / x.size()) + 20 + M_E; 
}

double eggholder(std::vector<double>& x) {
  return -x[1] * sin(sqrt(abs(x[0] + x[1] + 47))) 
         - x[0] * sin(sqrt(abs(x[0] - (x[1] + 47))));
}

double goldstein_price(std::vector<double>& x) {
    double x1 = x[0];
    double x2 = x[1];

    double term1 = (1 + pow(x1 + x2 + 1, 2) * (19 - 14*x1 + 3*pow(x1, 2) - 14*x2 + 6*x1*x2 + 3*pow(x2, 2)));
    double term2 = (30 + pow(2*x1 - 3*x2, 2) * (18 - 32*x1 + 12*pow(x1, 2) + 48*x2 - 36*x1*x2 + 27*pow(x2, 2)));

    return term1 * term2;
}

// Woods Function
double woods(std::vector<double>& x) {
  double x1 = x[0];
  double x2 = x[1];
  double x3 = x[2];
  double x4 = x[3];
  
  double term1 = 100*square(square(x1) - x2);
  double term2 = square(x1 - 1) + square(x3 - 1);
  double term3 = 90*square(square(x3) - x4); 
  double term4 = 10.1*(square(x2 - 1) + square(x4 - 1));
  double term5 = 19.8*(x2 - 1)*(x4 - 1);

  return term1 + term2 + term3 + term4 + term5;
}

// Powell's Quartic Function
double powell_quartic(std::vector<double>& x) {

  double x1 = x[0];
  double x2 = x[1];
  double x3 = x[2];
  double x4 = x[3];

  double term1 = pow(x1 + 10*x2, 2);
  double term2 = 5*pow(x3 - x4, 2);
  double term3 = pow(x2 - 2*x3, 4);
  double term4 = 10*pow(x1 - x4, 4);

  return term1 + term2 + term3 + term4;
}


// Fletcher and Powell 3 Variable Helical Valley
double helical_valley(std::vector<double>& x) {
    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];

    const double pi = M_PI;
    double theta;

    if (x1 > 0) {
        theta = (1/2*pi) * atan2(x2, x1); 
    } else {
        theta = (1/2*pi) * atan2(x2, x1) + 0.5;
    }

    double term1 = 100 * pow(x3 - 10*theta, 2);
    double term2 = pow(sqrt(x1*x1 + x2*x2) - 1, 2);

    return term1 + term2 + x3*x3;
}


// Fletcher - Powell Trigonometric function
double fletcher_powell_trig(std::vector<double>& x0){
    int n = 5 + rand() % 71; // Random n between 5 and 75

    // Initialize x, a, b with random values
    std::vector<double> x(n);
    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    std::vector<std::vector<double>> b(n, std::vector<double>(n));

    for(int i = 0; i < n; i++) {
        // Random value between -pi and pi
        x[i] = -M_PI + (2 * M_PI * (rand() / (double)RAND_MAX));
        for(int j = 0; j < n; j++) {
            // Random values between -100 and 100
            a[i][j] = -100.0 + (200.0 * (rand() / (double)RAND_MAX));
            b[i][j] = -100.0 + (200.0 * (rand() / (double)RAND_MAX));
        }
    }

    n = x.size();
    double sum = 0.0;
    for(int i = 0; i < n; i++) {
        double e_i = 0.0;
        double inner_sum = 0.0;
        for(int j = 0; j < n; j++) {
            double value = a[i][j] * sin(x[j]) + b[i][j] * cos(x[j]);
            e_i += value;
            inner_sum += value;
        }
        sum += (e_i - inner_sum) * (e_i - inner_sum);
    }
    return sum;
}

double randomValue(double lower, double upper) {
    return lower + (upper - lower) * (rand() / (double)RAND_MAX);
}

double thermister(std::vector<double>& x) {
    const int n = 16;
    
    // Initialize y_hat, T with random values ???
    std::vector<double> y_hat(n);
    std::vector<double> T(n);
    for(int i = 0; i < n; i++) {
        y_hat[i] = randomValue(0.0, 1.0);
        T[i] = randomValue(0.0, 1.0);
    }

    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];
    
    double sum = 0.0;

    for(int i = 0; i < n; i++) {
        double y_i = x1 * exp(x2 / (T[i] + x3));
        sum += (y_i - y_hat[i]) * (y_i - y_hat[i]);
    }

    return sum;
}

// Sum of Two Exponentials
double two_exponentials(std::vector<double>& x) {

  double x1 = x[0]; 
  double x2 = x[1];

  double sum = 0.0;

  for(int i=1; i<=10; i++) {

    double ti = 0.1 * i;
    
    double term1 = std::exp(-x1*ti);
    double term2 = std::exp(-x2*ti);
    double term3 = std::exp(ti) - std::exp(-10*ti);
    
    sum += (term1 - term2) - term3;
  }
  return sum;
}

// Chemical Equilibrium Problem 
double chemical_equilibrium(std::vector<double>& x) {

  double x1 = x[0];
  double x2 = x[1];
  double x3 = x[2];

  double term1 = pow((1 - x1 - x2)*(1 - x3 - x1) - 4*pow(x1,2)/549000, 2); 
  double term2 = pow((1 - x1 - x2)*(1 - x2 - x3) - 4*pow(x2,2)/362, 2);
  double term3 = pow((1 - x2 - x3)*(1 - x3 - x1) - 4*pow(x3,2)/3.28, 2);

  return term1 + term2 + term3;
}

// Heat Conduction Problem 
double heat_conduction(std::vector<double>& x ) {

  double x1 = x[0];
  double x2 = x[1]; 
  double x3 = x[2];
  double x4 = x[3];

  double term1 = pow(2*(x2 + x3 - 4*x1) + 20 - 1.5*x1 + pow(x1,2)/20, 2);
  double term2 = pow(2*(x1 - 3*x3 + x4) + 20 - 1.5*x3 + pow(x3,2)/20, 2);
  double term3 = pow(2*(x2 - x4) + 20 - 1.5*x2 + pow(x2,2)/20, 2);
  double term4 = pow(2*(x1 + x3 - 2*x4) + 20 - 1.5*x4 + pow(x4,2)/20, 2);

  return term1 + term2 + term3 + term4;

}
