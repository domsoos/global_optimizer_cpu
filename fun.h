#pragma once

#include <vector>
#include <functional>

#include "dual.h"

template<int DIM>
dual::DualNumber rosenbrock_ad(const dual::DualNumber* x);

namespace util {

template<int DIM>
dual::DualNumber rosenbrock(const dual::DualNumber* x) {
  dual::DualNumber sum(0.0, 0.0);
  for (int i = 0; i < DIM - 1; ++i) {
    dual::DualNumber t1 = dual::DualNumber(1.0, 0.0) - x[i];
    dual::DualNumber t2 = x[i+1] - x[i]*x[i];
    sum = sum + t1*t1 + dual::DualNumber(100.0, 0.0)*t2*t2;
  }
  return sum;
}

template<int DIM>
double rosenbrock(const double* x) {
  double sum = 0.0;
  for (int i = 0; i < DIM - 1; ++i) {
    double t1 = 1.0 - x[i];
    double t2 = x[i+1] - x[i]*x[i];
    sum += t1*t1 + 100.0*t2*t2;
  }
  return sum;
}

template<int DIM>
dual::DualNumber rastrigin(const dual::DualNumber* x) {
  dual::DualNumber sum(10.0*DIM, 0.0);
  for (int i = 0; i < DIM; ++i) {
    sum = sum + ( x[i]*x[i]
                - dual::DualNumber(10.0,0.0)*dual::cos(x[i]*dual::DualNumber(2.0*M_PI,0.0)) );
  }
  return sum;
}

template<int DIM>
double rastrigin(const double* x) {
  double sum = 10.0*DIM;
  for (int i = 0; i < DIM; ++i)
    sum += x[i]*x[i] - 10.0*std::cos(2.0*M_PI*x[i]);
  return sum;
}

// Ackley Function (general d-dimensions)
//   f(x) = -20 exp\Bigl(-0.2\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}\Bigr)
//          - exp\Bigl(\frac{1}{d}\sum_{i=1}^{d}\cos(2\pi x_i)\Bigr)
//          + 20 + e
template<int DIM>
dual::DualNumber ackley(const dual::DualNumber* x) {
    dual::DualNumber sum_sq = 0.0;
    dual::DualNumber sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum_sq += dual::pow(x[i], 2);
        sum_cos += dual::cos(2.0 * M_PI * x[i]);
    }
    dual::DualNumber term1 = dual::DualNumber(-20.0) * dual::exp(-0.2 * dual::sqrt(sum_sq / DIM));
    dual::DualNumber term2 = dual::DualNumber(0.0) - dual::exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + dual::exp(1.0);
}

template<int DIM>
double ackley(const double* x) {
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum_sq += x[i] * x[i];
        sum_cos += cos(2.0 * M_PI * x[i]);
    }
    double term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / DIM));
    double term2 = -exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + exp(1.0);
}

// Goldstein-Price Function
//   f(x,y) = [1+(x+y+1)^2 (19-14x+3x^2-14y+6xy+3y^2)]
//            [30+(2x-3y)^2 (18-32x+12x^2+48y-36xy+27y^2)]
template<int DIM>
dual::DualNumber goldstein_price(const dual::DualNumber* x) {
    static_assert(DIM == 2, "Goldstein-Price is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0];
    dual::DualNumber x2 = x[1];
    dual::DualNumber term1 = dual::DualNumber(1.0) + dual::pow(x1 + x2 + 1.0, 2) *
        (19.0 - 14.0 * x1 + 3.0 * dual::pow(x1, 2) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * dual::pow(x2, 2));
    dual::DualNumber term2 = dual::DualNumber(30.0) + dual::pow(2.0 * x1 - 3.0 * x2, 2) *
        (18.0 - 32.0 * x1 + 12.0 * dual::pow(x1, 2) + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * dual::pow(x2, 2));
    return term1 * term2;
}

template<int DIM>
double goldstein_price(const double* x) {
    static_assert(DIM == 2, "Goldstein-Price is defined for 2 dimensions only.");
    double x1 = x[0];
    double x2 = x[1];
    double term1 = 1.0 + pow(x1 + x2 + 1.0, 2) *
        (19.0 - 14.0 * x1 + 3.0 * pow(x1, 2) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * pow(x2, 2));
    double term2 = 30.0 + pow(2.0 * x1 - 3.0 * x2, 2) *
        (18.0 - 32.0 * x1 + 12.0 * pow(x1, 2) + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * pow(x2, 2));
    return term1 * term2;
}

template<int DIM>
struct GoldsteinPrice {
    static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return goldstein_price<DIM>(x);
    }
    static double evaluate(const double* x) {
        return goldstein_price<DIM>(x);
    }
};


template<int DIM> struct Rosenbrock {
  static dual::DualNumber evaluate(const dual::DualNumber* x) {
    return rosenbrock<DIM>(x);
  }
  static double evaluate(const double* x) {
    return rosenbrock<DIM>(x);
  }
};
template<int DIM> struct Rastrigin {
  static dual::DualNumber evaluate(const dual::DualNumber* x) {
    return rastrigin<DIM>(x);
  }
  static double evaluate(const double* x) {
    return rastrigin<DIM>(x);
  }
};
template<int DIM> struct Ackley {
  static dual::DualNumber evaluate(const dual::DualNumber* x) {
    return ackley<DIM>(x);
  }
  static double evaluate(const double* x) {
    return ackley<DIM>(x);
  }
};


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


} // namespace util