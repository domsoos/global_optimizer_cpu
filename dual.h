#pragma once
#include <vector>
#include <functional>

namespace dual {

class DualNumber {
public:
    double real;
    double dual;

    DualNumber(double real = 0.0, double dual = 0.0) : real(real), dual(dual) {}

    DualNumber& operator+=(const DualNumber& rhs) {
        real += rhs.real;
        dual += rhs.dual;
        return *this;
    }

    DualNumber operator+(const DualNumber& rhs) const {
        return DualNumber(real + rhs.real, dual + rhs.dual);
    }

    DualNumber operator-(const DualNumber& rhs) const {
        return DualNumber(real - rhs.real, dual - rhs.dual);
    }

    DualNumber operator*(const DualNumber& rhs) const {
        return DualNumber(real * rhs.real, dual * rhs.real + real * rhs.dual);
    }

    DualNumber operator/(const DualNumber& rhs) const {
        double denom = rhs.real * rhs.real;
        return DualNumber(real / rhs.real, (dual * rhs.real - real * rhs.dual) / denom);
    }
    // operator for double - DualNumber
    friend DualNumber operator-(double lhs, const DualNumber& rhs) {
        return DualNumber(lhs - rhs.real, -rhs.dual);
    }

    // operator for double * DualNumber
    friend DualNumber operator*(double lhs, const DualNumber& rhs) {
        return DualNumber(lhs * rhs.real, lhs * rhs.dual);
    }
};

inline dual::DualNumber dual_abs(const dual::DualNumber &a) {
    return (a.real < 0.0) ? dual::DualNumber(-a.real, -a.dual) : a;
}

static __inline__ DualNumber sin(const DualNumber& x) {
    return DualNumber(sinf(x.real), x.dual * cosf(x.real));
}

static __inline__ DualNumber cos(const DualNumber& x) {
    return DualNumber(cosf(x.real), -x.dual * sinf(x.real));
}

static __inline__ DualNumber exp(const DualNumber& x) {
    double ex = expf(x.real);
    return DualNumber(ex, x.dual * ex);
}

static __inline__ DualNumber sqrt(const DualNumber& x) {
    double sr = sqrtf(x.real);
    return DualNumber(sr, x.dual / (2.0 * sr));
}

static __inline__ DualNumber atan2(const DualNumber& y, const DualNumber& x) {
    double denom = x.real * x.real + y.real * y.real;
    return DualNumber(atan2f(y.real, x.real), (x.real * y.dual - y.real * x.dual) / denom);
}

template<typename T>
static __inline__ T pow(const T& base, double exponent) {
    return T(powf(base.real, exponent), exponent * powf(base.real, exponent - 1) * base.dual);
}

} // end of dual name space


/// A “pure AD” function: takes an n-vector of DualNumbers (where
/// the .dual fields seed partials) and returns f(x) in .real, df in .dual.
using ADFunc = std::function<dual::DualNumber(const std::vector<dual::DualNumber>&)>;

// serial gradient using automatic differentiation
inline std::vector<double> gradientAD(
    const ADFunc &f_ad,
    const std::vector<double> &x
) {
    size_t n = x.size();
    std::vector<dual::DualNumber> xDual(n);
    std::vector<double>           grad(n);

    // initialize real parts, zero dual parts
    for (size_t i = 0; i < n; ++i)
        xDual[i] = dual::DualNumber(x[i], 0.0);

    // seed one coordinate at a time
    for (size_t i = 0; i < n; ++i) {
        xDual[i].dual = 1.0;                   // dx_i/dx_i = 1
        dual::DualNumber res = f_ad(xDual);    // res.dual = df/dx_i
        grad[i]           = res.dual;
        xDual[i].dual     = 0.0;               // reset for next pass
    }

    return grad;
}


