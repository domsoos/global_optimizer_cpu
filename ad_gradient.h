// ad_gradient.h
#pragma once
#include <vector>
#include "dual.h"   // your dual::DualNumber

/// A “pure AD” function: takes an n-vector of DualNumbers (where
/// the .dual fields seed partials) and returns f(x) in .real, df in .dual.
using ADFunc = std::function<dual::DualNumber(const std::vector<dual::DualNumber>&)>;

/// Always use this gradient: no FD at all.
inline std::vector<double> gradientAD(
    const ADFunc &f_ad,
    const std::vector<double> &x
) {
    size_t n = x.size();
    std::vector<dual::DualNumber> xDual(n);
    std::vector<double>           grad(n);

    // 1) initialize real parts, zero dual parts
    for (size_t i = 0; i < n; ++i)
        xDual[i] = dual::DualNumber(x[i], 0.0);

    // 2) seed one coordinate at a time
    for (size_t i = 0; i < n; ++i) {
        xDual[i].dual = 1.0;                   // ∂x_i/∂x_i = 1
        dual::DualNumber res = f_ad(xDual);    // res.dual = ∂f/∂x_i
        grad[i]           = res.dual;
        xDual[i].dual     = 0.0;               // reset for next pass
    }

    return grad;
}
