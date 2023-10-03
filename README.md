# Global Optimizer CPU
HPC Research Project for Global Optimization

The full report can be found [here](https://rpubs.com/domsoos/1091304)

To compile the program, run the following command:

```
make && ./main
```
The program will ask how many dimensions the user would like to use the Rosenbrock function. Then the user have to choose the type of algorithm to run.      

To generate a resource usage report, install R and Quarto then use:
```
quarto render memory.qmd
```

# Unconstrained Optimization Algorithm
-   **Require:**
    -   Objective function $f: \mathbb{R}^n \rightarrow \mathbb{R}$
    -   Gradient of the function $\nabla f: \mathbb{R}^n \rightarrow \mathbb{R}^n$
    -   Initial guess $x_0$
-   **Ensure:** Local minimum $x^*$ of $f$

1.  Set $k = 0$
2.  Choose an initial approximation $H_0$ to the inverse Hessian - the identity matrix
3.  While not converged:
    1.  Compute the search direction: $p_k = -H_k \nabla f(x_k)$
    2.  Calculate step size $\alpha_k$ using Line Search
    3.  Update the new point: $x_{k+1} = x_k + \alpha_k p_k$
    4.  Compute $\delta x = x_{k+1} - x_k$
    5.  Compute $\delta g = \nabla f(x_{k+1}) - \nabla f(x_k)$
    6.  Update formula for $H_{k+1}$
    7.  $k = k + 1$
4.  Return $x_k$ as the local minimum

# Davidon-Fletcher-Powell update
The primary goal of DFP is to update the approximation of the inverse of the Hessian matrix.
The update formula is: 
$$H_{k+1} = H_k + \frac{\delta x \delta x^T}{\delta x^T \delta g} - \frac{H_k \delta g \delta g^T H_k}{\delta g^T H_k \delta g}$$

positive rank-1 update: $$\frac{\delta x \delta x^T}{\delta x^T \delta g} $$

negative rank-2 update: $$\frac{H_k \delta g \delta g^T H_k}{\delta g^T H_k \delta g} $$

# Broyden-Fletcher-Goldfarb-Shanno update
The BFGS method is another Quasi-Newton method that approximates the inverse of the Hessian using a combination of rank-1 updates.
The formula is: $$H_{k+1} = \left(I - \frac{\delta x \delta g^T}{\delta x^T \delta g} \right) H_k \left(I - \frac{\delta g \delta x^T}{\delta x^T \delta g} \right) + \frac{\delta x \delta x^T}{\delta x^T \delta g}$$

which can be expanded: $$H_{k+1} = H_k - H_k \frac{\delta x \delta g^T}{\delta x^T \delta g} H_k - \frac{\delta g \delta x^T}{\delta x^T \delta g} H_k + \frac{\delta x \delta x^T}{\delta x^T \delta g}$$

# Limited-memory BFGS or L-BFGS
L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is an optimization algorithm for unconstrained optimization problems.
It is a member of the quasi-Newton family and is particularly well-suited for high-dimensional machine learning tasks aimed at minimizing a differentiable scalar function $f(\mathbf{x})$

Unlike the full BFGS algorithm, which requires storing a dense $n \times n$ inverse Hessian matrix, L-BFGS is memory-efficient.
It only stores a limited history of $m$ updates of the gradient $\nabla f(x)$ and position $x$, generally with $m < 10$.

The algorithm starts with an initial estimate $\mathbf{x}_0$ and iteratively refines this using estimates $\mathbf{x}_1, \mathbf{x}_2, \ldots$ The gradient $g_k = \nabla f(\mathbf{x}_k)$ 

is used to approximate the inverse Hessian and identify the direction of steepest descent.
The L-BFGS update for the inverse Hessian $H_{k+1}$ is given by:

$$H_{k+1} = (I - \rho_k s_k y_k^\top) H_k (I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top$$

where $\rho_k = \frac{1}{y_k^\top s_k}$, $s_k = x_{k+1} - x_k$, $y_k = g_{k+1} - g_k$

## L-BFGS-B (L-BFGS with Box Constraints)

L-BFGS-B is an extension of L-BFGS that can handle bound constraints, i.e., constraints of the form 
$$l \leq x \leq u$$
The algorithm uses projected gradients to ensure that the bounds are obeyed at every iteration.


## MultiDimensional Rosenbrock Function
The Rosenbrock function is a commonly used test problem for optimization algorithms due to its non-convex nature and the difficulty in converging to the global minimum. 

10 Dimensional Rosenbrock Summation

$$f(\mathbf{x}) = \sum_{i=0}^{9} \left[ (1 - x_i)^2 + 100 \cdot (x_{i+1} - x_i^2)^2 \right]$$
where $\mathbf{x} = (x_1, x_2, \ldots, x_{10})$ and $\mathbf{x} \in \mathbb{R}^{10}$

We expect to see a difference in the amount of resources used by each algorithm as we increase the number of dimensions. The limited memory variant of BFGS, L-BFGS stores only a few vectors that implicitly represent the approximation of the inverse Hessian matrix. It should be highly suitable for high-dimensional problems.
