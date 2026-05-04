## Diversion: Quadratic Programming and an Example with `cvxopt`

> This examples is here solely to show the use of cvxopt for quadratic programming, the library which we then use for SVM.

When solving the dual form of the SVM, we face a specific type of optimization problem known as a quadratic program. Quadratic programming involves minimizing (or maximizing) a quadratic objective function subject to linear constraints. To understand how this works in practice, we first introduce a simple real-world example that shows how quadratic programming problems arise and how they can be solved using Python.

Imagine you are investing money between two assets: stocks and bonds. You want to allocate your investment in a way that minimizes the overall risk. In finance, the risk of an investment is often quantified using the variance or covariance of returns. The idea is that if two assets tend to move together, the overall portfolio is riskier. If they move independently or in opposite directions, the risk is reduced. Mathematically, the total risk of the investment can be expressed as a quadratic function of the investment proportions.

Let \( x = (x_1, x_2) \) represent the fractions of your money invested in stocks and bonds, respectively. The risk is given by the expression:

\[
\text{Risk} = \frac{1}{2} x^\top P x
\]

where \( P \) is the covariance matrix of returns. Each entry of \( P \) has a specific meaning: \( P_{11} \) measures how much stock returns fluctuate on their own (the variance of stocks), \( P_{22} \) measures the variance of bond returns, and \( P_{12} = P_{21} \) measures the covariance between stock and bond returns. A positive covariance means that stocks and bonds tend to move together, while a negative covariance would indicate that they move in opposite directions.

For simplicity, we assume that the covariance matrix is:

\[
P = \begin{bmatrix} 0.1 & 0.05 \\ 0.05 & 0.2 \end{bmatrix}
\]

In this setup, stocks have a variance of 0.1, bonds have a higher variance of 0.2, and their returns are positively correlated with a covariance of 0.05. Your goal is to minimize the risk while investing all of your available money and ensuring that no negative investments are made (no short selling).

The optimization problem can be formally stated as:

\[
\min_{x} \quad \frac{1}{2} x^\top P x
\]
subject to:

\[
x_1 + x_2 = 1, \quad x_1 \geq 0, \quad x_2 \geq 0
\]

This says that the sum of investments must be exactly one (you invest all your money), and each investment must be non-negative.

We can solve this quadratic program using Python and the `cvxopt` library. The `cvxopt` package is designed for convex optimization and can efficiently solve quadratic problems. The following code demonstrates how to set up and solve the problem:

```python
import numpy as np
from cvxopt import matrix, solvers

# Covariance matrix
P = matrix([[0.1, 0.05],
            [0.05, 0.2]])

# No linear term in the objective
q = matrix([0.0, 0.0])

# Constraints: Gx <= h for x1 >= 0, x2 >= 0
G = matrix([[-1.0, 0.0],
             [0.0, -1.0]])
h = matrix([0.0, 0.0])

# Constraint: Ax = b for x1 + x2 = 1
A = matrix([[1.0], [1.0]])   # Note: A must have size (2, 1)
b = matrix([1.0])

# Solve the quadratic program
solution = solvers.qp(P, q, G, h, A, b)

# Extract and display solution
x = np.array(solution['x']).flatten()
print("Optimal portfolio allocation:", x)
```

In the quadratic program we want to solve, each matrix and vector has a specific role that corresponds to part of the mathematical formulation.

The matrix \( P \) represents the quadratic part of the objective function. The objective we are minimizing is

\[
\frac{1}{2} x^\top P x + q^\top x,
\]

where in our case \( q = 0 \), so the objective reduces to minimizing only the quadratic term. In our example, \( P \) is the covariance matrix of asset returns, encoding how much each asset fluctuates on its own (the variances) and how much the two assets fluctuate together (the covariance). The diagonal elements \( P_{11} \) and \( P_{22} \) represent the variance of stocks and bonds, respectively, while the off-diagonal elements \( P_{12} = P_{21} \) represent the covariance between stocks and bonds.

The vector \( q \) represents the linear part of the objective function. Since there is no linear component in the risk function, \( q \) is simply the zero vector.

The matrix \( G \) and the vector \( h \) encode the inequality constraints. Inequality constraints are written as

\[
Gx \leq h,
\]

and in this case, they enforce that the investment fractions \( x_1 \) and \( x_2 \) must be non-negative:

\[
x_1 \geq 0, \quad x_2 \geq 0.
\]

These conditions ensure that no negative investments (no short selling) are allowed.

The matrix \( A \) and the vector \( b \) encode the equality constraints. Equality constraints are written as

\[
Ax = b,
\]

and here they enforce that the entire available amount is invested, meaning:

\[
x_1 + x_2 = 1.
\]

This guarantees that the full investment is allocated between stocks and bonds without any leftover.

When setting up a quadratic program, arranging the problem into the standard form with matrices \( P \), \( q \), \( G \), \( h \), \( A \), and \( b \) is crucial, as solvers like cvxopt expect the input in exactly this structure. This same decomposition will be used in the next section to solve the dual form of the support vector machine.

When we run this code, the solver finds the optimal fractions of money to invest in stocks and bonds to achieve the minimum risk according to the given covariance matrix. 

```python
     pcost       dcost       gap    pres   dres
 0:  4.8915e-02 -9.7278e-01  1e+00  6e-17  2e+00
 1:  4.8045e-02  1.9093e-02  3e-02  1e-16  6e-02
 2:  4.3933e-02  4.1156e-02  3e-03  2e-16  4e-18
 3:  4.3750e-02  4.3676e-02  7e-05  1e-16  1e-17
 4:  4.3750e-02  4.3749e-02  7e-07  1e-16  4e-18
 5:  4.3750e-02  4.3750e-02  7e-09  1e-16  5e-18
Optimal solution found.
Optimal portfolio allocation: [0.74999986 0.25000014]
```

When we run the solver, it outputs information about the optimization process, showing how the primal cost and dual cost converge toward each other, with the gap between them shrinking at each iteration. These diagnostics indicate that the optimization is proceeding correctly. At the end, the solver reports that the optimal solution has been found. The final result shows that approximately 75% of the money should be invested in stocks and 25% in bonds to minimize the overall portfolio risk according to the given covariance matrix.

In convex optimization, the primal cost refers to the value of the original optimization objective, while the dual cost refers to the value of the dual optimization problem, which is mathematically derived from the primal by introducing Lagrange multipliers. For convex problems, strong duality usually holds, meaning that at the optimal solution, the primal and dual costs should be equal. During the optimization process, the solver monitors both the primal and dual costs, and the difference between them, called the duality gap, measures how close the current solution is to optimality. When the primal and dual costs converge and the duality gap becomes very small, the solver concludes that it has found an optimal solution.

