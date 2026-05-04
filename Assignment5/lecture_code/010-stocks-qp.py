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
