import numpy as np


def f(X: np.ndarray):
    B = np.array([[0.5, 0.2],
                  [0.8, 0.5],
                  [0.1, 0.2]])

    # f(x): T s.t. R^3 → R^2
    # J(f) must be of shape (2 x 3) 
    return X @ B


def main():
    A = np.ones((1, 3))
    """
    A = (1, 1, 1)
    f(A) = (0.5 + 0.8 + 0.1, 0.2 + 0.5 + 0.2)

    more generally, f(A) = (0.5 * A_1 + 0.8 * A_2 + 0.1 * A_3, 0.2 * A_1 + 0.5 * A_2 + 0.2 * A_3)
                         = (f_1, f_2)
    f_1(x) = 0.5 * x_1 + 0.8 * x_2 + 0.1 * x_3
    f_2(x) = 0.2 * x_1 + 0.5 * x_2 + 0.2 * x_3

    J(f(x)) = (∂f_1/∂x_1, ∂f_1/∂x_2, ∂f_1/∂x_3
               ∂f_2/∂x_1, ∂f_2/∂x_2, ∂f_2/∂x_3)
            = (0.5, 0.8, 0.1
               0.2, 0.5, 0.2)
    """
    output = f(A)
    jacobian = np.array([[0.5, 0.8, 0.1],
                         [0.2, 0.5, 0.2]])

    """
    focusing on VJPs rather than JVPs, since we're gonna use reverse-mode auto-diff...
    to calculate vector-Jacobian product x^T @ J(f), need an m-dimensional vector (m = 2, in this case)
    why? since f: R^n → R^m, we need to multiply x^T (belonging to the output vector space) by J(f) to get something in the input vector space
    """

    u = np.array([[1],
                  [0]])  # results in a vector that shows how each element of f_1(x) affects f(A)
    v = np.array([[0],
                  [1]])  # similarly, results in a vector that shows how each element of f_2(x) affects f(A)
                         # i.e., the sensitivity of the second element of f(A) to inputs
    print(u.T @ jacobian)  # (0.5, 0.8, 0.1)
    print(v.T @ jacobian)  # (0.2, 0.5, 0.2)
    # intuitively makes sense, since first element of f(A) is (0.5 * A_1) + (0.8 * A_2) + (0.1 * A_3) 

    w = np.array([[1],
                  [1]])
    print(w.T @ jacobian)  # the sensitivity of all output elements to inputs

    # u, v, w don't have to only consist of 0 or 1
    # arbitrarily valued vector in VJP essentially means individually weighting the sensitivity of output elements to inputs (see below)

    x = np.array([[0.2],
                  [0.8]])
    print(x.T @ jacobian)

if __name__ == "__main__":
    main()
