#ifndef HELPER_H
#define HELPER_H


inline int min ( int a, int b ) { return a < b ? a : b; }
inline int max ( int a, int b ) { return a > b ? a : b; }

/**
 * @brief createMatrixScheme1 Create a symm. tridiagonal matrix with rows [-1,d_i,-1] where the d_i are evenly spaced in the interval [1,100]
 * @param n Dimension of matrix
 * @param D Diagonal elements of matrix.
 * @param E Off-diagonal elements of matrix.
 */
void createMatrixScheme1(double** D, double** E, int n);

/**
 * @brief createMatrixScheme2 Create a symm. tridiagonal matrix with rows [-1,2,-1]
 * @param n Dimension of matrix
 * @param D Diagonal elements of matrix.
 * @param E Off-diagonal elements of matrix.
 *
 * The eigenvalues of this matrix should have the form:
 * lambda_i = 4 - 2 * cos((PI*i)/(n+1))
 */
void createMatrixScheme2(double** D, double** E, int n);

#endif // HELPER_H
