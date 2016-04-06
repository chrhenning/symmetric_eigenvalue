#ifndef HELPER_H
#define HELPER_H


inline int min ( int a, int b ) { return a < b ? a : b; }
inline int max ( int a, int b ) { return a > b ? a : b; }

/**
 * @brief createMatrixScheme1 Create a tridiagonal matrix with rows [-1,d_i,-1] where the d_i are evenly spaced in the interval [1,100]
 * @param n Dimension of matrix
 * @return Returns the created matrix in banded row-major format.
 */
double* createMatrixScheme1(int n);

double* createMatrixScheme2(int n);

#endif // HELPER_H
