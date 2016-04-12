#ifndef HELPER_H
#define HELPER_H

#include <stdlib.h>

/*
 * This file contains a bunch of helper functions to keep the main file clearly arranged
 */

/**
 * @brief myfree Free the pointer were the given pointer points to and set it to NULL
 * @param ptr Pointer to pointer
 */
inline void myfree(double** ptr) { free(*ptr); *ptr = NULL; }

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
 * lambda_i = 2 + 2 * cos((PI*i)/(n+1))
 */
void createMatrixScheme2(double** D, double** E, int n);

/**
 * @brief computeZ See description below
 * @param Q1l Last row of Q1
 * @param Q2f First row of Q2
 * @param nq1 Dimension of Q1
 * @param nq2 Dimension of Q2
 * @param theta
 * @return Pointer to vector z of size q1+q2
 *
 * Compute z, where z is
 *
 * z = | Q1^T   0 | | e_k            |
 *     | 0   Q2^T | | theta^-1 * e_1 |
 *
 * Note, I only need the last row of Q1 and the first row of Q2 in order to compute z
 */
double* computeZ(double* Q1l, double* Q2f, int nq1, int nq2, double theta);

/**
 * @brief computeEigenvaluesOfScheme2 Compute the eigenvalues: lambda_i = 2 + 2 * cos((PI*i)/(n+1))
 * @param n Size of matrix
 * @return
 */
double* computeEigenvaluesOfScheme2(int n);

/**
 * @brief printVector Print vector on console
 * @param vec Vector
 * @param n Size
 */
void printVector(double* vec, int n);

/**
 * @brief printTridiagonalMatrix Print tridiagonal matrix on console
 * @param D Diagonal elements
 * @param E Off-diagonal elements
 * @param n Size
 */
void printTridiagonalMatrix(double* D, double* E, int n);

/**
 * @brief printMatrix Print general matrix
 * @param M Matrix
 * @param r Number of rows
 * @param c Number of columns
 */
void printMatrix(double* M, int r, int c);

#endif // HELPER_H
