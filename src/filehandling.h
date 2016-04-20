#ifndef FILEHANDLING_H
#define FILEHANDLING_H

#include "helper.h"
#include "backtransformation.h"

/**
 * @brief readBandedMatrixFromSparseMTX Read a tridiagonal matrix from a file
 * @param filename Name of a file, which contains a sparse matrix in mtx format
 * @param T Tridiagonal matrix in Intel banded row-major matrix format
 * @param n number of rows resp. columns
 * @return Was reading process successfull
 * @see https://software.intel.com/en-us/node/520871
 *
 * Note, the matrix has to be a square matrix
 */
int readTriadiagonalMatrixFromSparseMTX(const char* filename, double **T, int *n);

/**
 * @brief readBandedMatrixFromSparseMTX Read a symmetric tridiagonal matrix from a file
 * @param filename Name of a file, which contains a sparse matrix in mtx format
 * @param D Array with diagonal elements of matrix (size n)
 * @param E Array with off-diagonal elements of matrix (size n-1)
 * @param n number of rows resp. columns
 * @return Was reading process successfull
 *
 * Note, the matrix has to be a square matrix
 */
int readSymmTriadiagonalMatrixFromSparseMTX(const char* filename, double **D, double **E, int *n);

/**
 * @brief writeResults Write the results to an outputfile where each line has the form "lambda_i ||T*xi - lambda_i*xi||, if xi should be computed, otherwise only the eigenvalue
 * @param filename Name of file
 * @param OD Array with diagonal elements of original matrix T
 * @param OE Array with off-diagonal elements of original matrix T (size n-1)
 * @param t The tree, that stores all the eigenvector matrix represenatitions for the curren task
 * @param comm The MPI handle with the information neccessary to allow inter-task communication
 *
 * To access eigenvectors, we need to compute columns of the matrix
 * W = Q * U_(d-2) * ... * U_1 * U_0, where Q is the block diagonal matrix
 * with the dense eigenvector matrices from the leaf nodes. The U_i's are the block diagonal
 * matrices, which are the eigenvector matrices of the rank-one perturbation int the intermediate
 * stage i. (Stage 0 is root, stage (d-1) is leaf).
 *
 * Let's define U = Q * U_(d-2) * ... * U_1
 * To compute the j-th column in W, we have to multiply the i-th row from U (1<=i<=n)
 * with the j-th column from U_0.
 * This algorithm will compute the rows of U in parallel.
 */
int writeResults(const char* filename, double* OD, double* OE, EVRepTree *t, MPIHandle comm);

#endif // FILEHANDLING_H
