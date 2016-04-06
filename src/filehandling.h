#ifndef FILEHANDLING_H
#define FILEHANDLING_H

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

#endif // FILEHANDLING_H
