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

/**
 * @brief writeResults Write the results to an outputfile where each line has the form "lambda_i ||T*xi - lambda_i*xi||
 * @param filename Name of file
 * @param OD Array with diagonal elements of original matrix T
 * @param OE Array with off-diagonal elements of original matrix T (size n-1)
 * @param D Diagonal elements
 * @param z Vector z
 * @param L Eigenvalues lambda_i
 * @param N Normalization factors
 * @param Q Square matrix of order n with eigenvectors as columns
 * @param n Size of D,z,L,N, OD
 * @return 0, if write process was successful
 *
 * D,z,L,N are the vectors that results from the rank-1 update in the highest stage, if Cuppen's algorithm (thus splitting of T)
 * was applied. In this case Q == NULL. If no splitting was performed (because T is too small to divide the problem),
 * then Q will contain the eigenvectors, where Q is the results of MKL's QR algorithm.
 */
int writeResults(const char* filename, double* OD, double* OE, double* D, double* z, double* L, double* N, double* Q, int n);

#endif // FILEHANDLING_H
