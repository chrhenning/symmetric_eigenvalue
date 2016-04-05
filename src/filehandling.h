#ifndef FILEHANDLING_H
#define FILEHANDLING_H

/**
 * @brief readBandedMatrixFromSparseMTX Read a tridiagonal matrix from a file
 * @param filename Name of a file, which contains a sparse matrix in mtx format
 * @param T Tridiagonal matrix in Intel banded row-major matrix format
 * @param n number of rows resp. columns
 * @return Tridiagonal matrix in Intel banded row-major matrix format
 *
 * Note, the matrix has to be a square matrix
 */
void readTriadiagonalMatrixFromSparseMTX(const char* filename, double *T, int *n);

#endif // FILEHANDLING_H
