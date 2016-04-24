#ifndef FILEHANDLING_H
#define FILEHANDLING_H

#include "helper.h"
#include "backtransformation.h"

/**
 * @brief The EVToComputeStruct struct The user may define in a file, which eigenvectors he wants to compute. We extract this information for the backtransformation in such a struct
 */
struct EVToComputeStruct {
    /**
     * @brief all Set to true, if all eigenvectors, should be computed.
     */
    int all;
    /**
     * @brief n Number of eigenvectors to compute
     */
    int n;
    /**
     * @brief indices Indices Indices of eigenvectors, we want to compute
     */
    int* indices;
};
typedef struct EVToComputeStruct EVToCompute;

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
 * @brief determineEigenvectorsToCompute Determine, which eigenvectors we have to compute at the end from the file given by the user
 * @param compEV Flag, if we should compute any eigenvectors
 * @param filename Name of filename which contains the indices of the eigenvectors we should compute (NULL otherwise)
 * @param n Size of matrix T
 * @param ret A sorted array with the eigenvectors to compute
 * @return -1, if an IO error occured, else 0
 */
int determineEigenvectorsToCompute(int compEV, char* filename, int n, EVToCompute *ret);

/**
 * @brief writeResults Write the results to an outputfile where each line has the form "lambda_i ||T*xi - lambda_i*xi||, if xi should be computed, otherwise only the eigenvalue
 * @param filename Name of file
 * @param OD Array with diagonal elements of original matrix T
 * @param OE Array with off-diagonal elements of original matrix T (size n-1)
 * @param t The tree, that stores all the eigenvector matrix represenatitions for the curren task
 * @param comm The MPI handle with the information neccessary to allow inter-task communication
 * @param compEV Flag, if we should compute any eigenvectors
 * @param evFile Name of filename which contains the indices of the eigenvectors we should compute (NULL otherwise)
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
int writeResults(const char* filename, double* OD, double* OE, EVRepTree *t, MPIHandle comm, int computeEV, char* evFile);

#endif // FILEHANDLING_H
