#ifndef EIGENVALUES_H
#define EIGENVALUES_H
/*
 * Handle eigenvalue and eigenvector computation of rank-one updated matrix D + roh * z*z^T in this file
 */

/**
 * @brief computeEigenvalues Compute eigenvalues of matrix D + beta*theta * z * z^T
 * @param D Diagonal matrix of dimension n (diagonal elements are stored in 1D array of size n)
 * @param z Vector z of size n
 * @param n Size of D resp. z
 * @param beta
 * @param theta
 * @return Array of size n where entry i is the eigenvalue correspondending to diagonal entry d_i in D.
 */
double* computeEigenvalues(double* D, double* z, int n, double beta, double theta);


#endif // EIGENVALUES_H
