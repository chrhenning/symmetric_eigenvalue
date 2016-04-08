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

/**
 * @brief computeNormalizationFactors Compute the normalization factors for the eigenvector construction
 * @param D Diagonal elements
 * @param z Vector z
 * @param L Eigenvalues lambda_i
 * @param n Size of D,z,L
 * @return Vector of size n with normalization factors
 *
 * Each normalization factor is computed as: sqrt(sum_i(z_i^2/(d_i - lambda_i)^2))
 */
double* computeNormalizationFactors(double* D, double* z, double* L, int n);

/**
 * @brief getEVElement Get an entry j from eigenvector i
 * @param D Diagonal elements
 * @param z Vector z
 * @param L Eigenvalues lambda_i
 * @param N Normalization factors
 * @param n Size of D,z,L,N
 * @param i Considered eigenvector
 * @param j Considered entry in i-th eigenvector
 * @return  Entry j of eigenvector i
 *
 * The element is computed as follows: z_j / ((d_j-lambda_i) * N[i])
 */
inline double getEVElement(double* D, double* z, double* L, double* N, int n, int i, int j) { return z[j] / ((D[j]-L[i]) * N[i]); }


#endif // EIGENVALUES_H
