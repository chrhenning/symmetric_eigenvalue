#include "eigenvalues.h"

#include <omp.h>
#include <math.h>

double* computeEigenvalues(double* D, double* z, int n, double beta, double theta) {
    /*
     * Store eigenvalues in new array (do not overwrite D), since the elements in D are needed later on to compute the eigenvectors)S
     */
    double* L = malloc(n * sizeof(double));

    // TODO: compute eigenvalues
    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i)
        L[i] = 1;

    return L;
}

double* computeNormalizationFactors(double* D, double* z, double* L, int n) {
    double *N = malloc(n * sizeof(double));

    int i, j;
    double tmp;
    #pragma omp parallel for default(shared) private(i,j,tmp) schedule(static)
    for (i = 0; i < n; ++i) {
        N[i] = 0;
        for (j = 0; j < n; ++j) {
            tmp = D[j]-L[i];
            N[i] += z[j]*z[j] / (tmp*tmp);
        }

        N[i] = sqrt(N[i]);
    }

    return N;
}
