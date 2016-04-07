#include <omp.h>

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
