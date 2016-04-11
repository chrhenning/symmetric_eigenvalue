#include "helper.h"

#include <omp.h>
#include <math.h>
#include <assert.h>

void createMatrixScheme1(double** D, double** E, int n) {
    *D = malloc(n * sizeof(double));
    *E = malloc((n-1) * sizeof(double));

    double diagSpacing = (100.0-1.0) / (n-1);

    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n-1; ++i) {
        (*E)[i] = -1; // off diagonal
        (*D)[i] = 1.0 + i * diagSpacing;
    }
    (*D)[n-1] = 1.0 + i * diagSpacing; // one more diagonal element than off diagonal elements
}

void createMatrixScheme2(double **D, double **E, int n) {
    *D = malloc(n * sizeof(double));
    *E = malloc((n-1) * sizeof(double));

    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n-1; ++i) {
        (*E)[i] = -1; // off diagonal
        (*D)[i] = 2;
    }
    (*D)[n-1] = 2.0; // one more diagonal element than off diagonal elements
}


double* computeZ(double* Q1l, double* Q2f, int nq1, int nq2, double theta) {
    double* z = malloc((nq1+nq2) * sizeof(double));

    // copy last row of Q1 into z
    memcpy(z, Q1l, nq1*sizeof(double));

    // multiply first row of Q2 by theta^-1
    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for(i = 0; i < nq2; ++i)
        z[nq1+i] = Q2f[i] * theta;

    return z;
}

double* computeEigenvaluesOfScheme2(int n) {

    double* L = malloc(n * sizeof(double));

    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i)
        L[i] = 2 + 2 * cos((M_PI*(i+1))/(n+1));

    return L;
}

void printVector(double* vec, int n) {
    int i = 0;
    for (i = 0; i < n-1; ++i)
        printf("%g, ", vec[i]);
    printf("%g\n", vec[n-1]);
}

void printTridiagonalMatrix(double* D, double* E, int n) {
    assert(n>0);
    if (n == 1)
        printf("%g\n", D[0]);
    else if (n == 2) {
        printf("%g\t%g\n", D[0], E[0]);
        printf("%g\t%g\n", E[0], D[1]);
    }
    else {
        int i = 0;
        printf("%g\t%g\t0\n", D[0], E[0]);
        for (i = 1; i < n-2; ++i)
            printf("%g\t%g\t%g\n", E[i-1], D[i], E[i]);
        printf("0\t%g\t%g\n", E[n-2], D[n-1]);
    }
}
