#include "helper.h"

#include <stdlib.h>
#include <omp.h>

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
