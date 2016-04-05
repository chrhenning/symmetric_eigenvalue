#include "helper.h"

#include <stdlib.h>

double* createMatrixScheme1(int n) {
    double* T = malloc(3*n * sizeof(double));

    double diagSpacing = (100.0-1.0) / (n-1);

    int i;
    for (i = 0; i < n; ++i) {
        T[i*3 + 0] = -1; // sub diagonal
        T[i*3 + 1] = 1.0 + i * diagSpacing;
        T[i*3 + 2] = -1; // super diagonal
    }

    // special cases
    T[0] = 0; // sub diagonal of row 0
    T[(n-1)*3 + 2] = 0; // super diagonal of row n-1

    return T;
}

double* createMatrixScheme2(int n) {
    // TODO
    return NULL;
}
