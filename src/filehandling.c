#include "filehandling.h"

#include <stdlib.h>
#include <stdio.h>
#include "../lib/mmio.h"

int readTriadiagonalMatrixFromSparseMTX(const char* filename, double* T, int* n) {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int R; // rows
    int C; // columns
    int NNZ; // number of non-zero elements

    if ((f = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Could not open file\n");
        return -1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -1;
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (!(mm_is_coordinate(matcode) && mm_is_real(matcode) &&
            mm_is_general(matcode)))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &R, &C, &NNZ)) !=0)
        return -1;

    if (R != C) {
        printf("Matrix is not square\n");
        return -1;
    }

    *n = R;

    /*
     * Read elements to matrix
     */
    T = malloc(3*R * sizeof(double));

    int r,c;
    double v;
    int i;
    for (i = 0; i<NNZ; ++i) {
        fscanf(f, "%d %d %lg\n", &r, &c, &v);
        if (r-c > 1 || c-r > 1) {
            printf("Matrix is not tridiagonal\n");
            return -1;
        }
        T[(r-1)*3 + (c-1)] = v;
    }

    if (f !=stdin) fclose(f);

    return 0;
}
