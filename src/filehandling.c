#include "filehandling.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <omp.h>
//#include "mkl.h"

#include "../lib/mmio.h"

int readTriadiagonalMatrixFromSparseMTX(const char* filename, double** T, int* n) {
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
    *T = malloc(3*R * sizeof(double));

    int r,c;
    double v;
    int i;
    for (i = 0; i<NNZ; ++i) {
        fscanf(f, "%d %d %lg\n", &r, &c, &v);
        if (r-c > 1 || c-r > 1) {
            printf("Matrix is not tridiagonal\n");
            return -1;
        }
        (*T)[(r-1)*3 + (c-1)] = v;
    }

    if (f !=stdin) fclose(f);

    return 0;
}

int readSymmTriadiagonalMatrixFromSparseMTX(const char* filename, double **D, double **E, int *n) {
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
    *D = malloc(R * sizeof(double));
    *E = malloc((R-1) * sizeof(double));

    int r,c;
    double v;
    int i;
    for (i = 0; i<NNZ; ++i) {
        fscanf(f, "%d %d %lg\n", &r, &c, &v);
        if (r-c > 1 || c-r > 1) {
            printf("Matrix is not tridiagonal\n");
            return -1;
        }

        if (r == c)
            (*D)[(r-1)] = v;
        else {
            // in the mtx format, I will read sub diagonal (r == c+1) elements always before super diagonal (c == r+1) elements
            if (c == r + 1) { // if super diagonal
                // check that matrix is symmetric
                if ((*E)[r-1] != v) {
                    printf("Matrix is not symmetric\n");
                    return -1;
                }
            } else { // store sub diagonal element
                (*E)[c-1] = v;
            }
        }
    }

    if (f !=stdin) fclose(f);

    return 0;
}

int writeResults(const char* filename, double* OD, double* OE, double* D, double* z, double* L, double* N, double* Q, int n) {

    assert(n > 0);
    FILE *f;

    if ((f = fopen(filename, "w")) == NULL) {
        fprintf(stderr, "Could not open file\n");
        return -1;
    }

    double norm, lambda;
    int i,j;
    double *x = malloc(n * sizeof(double));

    // current eigenvector
    double* xi = malloc(n * sizeof(double));

    // for each eigenvalue
    for (i = 0; i < n; ++i) {
        // extract current eigenvector
        if (Q != NULL) { // if we haven't applied cuppens algorithm (no splits)
            #pragma omp parallel for default(shared) private(j) schedule(static)
            for (j = 0; j < n; ++j) {
                xi[j] = Q[n*j + i];
            }
            lambda = D[i];
        } else {
            #pragma omp parallel for default(shared) private(j) schedule(static)
            for (j = 0; j < n; ++j) {
                xi[j] = getEVElement(D,z,L,N,n,i,j);
            }
            lambda = L[i];
        }

        // compute x = T*x_i, where x_i is the current eigenvector
        if (n == 1) {
            x[0] = OD[0] * xi[0];
        } else {
            x[0] = OD[0] * xi[0] + OE[0] * xi[1];
            #pragma omp parallel for default(shared) private(j) schedule(static)
            for (j = 1; j < n-1; ++j) {
                x[j] = OE[j-1] * xi[j-1] + OD[j] * xi[j] + OE[j] * xi[j+1];
            }
            x[n-1] = OE[n-2] * xi[n-2] + OD[n-1] * xi[n-1];
        }

        norm = 0;
        // compute ||x - lambda_i*x_i||
        #pragma omp parallel for default(shared) private(j) schedule(static) reduction(+:norm)
        for (j = 0; j < n; ++j) {
            x[j] -= lambda * xi[j];
            norm = norm + x[j]*x[j];
        }
        norm = sqrt(norm);

        // compute norm of x
        //norm = cblas_dnrm2(n, x, 1);

        // write results to file
        fprintf(f, "%20.19g %20.19g\n", lambda, norm);
    }

    free(xi);
    free(x);

    if (f !=stdout) fclose(f);

    return 0;
}
