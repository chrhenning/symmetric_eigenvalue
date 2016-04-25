#include "filehandling.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <omp.h>
//#include "mkl.h"
#include "eigenvalues.h"

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

/* qsort int comparison function */
int int_cmp(const void *a, const void *b)
{
    const int *ia = (const int *)a; // casting pointer types
    const int *ib = (const int *)b;
    return *ia  - *ib;
    /* integer comparison: returns negative if b > a
    and positive if a > b */
}

int determineEigenvectorsToCompute(int compEV, char* filename, int n, EVToCompute* ret) {
    ret->all = 0;
    ret->n = 0;
    ret->indices = NULL;
    if (!compEV)
        return 0;
    else if(filename == NULL) {
        // in this case, we have to compute all eigenvectors
        ret->all = 1;
        return 0;
    }

    // number of lines in file
    int numLines = 0;

    // read eigenvectors to read from file
    FILE *f;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    /*
     * Determine at first the number of lines in the file to allocate an appropriate vector
     */
    if ((f = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return -1;
    }
    // for each line
    while ((read = getline(&line, &len, f)) != -1) {
        int curr = atoi(line);
        if (curr == 0 || curr > n) {
            // replace last character (\n) in line by \0
            line[strlen(line)-1] = '\0';
            printf("WARNING: Line %d (\"%s\") in file %s will be ignored. No valid eigenvector index for given problem.\n", numLines, line, filename);
        } else {
            numLines++;
        }
    }
    if (f !=stdout) fclose(f);
    if (line) {
        free(line);
        line = NULL;
    }


    /*
     * Store eigenvectors to compute in struct
     */
    if ((f = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return -1;
    }
    ret->n = numLines;
    ret->indices = malloc(ret->n * sizeof(int));
    // for each line
    int j = 0;
    while ((read = getline(&line, &len, f)) != -1) {
        int curr = atoi(line);
        if (curr > 0 && curr <= n) {
            assert(j < numLines);
            ret->indices[j] = curr-1;
            j++;
        }
    }
    assert(j == numLines);
    if (f !=stdout) fclose(f);
    if (line)
        free(line);

    // sort array
    qsort(ret->indices, ret->n, sizeof(int), int_cmp); // there still might be values appearing more than once

    return 0;
}

int writeResults(const char* filename, double* OD, double* OE, EVRepTree* t, MPIHandle comm, int computeEV, char* evFile) {

    MPI_Status status;
    // abbreviations
    int taskid = comm.taskid;
    int numtasks = comm.numtasks;

    // size of original T (OE, OD)
    int n;

    // time measurement for backtransformation
    double esum = 0, etic, etoc;
    // time measurement for eigenvector calculation
    double evsum = 0, evtic, evtoc;

    // get root node
    EVRepNode* root = NULL;
    if (taskid == MASTER) {
        assert(t->d > 0 && t->t[0].n == 1);
        root = &(t->t[0].s[0]);
        n = root->n;
    }
    MPI_Bcast(&n,1,MPI_INT,MASTER,comm.comm);
    assert(n > 0);

    FILE *f;
    if (taskid == MASTER) {
        if ((f = fopen(filename, "w")) == NULL) {
            fprintf(stderr, "Could not open file\n");
            MPI_ABORT(comm.comm, 3);
        }
    }

    int i,j;

    double* Q = NULL;
    double* D = NULL;
    double* z = NULL;
    double* L = NULL;
    double* N = NULL;
    int* G = NULL;

    if (taskid == MASTER) {
        Q = root->Q;
        D = root->D;
        z = root->z;
        L = root->L;
        N = root->N;
        G = root->G;
    }


    double norm, lambda;
    // I need this vector to compute the relative residual T*xj -lambda*xj
    double *x = NULL;
    // The i-th eigenvector
    double *xi = NULL;

    if (taskid == MASTER) {
        x = malloc(n * sizeof(double));
        xi = malloc(n * sizeof(double));
    }

    // current line computed from the product of U = Q*U_(d-1)*...*U_1
    double* rj = malloc(n * sizeof(double));
    double* rjTemp = malloc(n * sizeof(double)); // buffer for vec-mat multiplication

    // decide which eigenvector we wanna compute and sort the eigenvalues, such that we can write them in the right order to a file
    EVToCompute evToCompute;
    DiagElem* SL = NULL; // eigenvalues sorted
    if (taskid == MASTER) {
        // sort eigenvalues (note, their order might not be determined by the permutation given in
        // P, because deflation might have changed the order, that's the reason why we have to
        // sort Lambda from scratch
        SL = malloc(n * sizeof(DiagElem));
        #pragma omp parallel for default(shared) private(i) schedule(static)
        for (i = 0; i < n; ++i) {
            SL[i].e = L[i];
            SL[i].i = i;
        }
        qsort(SL, n, sizeof(DiagElem), compareDiagElem);

        if(determineEigenvectorsToCompute(computeEV, evFile, n, &evToCompute) == -1) {
            fclose(f);
            MPI_ABORT(comm.comm, 3);
        }
    }

    // for each eigenvalue i (in sorted order
    int iter;
    int iterEV = 0;
    for (iter = 0; iter < n; ++iter) {
        // decide which is the index of the current eigenvalue to compute and if we should compute its eigenvector as well
        int computeCurrEV = 0; // flag, if we should compute the current eigenvector
        if (taskid == MASTER) { // decide, which is the next eigenvector, in the ascending order
            i = SL[iter].i; // Note, we consider eigenvalue i, but if we L would be stored sorted, then it would be eigenvallue iter

            // should we compute eigenvector iter?
            if (evToCompute.all) {
                computeCurrEV = 1;
            } else if (evToCompute.n > 0) {
                while (iterEV < evToCompute.n && evToCompute.indices[iterEV] < iter) iterEV++;
                if (iterEV < evToCompute.n && evToCompute.indices[iterEV] == iter)
                    computeCurrEV = 1;
            }
        }
        MPI_Bcast(&i,1,MPI_INT,MASTER,comm.comm);
        MPI_Bcast(&computeCurrEV,1,MPI_INT,MASTER,comm.comm);

        if (taskid == MASTER)
            lambda = L[i];

        // if we should extract the current eigenvector
        if (computeCurrEV) {
            etic = omp_get_wtime();

            // extract current eigenvector
            if (Q != NULL) { // if we haven't applied cuppens algorithm (no splits)
                assert(numtasks == 1);
                #pragma omp parallel for default(shared) private(j) schedule(static)
                for (j = 0; j < n; ++j) {
                    xi[j] = Q[n*j + i];
                }
            } else {
                /*
                 * Extract eigenvector through backtransformation
                 */
                // current leaf node
                EVRepNode* cn = &(t->t[t->d-1].s[taskid]);
                EVRepNode* cnc = NULL; // child
                // size of leaf node (determines how many lines this task computes)
                int nl = cn->n;
                // what is the line in the overall matrix U that we are currently calculating?
                int currJ = cn->o;
                // extract 'j'-th component of xi
                for (j = 0; j < nl; ++j) {
                    //printf("Task %d working on row %d\n", taskid, currJ);
                    cn = &(t->t[t->d-1].s[taskid]);
                    currJ = cn->o + j;
                    // compute line currJ
                    // set rj to zero
                    memset(rj, 0, n * sizeof(double));

                    // copy elements from leaf node in rj (j-th line in Q
                    assert(cn->Q != NULL);
                    memcpy(rj+cn->o, cn->Q+nl*j, nl*sizeof(double));

                    // helper variables
                    int pn = nl; // size of child problem
                    int po = cn->o; // offset of child

                    // climb up the tree
                    int s;
                    for (s = t->d-1; s >= 0; --s) {

                        int rowsToForward = 1; // how many rows we have to forward in current stage
                        if (s < t->d-1) { // we have to receive the rows from our right child
                            if (cn->right != cn->left)
                                rowsToForward = cn->right->numLeaves;
                            else
                                rowsToForward = 0;
                        }
                        // I have to forward a row for each of my leaves
                        int l;
                        for (l = 0; l < rowsToForward; ++l) {
                            // if we are not in the bottom stage, then we need to receive the rows from the right child
                            if (s < t->d-1) {
                                // leave l
                                EVRepNode* cnl = &(t->t[t->d-1].s[cn->right->taskid+l]);
                                if (j >= cnl->n)
                                    continue;
                                currJ = cnl->o + j;
                                memset(rj, 0, n * sizeof(double));
                                // receive row
                                MPI_Recv(&pn, 1, MPI_INT, cn->right->taskid, currJ, comm.comm, &status);
                                MPI_Recv(&po, 1, MPI_INT, cn->right->taskid, n+currJ, comm.comm, &status);
                                MPI_Recv(rj+po, pn, MPI_DOUBLE, cn->right->taskid, 2*n+currJ, comm.comm, &status);
                            }

                            // now I have to forward the row as high as possible
                            EVRepNode* tcn = cn;
                            // climb up the tree
                            int ts;
                            for (ts = s; ts >= 0; --ts) {                                
                                // if I climb up along a single path without splits
                                if (ts < t->d-1 && tcn->left == tcn->right) {
                                    //printf("task %d forward in stage %d\n", taskid, ts);
                                    tcn = tcn->parent;
                                    continue;
                                }

                                // send current row to parent                                
                                if (tcn->taskid != taskid) {
                                    //printf("Task %d forward row %d to %d\n", taskid, currJ, tcn->taskid);
                                    MPI_Send(&pn, 1, MPI_INT, tcn->taskid, currJ, comm.comm);
                                    MPI_Send(&po, 1, MPI_INT, tcn->taskid, n+currJ, comm.comm);
                                    MPI_Send(rj+po, pn, MPI_DOUBLE, tcn->taskid, 2*n+currJ, comm.comm);
                                    break;
                                }

                                // if U_s is stored in this task
                                // multiply row rj with U_ts
                                if (ts < t->d-1 && ts > 0) { // if ts == d-1: nothing to do, since we already copied row from Q

                                    assert(tcn->L != NULL);
                                    assert(tcn->o <= po);
                                    int ro = po - tcn->o;
                                    // compute rj*U_ts
                                    memcpy(rjTemp+po, rj+po, pn*sizeof(double));
                                    int r, c;

                                    #pragma omp parallel private(r,c) // parallel region to ensure, that each thread has another array allocated for the eigenvector
                                    {
                                        // store c-th eigenvector of U
                                        double* ev = malloc(tcn->n * sizeof(double));

                                        #pragma omp for private(evtic, evtoc) reduction(+:evsum)
                                        for (c = 0; c < tcn->n; ++c) {
                                            // get c-th eigenvector of U
                                            evtic = omp_get_wtime();
                                            getEigenVector(tcn, ev, c);
                                            evtoc = omp_get_wtime();
                                            evsum += (evtoc - evtic);

                                            rj[tcn->o+c] = 0;
                                            for (r = 0; r < pn; ++r) {
                                                rj[tcn->o+c] += rjTemp[po+r] * ev[ro+r];
                                            }
                                        }

                                        free(ev);
                                    }                                    
                                }

                                // compute element currJ of eigenvector i by multiplying currJ-th row of U with i-th column of U_0
                                if (ts == 0) {
                                    assert(taskid == MASTER);
                                    int k;
                                    xi[currJ] = 0;
                                    // store i-th eigenvector of U
                                    double* ev = malloc(n * sizeof(double));
                                    // get i-th eigenvector of U
                                    evtic = omp_get_wtime();
                                    getEigenVector(root, ev, i);
                                    evtoc = omp_get_wtime();
                                    //if (currJ > 15) printVector(ev, n);
                                    evsum += (evtoc - evtic);
                                    //#pragma omp parallel for default(shared) private(k) schedule(static)
                                    for (k = 0; k < pn; ++k) {
                                        xi[currJ] += rj[po+k] * ev[po+k];
                                    }
                                    free(ev);
                                }

                                po = tcn->o;
                                pn = tcn->n;
                                tcn = tcn->parent;
                            }
                        }

                        cnc = cn;
                        cn = cn->parent;
                        if (s == 0 || cn->taskid != taskid) // we can continue with next j
                            break;
                    }
                }
            }

            // compute relative residual (current eigenvector is now in xi
            if (taskid == MASTER) {
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
            etoc = omp_get_wtime();
            esum += (etoc - etic);
        } else {
            if (taskid == MASTER) {
                // we do not compute eigenvector i
                fprintf(f, "%20.19g\n", lambda);
            }
        }
        MPI_Barrier(comm.comm);
    }

    if (taskid == MASTER) {
        free(x);
        free(xi);
        // those to pointers might be still NULL
        free(SL);
        free(evToCompute.indices);
    }
    free(rj);
    free(rjTemp);

    if (taskid == MASTER) {
        if (f !=stdout) fclose(f);
    }

    if (taskid == MASTER) {
        printf("\n");
        printf("Required time for backtransformation: %f seconds\n", esum);
        printf("Required time eigenvector extraction within backtransformation: %f seconds; fraction: %.1f%%\n", evsum, 100*evsum/esum);
    }

    //MPI_Barrier(comm.comm);
    return 0;
}
