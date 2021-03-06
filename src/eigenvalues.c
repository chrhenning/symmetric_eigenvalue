#include "eigenvalues.h"

#include <omp.h>
#include <math.h>
#include <assert.h>
#include "mkl.h"

inline double secularEquation(double lambda, double roh, double* z, double* D, int n, int* G) {
    double sum = 0;
    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static) reduction(+:sum)
    for (i = 0; i < n; ++i) {
        if (G[i] == -1)
            sum += z[i]*z[i] / (D[i]-lambda);
    }
    return 1+roh*sum;
}

void computeEigenvalues(EVRepNode* node, MPIHandle mpiHandle) {
    // abbreviations
    int taskid = mpiHandle.taskid;
    int numtasks = mpiHandle.numtasks;

    int i;

    double* D = NULL;
    double* z = NULL;
    double* L = NULL;
    double* C = NULL;
    double* S = NULL;
    int* G = NULL;
    int* P = NULL;
    double roh;
    int n;

    if (taskid == node->taskid) {
        n = node->n;
        node->G = malloc(n * sizeof(int));
        node->P = malloc(n * sizeof(int));
        node->C = malloc(n * sizeof(double));
        node->S = malloc(n * sizeof(double));
        /*
     * Store eigenvalues in new array (do not overwrite D), since the elements in D are needed later on to compute the eigenvectors)S
     */
        node->L = malloc(n * sizeof(double));

        D = node->D;
        z = node->z;
        L = node->L;
        G = node->G;
        P = node->P;
        C = node->C;
        S = node->S;
        roh = node->beta * node->theta;
        node->numGR = 0;

        // initialize G properly, because later on I perform tests like, G[i] != -2, where G[i] has a random value at this time
        #pragma omp parallel for default(shared) private(i) schedule(static)
        for (i = 0; i < n; ++i)
            G[i] = -1;
    }

    // we don't use parallelism yet, so just return if other task
    if (taskid != node->taskid) {
        return;
    }

    assert(roh != 0);

    // copy and sort diagonal elements
    DiagElem* SD = malloc(n * sizeof(DiagElem));
    double eps = 1e-6;

    // scan z for zero element and mark it in G with -2
#pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; i++) {
        if (fabs(z[i]) < eps) {
            //printf("Deflation happens (z) for index %d\n", i);
            G[i] = -2;
        }
    }

    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i) {
        SD[i].e = D[i];
        SD[i].i = i;
    }
    qsort(SD, n, sizeof(DiagElem), compareDiagElem);

    // calculate Givens rotation
    /* G is a vector that keeps track of Givens rotation for SD
   * Since SD has been sorted ascendingly, we should always make the  off-diagonal
   * element that corresponds to the smaller diagonal element to be zero*/

    int a, b;
    double c, s ,r, tmpi, tmpj;
    int nextNonZero;
    for (i = 0; i < n-1; i++){
        if (G[SD[i].i] != -2) { // for those elements correspond to non-zero z
            nextNonZero = i + 1;
            while (G[SD[nextNonZero].i] == -2) {
                if (++nextNonZero == n) break;
            }
            if (nextNonZero >= n) {
                G[SD[i].i] = -1;
                continue;
            }

            if (fabs(SD[nextNonZero].e - SD[i].e) < 1e-5) {

              a = SD[i].i;
              b = SD[nextNonZero].i;
              r = sqrt(z[a] * z[a] + z[b] * z[b]);
              //printf("Deflation happens (d) for element %g (r=%g)\n", SD[i].i, r);
              c = z[b] / r;
              s = z[a] / r;
              C[node->numGR] = c;
              S[node->numGR] = s;


              G[SD[i].i] = SD[nextNonZero].i;
              z[SD[nextNonZero].i] = r;
              z[SD[i].i] = 0;
              P[node->numGR] = SD[i].i;

              tmpi = c * c * SD[i].e + s * s * SD[nextNonZero].e;
              tmpj = s * s * SD[i].e + c * c * SD[nextNonZero].e;
              SD[i].e = tmpi;
              SD[nextNonZero].e = tmpj;
              D[a] = tmpi;
              D[b] = tmpj;
              node->numGR++;
            }
        }
    }

    /* Note, if roh > 0, then the last eigenvalue is behind the last d_i
     * If roh < 0, then the first eigenvalue is before the first d_i */

    // use norm of z as an approximation to find the first resp. last eigenvalue
    double normZ = cblas_dnrm2(n, z, 1);

    /******************
     * Simple Bisection algorithm
     * ****************/
    long maxIter = 10000;
    /*
       N ← 1
       While N ≤ NMAX # limit iterations to prevent infinite loop
       c ← (a + b)/2 # new midpoint
       If f(c) = 0 or (b – a)/2 < TOL then # solution found
       Output(c)
       Stop
       EndIf
       N ← N + 1 # increment step counter
       If sign(f(c)) = sign(f(a)) then a ← c else b ← c # new interval
       EndWhile
       */
    int boundaryHandled = 0;
    //#pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i) { // for each eigenvalue
        double lambda = 0;
        double a, b; // interval boundaries
        double fa, flambda, fb; // function values

        int ind = SD[i].i;
        int prevNonZeroIdx;
        double di = SD[i].e;

        if (G[ind] != -1) {
            L[ind] = di;
        } else {
            // set initial interval
            if (roh < 0) {
                if (!boundaryHandled) {
                    a = di - normZ;
                    int j = 0;
                    while(secularEquation(a, roh, z, D, n, G) < 0) {
                        a -= normZ;
                        assert(++j < 100);
                    }
                    boundaryHandled = 1;
                } else {
                    prevNonZeroIdx = i - 1;
                    while(G[SD[prevNonZeroIdx].i] != -1) // TODO: Take the first element is zero into consideration
                        prevNonZeroIdx--;
                    a = SD[prevNonZeroIdx].e;
                }
                b = di;
            } else {
                a = di;

                prevNonZeroIdx = i + 1;
                while(prevNonZeroIdx < n && G[SD[prevNonZeroIdx].i] != -1) { // TODO: Take the last element is zero into consideration
                    prevNonZeroIdx++;
                }

                if (prevNonZeroIdx >= n) {
                    b = di + normZ;
                    int j = 0;
                    while(secularEquation(b, roh, z, D, n, G) < 0) {
                        b += normZ;
                        assert(++j < 100);
                    }
                } else {
                    b = SD[prevNonZeroIdx].e;
                }
            }

        int j = 0;
        while (++j < maxIter) {

          // new lambda
          lambda = (a+b) / 2;
          // compute current function values
          fa = secularEquation(a, roh, z, D, n, G);
          flambda = secularEquation(lambda, roh, z, D, n, G);
          //fb = secularEquation(b, roh, z, D, n, G);

          // if a function value is inf, then it has probably not the right sign
          // initial function values are in +/- infinity, depending on the gradiend of the secular equation
          if (fa == INFINITY || fa == -INFINITY)
            fa = (roh > 0 ? -INFINITY : INFINITY);

          //if (fb == INFINITY || fb == -INFINITY)
          //   fb = (roh > 0 ? INFINITY : -INFINITY);

//          if (isnan(a) || isnan(b) || isnan(lambda)) {
//              printf("interval: %g, %g, %g, %g, %g, %g %d\n", fa, flambda, fb, a, lambda, b, j);
//              printVector(z,n);
//              printf("%g\n",normZ);
//              MPI_ABORT(MPI_COMM_WORLD, 1);
//          }

          if (flambda == 0 || (b-a)/2 < 1e-14)
            break;

          // if sign(a) == sign(lambda)
          if ((fa >= 0 && flambda >= 0) || (fa < 0 && flambda < 0))
            a = lambda;
          else
            b = lambda;
        }
        L[ind] = lambda;
        //printf("f(%g) = %g\n", lambda, secularEquation(lambda, roh, z, D, n, G));
      }
    }


    free(SD);

    //    printVector(z,n);
    //    printVector(D,n);
    //printVector(L,n);
}

void computeNormalizationFactors(EVRepNode *node) {
  int* G = node->G;
  int n = node->n;

  node->N = malloc(n * sizeof(double));
  double* N = node->N;
  // set normalization vector to 1, to compute unnormalized eigenvectors
  int i;
  #pragma omp parallel for default(shared) private(i) schedule(static)
  for (i = 0; i < n; ++i) {
    N[i] = 1;
  }

  // actual normalization vector
  double* Ntemp = malloc(n * sizeof(double));

  // current ev
  double* ev = malloc(n * sizeof(double));

  for (i = 0; i < n; ++i) {
    if (G[i] != -1) {
      Ntemp[i] = 1;
    } else {
      getEigenVector(node, ev, i);
      Ntemp[i] = cblas_dnrm2(n, ev, 1);
    }
  }

  node->N = Ntemp;

  free(ev);
  free(N);
}

void getEigenVector(EVRepNode *node, double* ev, int i) {
  double* D = node->D;
  double* z = node->z;
  double* L = node->L;
  double* N = node->N;
  double* C = node->C;
  double* S = node->S;
  int* G = node->G;
  int* P = node->P;
  int n = node->n;
  int numGR = node->numGR;


  // TODO compute i-th eigenvector and store in ev
  int j;
  if(G[i] != -1) {
      #pragma omp parallel for default(shared) private(j) schedule(static)
    for (j = 0; j < n; j++) {
      if (j == i){
        ev[j] = 1;
      } else {
        ev[j] = 0;
      }
    }
  } else {
      #pragma omp parallel for default(shared) private(j) schedule(static)
    for (j = 0; j < n; j ++)
      if (G[j] < -1) {
        ev[j] = 0;
      } else {
        ev[j] = z[j] / ((D[j] - L[i]) * N[i]);
//        if (isinf(ev[j])) {
//            printVector(ev,n);
//            printVector(D,n);
//            printVector(L,n);
//            printVector(N,n);
//            printVector(z,n);
//            printf("test %d, %.20g %.20g\n", G[j], D[j],L[j]);
//            break;
//        }
      }
  }



  /* recover the original rank-one update
   * apply the inverse of Givens rotation from outside to inside
   * for example, if the Givens rotation on the original problem is
   * G3 * G2 * G1 * (D + zz') G1' * G2' * G3'
   * The order here should be G3^-1, G2^-1 nad G1^-1 */

  // don't use openMP fur this loop, the rotations have to be applied in a certain order
  for (j = numGR - 1; j >= 0 ; j--) {
    int a, b;
    double s, c;
    double tmpi, tmpj;

    a = P[j];
    b = G[a];
    c = C[j];
    s = S[j]; // TODO: probably it's better to store s as well, since the product c*c halves the precision (e^-10 * e^-10 = e^-20)

    tmpi = c * ev[a] + s * ev[b];
    tmpj = -s * ev[a] + c * ev[b];
    ev[a] = tmpi;
    ev[b] = tmpj;
  }
}

