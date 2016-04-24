#include "eigenvalues.h"

#include <omp.h>
#include <math.h>
#include <assert.h>
#include "mkl.h"

inline double secularEquation(double lambda, double roh, double* z, double* D, int n, int* G) {
    double sum = 0;
    int i;
#pragma omp parallel for default(shared) private(i) schedule(static) reduction(+:sum)
    for (i = 0; i < n; ++i)
	if (G[i] < 0)
	    sum += z[i]*z[i] / (D[i]-lambda);
    return 1+roh*sum;
}

void computeEigenvalues(EVRepNode* node, MPIHandle mpiHandle) {
    // abbreviations
    int taskid = mpiHandle.taskid;
    int numtasks = mpiHandle.numtasks;

    double* D = NULL;
    double* z = NULL;
    double* L = NULL;
    int* G = NULL;
    int* P = NULL;
    double roh;
    int n;

    if (taskid == node->taskid) {
	node->G = malloc(n * sizeof(int));
	node->P = malloc(n * sizeof(int));
	/*
	 * Store eigenvalues in new array (do not overwrite D), since the elements in D are needed later on to compute the eigenvectors)S
	 */
	node->L = malloc(n * sizeof(double));

	D = node->D;
	z = node->z;
	L = node->L;
	G = node->G;
	P = node->P;
	roh = node->beta * node->theta;
	n = node->n;
	node->numGR = 0;
    }

    // we don't use parallelism yet, so just return if other task
    if (taskid != node->taskid) {
	return;
    }

    assert(roh != 0);

    // copy and sort diagonal elements
    DiagElem* SD = malloc(n * sizeof(DiagElem));
    int i;
#pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i) {
	SD[i].e = D[i];
	SD[i].i = i;
    }
    qsort(SD, n, sizeof(DiagElem), compareDiagElem);
    double eps = 1e-14;

    // calculate Givens rotation
    /* G is a vector that keeps track of Givens rotation for SD 
     * Since SD has been sorted ascendingly, we should always make the  off-diagonal 
     * element that corresponds to the smaller diagonal element to be zero*/

    for (i = 0; i < n - 1; i++){
      if (fabs(SD[i + 1].e - SD[i].e) < eps) {
        G[SD[i].i] = SD[i + 1].i;
        P[node->numGR] = SD[i].i;
        node->numGR++;
      }
      else 
        G[SD[i].i] = -1;
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
#pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i) { // for each eigenvalue
      double lambda = 0;
      double a, b; // interval boundaries
      double fa, flambda, fb; // function values

      int ind = SD[i].i;
      int prevNonZeroIdx;
      double di = SD[i].e;

      if (G[ind] >= 0) {
        L[ind] = di;
      } else {
        // set initial interval
        if (roh < 0) {
          if (i == 0) {
            a = di - normZ;
            int j = 0;
            while(secularEquation(a, roh, z, D, n, G) < 0) {
              a -= normZ;
              assert(++j < 100);
            }
          } else {
            prevNonZeroIdx = i - 1;
            while(G[SD[prevNonZeroIdx].i] > 0) // TODO: Take the first element is zero into consideration
              prevNonZeroIdx = prevNonZeroIdx - 1;
            a = SD[prevNonZeroIdx].e;
          }
          b = di;
        } else {
          a = di;
          if (i == n-1) {
            b = di + normZ;
            int j = 0;
            while(secularEquation(b, roh, z, D, n, G) < 0) {
              b += normZ;
              assert(++j < 100);
            }
          } else {
            prevNonZeroIdx = i + 1;
            while(G[SD[prevNonZeroIdx].i] > 0) // TODO: Take the last element is zero into consideration
              prevNonZeroIdx = prevNonZeroIdx + 1;
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

          //if (j==10)
          //    printf("interval: %g, %g, %g, %g, %g, %g\n", fa, flambda, fb, a, lambda, b);

          if (flambda == 0 || (b-a)/2 < eps)
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
    //    printVector(L,n);
}

double* computeNormalizationFactors(double* D, double* z, double* L, int *G, int n) {
  double *N = malloc(n * sizeof(double));

  int i, j;
  double tmp;
#pragma omp parallel for default(shared) private(i,j,tmp) schedule(static)
  for (i = 0; i < n; ++i) {
    if (G[i] > 0) {
      N[i] = 1;
    } else {
      N[i] = 0;
      for (j = 0; j < n; ++j) {
        if (G[j] > 0) {
          tmp = D[j]-L[i];
          N[i] += z[j]*z[j] / (tmp*tmp);
        }
      }
      N[i] = sqrt(N[i]);
    }
  }

  return N;
}

void getEigenVector(EVRepNode *node, double* ev, int i) {
  double* D = node->D;
  double* z = node->z;
  double* L = node->L;
  double* N = node->N;
  int* G = node->G;
  int* P = node->P;
  double roh = node->beta * node->theta;
  int n = node->n;
  int numGR = node->numGR;


  // TODO compute i-th eigenvector and store in ev
  int j;
  if(G[i] > 0) {
    for (j = 0; j < n; j++) {
      if (j == i){
        ev[j] = 1;
      } else {
        ev[j] = 0;
      }
    }
  } else {
    for (j = 0; j < n; j ++) 
      ev[j] = z[j] / ((D[j] - L[i]) * N[i]);
  }


  /* recover the original rank-one update
   * apply the inverse of Givens rotation from outside to inside
   * for example, if the Givens rotation on the original problem is 
   * G3 * G2 * G1 * (D + zz') G1' * G2' * G3' 
   * The order here should be G3^-1, G2^-1 nad G1^-1 */

#pragma omp parallel for default(shared) private(j) schedule(static)
  for (j = numGR - 1; j >= 0 ; j--) {
    double r, s, c;
    int a, b;
    double tmpi, tmpj;

    a = P[j];     
    b = G[a]; 
    r = sqrt(z[a] * z[a] + z[b] * z[b]);
    c = z[b] / r;
    s = z[a] / r;
    tmpi = c * ev[a] + s * ev[b];
    tmpj = -s * ev[a] + c * ev[b];
    ev[a] = tmpi;
    ev[b] = tmpj;
  }
}

