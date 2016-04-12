#include "eigenvalues.h"

#include <omp.h>
#include <math.h>
#include <assert.h>
#include "mkl.h"

#include "helper.h"

/* When I sort the diagonal elements, I need a mapping back to the original order */
struct diagElem {
    double e; // element
    int i; // index
};

int compare( const void* a, const void* b)
{
     struct diagElem e1 = * ( (struct diagElem*) a );
     struct diagElem e2 = * ( (struct diagElem*) b );

     if ( e1.e == e2.e ) return 0;
     else if ( e1.e < e2.e ) return -1;
     else return 1;
}

inline double secularEquation(double lambda, double roh, double* z, double* D, int n) {
    double sum = 0;
    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static) reduction(+:sum)
    for (i = 0; i < n; ++i)
        sum += z[i]*z[i] / (D[i]-lambda);
    return 1+roh*sum;
}

double* computeEigenvalues(double* D, double* z, int n, double beta, double theta) {
    /*
     * Store eigenvalues in new array (do not overwrite D), since the elements in D are needed later on to compute the eigenvectors)S
     */
    double* L = malloc(n * sizeof(double));

    double roh = beta * theta;
    assert(roh != 0);

    // copy and sort diagonal elements
    struct diagElem* SD = malloc(n * sizeof(struct diagElem));
    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < n; ++i) {
        SD[i].e = D[i];
        SD[i].i = i;
    }
    qsort(SD, n, sizeof(struct diagElem), compare);

    /* Note, if roh > 0, then the last eigenvalue is behind the last d_i
     * If roh < 0, then the first eigenvalue is before the first d_i */

    // use norm of z as an approximation to find the first resp. last eigenvalue
    double normZ = cblas_dnrm2(n, z, 1);

    /******************
     * Simple Bisection algorithm
     * ****************/
    long maxIter = 10000;
    double eps = 1e-14;
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

        int ind = SD[i].i;
        double di = SD[i].e;

        // set initial interval
        if (roh < 0) {
            if (i == 0) {
                a = di - normZ;
                int j = 0;
                while(secularEquation(a, roh, z, D, n) < 0) {
                    a -= normZ;
                    assert(++j < 100);
                }
            } else {
                a = SD[i-1].e;
            }
            b = di;
        } else {
            a = di;
            if (i == n-1) {
                b = di + normZ;
                int j = 0;
                while(secularEquation(b, roh, z, D, n) < 0) {
                    b += normZ;
                    assert(++j < 100);
                }
            } else {
                b = SD[i+1].e;
            }
        }

        int j = 0;
        while (++j < maxIter) {
            lambda = (a+b) / 2;

            double fa = secularEquation(a, roh, z, D, n);
            double flambda = secularEquation(lambda, roh, z, D, n);
            double fb = secularEquation(b, roh, z, D, n);
            if (j==10)
                printf("interval: %g, %g, %g, %g, %g, %g\n", fa, flambda, fb, a, lambda, b);

            if (flambda == 0 || (b-a)/2 < eps)
                break;

            // if sign(a) == sign(lambda)
            if ((fa >= 0 && flambda >= 0) || (fa < 0 && flambda < 0))
                a = lambda;
            else
                b = lambda;
        }
        L[ind] = lambda;
        printf("f(%g) = %g\n", lambda, secularEquation(lambda, roh, z, D, n));
    }

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
