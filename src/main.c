#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <assert.h>
#include "mpi.h"
#include "mkl.h"

#include "helper.h"
#include "filehandling.h"

#define MASTER 0

void showHelp();

int main (int argc, char **argv)
{
    // ///////////////////////////
    // initialize MPI
    // ///////////////////////////

    int numtasks, taskid, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Get_processor_name(hostname, &len);

    // we don't want to use nested parallelism, because it yields to worse results, if it is not controlled by any intelligence
    omp_set_nested(0);

    // size of tridiagonal matrix
    int n;
    // symmetric tridiagonal matrix T is splitted into diagonal elements D and off-diagonal elements E
    double* D = NULL; // diagonal elements
    double* E = NULL; // off diagonal elements

    // name of output file
    char* outputfile = NULL;

    // some indices to use in for loops
    int i,j,k;

    if (taskid == MASTER) {
        // ///////////////////////////
        // parse command line arguments
        // ///////////////////////////

        // no parameters are given, thus print usage hints and close programm
        if (argc == 1) {
            showHelp();
            MPI_ABORT(MPI_COMM_WORLD, 0);
        }

        char* inputfile = NULL;

        /*
         * Scheme to used as specified by option -s
         */
        int usedScheme = 1;
        n = 1000; // size of predefined matrix

        int c;

        opterr = 0;

        while ((c = getopt (argc, argv, "hi:n:s:")) != -1)
            switch (c)
            {
            case 'h':
                showHelp();
                MPI_ABORT(MPI_COMM_WORLD, 0);
            case 'i':
                inputfile = optarg;
                break;
            case 's':
                usedScheme = atoi(optarg); // keep in mind, that atoi returns 0,  if string is not and integer
                if (usedScheme < 1 || usedScheme > 2) {
                    fprintf (stderr, "Invalid argument for option -s. See help.\n");
                    MPI_ABORT(MPI_COMM_WORLD, 1);
                }
                break;
            case 'n':
                n = atoi(optarg);
                if (n < 1) {
                    fprintf (stderr, "Invalid argument for option -n. See help.\n");
                    MPI_ABORT(MPI_COMM_WORLD, 1);
                }
                break;
            case '?':
                if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown option character `\\x%x'.\n", optopt);
                MPI_ABORT(MPI_COMM_WORLD, 1);
            default:
                MPI_ABORT(MPI_COMM_WORLD, 1);
            }

        // if there are more than one positional argument
        if (argc - optind > 1) {
            fprintf (stderr, "Invalid number of positional arguments. See help.\n");
            MPI_ABORT(MPI_COMM_WORLD, 1);
        }

        outputfile = argv[optind];

        // print settings
        if (inputfile != NULL)
            printf("Input file: %s\n", inputfile);
        else
            printf("Use a matrix of scheme %d with dimension %d\n", usedScheme, n);

        if (outputfile != NULL)
            printf("Output file: %s\n", outputfile);


        // ///////////////////////////
        // read or create matrix T
        // ///////////////////////////

        /*
         * How to store the matrix?
         *
         * Since this program only deals with symm. tridiagonal matrices as input matrices, we store them as a special case of
         * Intel's packed matrix scheme .
         * A symmetric tridiagonal matrix has the same sub- and superdiagonal. So we store it in row-major layout as an n array
         * of diagonal elements and an (n-1) array of off-diagonal elements.
         */

        if (inputfile != NULL) { // read matrix from file
            if (readSymmTriadiagonalMatrixFromSparseMTX(inputfile, &D, &E, &n) != 0)
                MPI_ABORT(MPI_COMM_WORLD, 2);
        } else {
            switch (usedScheme) {
            case 1:
                createMatrixScheme1(&D, &E, n);
                break;
            case 2:
                createMatrixScheme2(&D, &E, n);
                break;
            }
        }

        printf("\n");
        printf("Number of MPI tasks is: %d\n", numtasks);

        for (i = 0; i < n-1; ++i) {
            assert(D[i] != 0);
            assert(E[i] != 0);
        }
        assert(D[n-1] != 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("   Task %d is running on node %s, which has %d available processors.\n", taskid, hostname, omp_get_num_procs());
    MPI_Barrier(MPI_COMM_WORLD);

    // ///////////////////////////
    // ///////////////////////////
    // Cuppen's Algorithm to obtain all eigenpairs
    // ///////////////////////////
    // ///////////////////////////

    double tic = omp_get_wtime();

    MPI_Bcast(&n,1,MPI_INT,MASTER,MPI_COMM_WORLD);

    // ///////////////////////////
    // Divide phase
    // ///////////////////////////
    /*
     * The goal of the divide phase is to create a binary tree which is as balanced as possible and contains nearly equal sized leaves.
     *
     * Here is how the splitting works: let p = 7 (numtasks) (smallest power of two greater than 7 is 8)
     * In the first stage, only the task with (taskid % 8 == 0) should perform a split (which is the MASTER).
     * it should send the result to taskid + 8/2 => 4
     *
     * In the second stage, only tasks with taskid 8/2 = 4 should perform a split => 0, 4.
     * The results are send to nodes with taskid: sender_taskid+4/2 => 2, 6
     *
     * In the third stage, only tasks with taskid 4/2 = 2 should perform a split => 0, 2, 4, 6. (6 can't perform a split)
     * The results are send to nodes with taskid: sender_taskid+2/2 => 1,2,3,4,5,6,7
     *
     * Note, since 2^(k-1) < p <= 2^k, the minimum depth of each binary tree is k. Our tree will always have depth k.
     */

    /*
     * How to split the matrix. Assume:
     * D = [1,2,3,4,5,6,7,8]
     * E = [a,b,c,d,e,f,g]
     * If we want to split between 4 and 5, the we want to have the following matrices
     * E1 = [a,b,c], E2 = [e,f,g]
     * The off diagonal element d would be the beta, which we have to eliminate in the splitting process.
     * So, the indices of the off-diagonals in the splitted matrix have the same start index as the diagonal elements but have one less element
     *
     * So, we split T into T1, T2 (note, the difference in notation: T1 and T2 have an hat on top in the book).
     * The last diagonal element in T1 differs from the orignal part by subtracting theta * beta.
     * The first diagonal element in T2 differs from the orignal part by subtracting theta^-1 * beta.
     */

    // all tasks that have zero remainder when computing (taskid % modulus) do a split in the current stage
    // find smallest power of two greater than numtasks
    int modulus = 1;
    int maxModulus = 1;
    int numSplitStages = 0; // number of tree levels where splits are performed
    while (modulus < numtasks) {
        maxModulus *= 2;
        numSplitStages++;
    }
    modulus = maxModulus;

    // at each split that we perform, we have to keep track of the lost beta entry
    /*
     * Since there are less than log(numtaks) stages, I allocate more the betas than necessary
     * (note, actually not each task performs a split on each stage)
     */
    int betas[numSplitStages];
    int thetas[numSplitStages];
    // TODO: choose meaningful values of theta
    for (i = 0; i < numSplitStages; ++i)
        thetas[i] = 1;

    // Note, our goal is to have equally sized leaves
    int leafSize, sizeRemainder;
    if (taskid == 0) {
        leafSize = n / numtasks; // FIXME: if leaf size is too small, we have to use less nodes for computation
        sizeRemainder = n % numtasks;
    }
    MPI_Bcast(&leafSize,1,MPI_INT,MASTER,MPI_COMM_WORLD);
    MPI_Bcast(&sizeRemainder,1,MPI_INT,MASTER,MPI_COMM_WORLD);
    // the actual leafsize of the current task
    int nl = leafSize + (taskid < sizeRemainder ? 1 : 0);
    // helper variables
    // size of T in left resp. right subtree
    int n1,n2;
    // number of leaves in the right subtree
    int numLeavesRight;

    // stage in divide tree
    int s = 0;
    for (s = 0; modulus > 1; s++) {

        // if task is to perform a split of T (Note: in the first stage, only MASTER statisfies the condition
        if (taskid % modulus == 0) {
            // Compute size of left and right subtree depending on the number of childrens in this tree.
            // The left subtree will always be a fully balanced tree (so it will have modulus/2 leaves).
            // The right subtree has min(numtasks-(taskid+modulus/2) , modulus/2) leaves.
            numLeavesRight = min(numtasks-(taskid+modulus/2), modulus/2);

            n1 = modulus/2 * leafSize;
            n2 = numLeavesRight * leafSize;

            // split the remaining lines equally on the leaves
            n1 += max(0, min(sizeRemainder-taskid, modulus/2));
            n2 += max(0, min(sizeRemainder-(taskid+modulus/2), numLeavesRight));

            //printf("Task %d: Splits into (Task %d: %d; Task %d: %d)\n", taskid, taskid, n1, taskid + modulus/2, n2);

            // save beta for later conquer phase and modify diagonal elements
            betas[s] = E[n1-1];
            // modify last diagonal element of T1
            D[n1-1] -= thetas[s] * betas[s];
            // modify first diagonal element of T2
            D[n1] -= 1.0/thetas[s] * betas[s];

            // send size and second half of matrix
            MPI_Send(&n2, 1, MPI_INT, taskid + modulus/2, 1, MPI_COMM_WORLD);
            MPI_Send(D+n1, n2, MPI_DOUBLE, taskid + modulus/2, 2, MPI_COMM_WORLD);
            MPI_Send(E+n1, n2-1, MPI_DOUBLE, taskid + modulus/2, 3, MPI_COMM_WORLD);
        }

        // if task is receiver of a subtree in this step
        if (taskid % modulus != 0 && taskid % (modulus/2) == 0) {
            // receive size of matrix to receive
            MPI_Recv(&n, 1, MPI_INT, taskid-modulus/2, 1, MPI_COMM_WORLD, &status);

            // receive matrix
            assert(D == NULL && E == NULL);
            D = malloc(n * sizeof(double));
            E = malloc((n-1) * sizeof(double));
            MPI_Recv(D, n, MPI_DOUBLE, taskid-modulus/2, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(E, n-1, MPI_DOUBLE, taskid-modulus/2, 3, MPI_COMM_WORLD, &status);
        }

        modulus /= 2;
    }

    /*
     * Some final remarks of the divide phase:
     * The size of the current leave is in nl.
     * The actual allocated memory is still stored in n (which will be needed in conquer phase)
     */

    // ///////////////////////////
    // Compute eigenpairs of leaves using QR algorithm
    // ///////////////////////////

    // orthonormal where the columns are eigenvectors
    double* Q1 = malloc(nl*nl * sizeof(double));

    int ret =  LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', nl, D, E, Q1, nl);
    assert(ret == 0);

    // ///////////////////////////
    // Conquer phase
    // ///////////////////////////

    // sizes of Q matrices
    int nq1 = nl, nq2;
    double* Q2 = NULL;

    for (s = numSplitStages-1; modulus < maxModulus; s--) {

        // if task computed a spectral decomposition in the last stage (but not in the current
        if (taskid % modulus == 0 && taskid % (modulus*2) != 0) {
            // send eigenvectors and eigenvalues to parent node in tree
            MPI_Send(&nq1, 1, MPI_INT, taskid - modulus, 4, MPI_COMM_WORLD);
            MPI_Send(D, nq1, MPI_DOUBLE, taskid - modulus, 5, MPI_COMM_WORLD);
            MPI_Send(Q1, nq1*nq1, MPI_DOUBLE, taskid - modulus, 6, MPI_COMM_WORLD);

            // Q1 is not needed anymore
            free(Q1);
            Q1 = NULL;
        }


        // if task combines two splits in this stage
        if (taskid % (modulus*2) == 0) {

            // receive size of matrix to receive
            MPI_Recv(&nq2, 1, MPI_INT, taskid + modulus, 4, MPI_COMM_WORLD, &status);

            // receive eigenvectors and eigenvalues from right child in tree
            MPI_Recv(D, nq2, MPI_DOUBLE, taskid + modulus, 5, MPI_COMM_WORLD, &status);
            Q2 = malloc(nq2*nq2 * sizeof(double));
            MPI_Recv(Q2, nq2*nq2, MPI_DOUBLE, taskid + modulus, 6, MPI_COMM_WORLD, &status);

            // TODO
        }

        }

        modulus *= 2;
    }


    // ///////////////////////////
    // End of algorithm
    // ///////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);
    double toc = omp_get_wtime();
    if (taskid == MASTER) {
        free(D);
        free(E);

        printf("\n");
        printf("Elapsed time in %f seconds\n", toc-tic);

        if (outputfile != NULL) {
            // TODO print file
        }
    }

    MPI_FINALIZE();
    return 0;
}

/**
 * @brief showHelp Show usage details to the user
 */
void showHelp() {
    printf("\n");
    printf("USAGE cuppens [options] [outputfile]\n");
    printf("\n");
    printf("The program can compute all the eigenpairs of a matrix on a parallel machine\n");
    printf("by using cuppens algorithm\n");
    printf("The results can be written into an outputfile, if specified.\n");
    printf("\n");
    printf("OPTIONS\n");
    printf(" -h\n");
    printf("    Show help.\n");
    printf(" -i FILENAME\n");
    printf("    The name of a file which contains a tridiagonal matrix in mtx format.\n");
    printf("    The eigenvalues of this matrix will then be computed.\n");
    printf(" -s NUM\n");
    printf("    If you want to compute the eigenvalues of a predefined matrix, you may.\n");
    printf("    use this option to define the scheme of the matrix.\n");
    printf("    1 - eigenvalue i has the form: 4 - 2*cos((PI*i)/(n+1)) \n");
    printf("    2 - Matrix will have the tridiagonal form [-1,d_i,-1] where the diagonal\n");
    printf("        elements will be evenly spaced in the interval [1,100] \n");
    printf("    If option i is used, then this option will be ignored.\n");
    printf(" -n NUM\n");
    printf("    Specify the dimension of the matrix chosen with option -s.\n");
    printf("\n");
}
