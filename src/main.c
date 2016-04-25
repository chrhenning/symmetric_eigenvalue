#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <assert.h>
#include "mkl.h"

#include "helper.h"
#include "filehandling.h"
#include "eigenvalues.h"
#include "backtransformation.h"

void showHelp();

int main (int argc, char **argv)
{
    /**********************
     * initialize MPI
     **********************/

    int numtasks, taskid, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Get_processor_name(hostname, &len);

    // store information, necessary to use MPI, in extra struct, that we can pass to function calls
    MPIHandle mpiHandle;
    mpiHandle.comm = MPI_COMM_WORLD;
    mpiHandle.numtasks = numtasks;
    mpiHandle.taskid = taskid;

    // we don't want to use nested parallelism, because it yields to worse results, if it is not controlled by any intelligence
    omp_set_nested(0);

    // If we case that we want to end the program while parsing the option (for example when showing the usage hints -h),
    // then I don't want to abort the program. So I need this variable to exit it properly
    int endProgram = 0;

    // size of tridiagonal matrix
    int n;
    // symmetric tridiagonal matrix T is splitted into diagonal elements D and off-diagonal elements E
    double* D = NULL; // diagonal elements
    double* E = NULL; // off diagonal elements

    // copies of D and E, which are needed to write the output file (we need the original T)
    // Note, even if a copy wastes memory, it's much faster then reading the matrix again from file later on when we need it
    double *OD, *OE;

    // helper variable to easily access current node in tree
    EVRepNode* currNode = NULL;

    // for time measurement of eigenvalue computation
    double tic, toc;
    // time measurement for roott finding
    double rsum = 0, rtic, rtoc;

    // name of output file
    char* outputfile = NULL;
    // name of the file, where the user defines which eigenvectors we are going to compute
    char* evFile = NULL;
    int computeEV = 0; // flag, if there should be any eigenvectors computed at the end
    int writeOutput = 0; // flag

    // some indices to use in for loops
    int i,j,k;

    if (taskid == MASTER) {
        /**********************
         * parse command line arguments
         **********************/

        // no parameters are given, thus print usage hints and close programm
        if (argc == 1) {
            showHelp();
            endProgram = 1;
            goto StartOfAlgorithm;
        }

        char* inputfile = NULL;

        /*
         * Scheme to used as specified by option -s
         */
        int usedScheme = 1;
        n = 1000; // size of predefined matrix

        int c;

        opterr = 0;

        while ((c = getopt (argc, argv, "hi:n:s:e::")) != -1)
            switch (c)
            {
            case 'h':
                showHelp();
                endProgram = 1;
                goto StartOfAlgorithm;
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
            case 'e':
                computeEV = 1;
                if (optarg)
                    evFile = optarg;
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

        if (computeEV) {
            if (evFile != NULL)
                printf("Compute the eigenvectors defined in: %s\n", evFile);
            else
                printf("Program will compute all eigenvectors\n");
        }

        if (outputfile != NULL) {
            writeOutput = 1;
            printf("Output file: %s\n", outputfile);
        }


        /**********************
         * read or create matrix T
         **********************/

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

        // create copies of E and D
        OD = malloc(n * sizeof(double));
        OE = malloc((n-1) * sizeof(double));
        memcpy(OD, D, n*sizeof(double));
        memcpy(OE, E, (n-1)*sizeof(double));
    }

    StartOfAlgorithm:

    //if (taskid == MASTER)
    //    printTridiagonalMatrix(D,E,n);

    // in case the MASTER tells us we should end here
    MPI_Bcast(&endProgram,1,MPI_INT,MASTER,MPI_COMM_WORLD);
    if (endProgram == 1) {
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&writeOutput,1,MPI_INT,MASTER,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("   Task %d is running on node %s, which has %d available processors.\n", taskid, hostname, omp_get_num_procs());
    MPI_Barrier(MPI_COMM_WORLD);

    /**********************
     **********************
     * Cuppen's Algorithm to obtain all eigenpairs
     **********************
     **********************/

    tic = omp_get_wtime();

    MPI_Bcast(&n,1,MPI_INT,MASTER,MPI_COMM_WORLD);

    /**********************
     * Divide phase
     **********************/
    if (taskid == MASTER)
        printf("Start divide phase ...\n");
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
    // depth of the tree, thus the tree has 'depth' stages with at maximum 2^(depth-1) leaves
    int treeDepth = 1;
    while (maxModulus < numtasks) {
        maxModulus *= 2;
        treeDepth++;
    }
    modulus = maxModulus;
    // helper variable
    int numSplitStages = treeDepth-1; // number of stages, where splits are performed

    /*
     * In this struct we store the whole information to reconstruct the eigenvectors.
     * Thus, for each node in the tree, the struct must store the information of
     * either the whole eigenvector matrix of its T (if leave node) or the vectors
     * representing the eigenvector matrix of the rank-one perturbation of its T,
     * where T is the tridiagonal matrix assigned to a node in the tree.
     *
     * Of course, each of these eigenvector matrix representation is only stored in one task.
     * If it's not stored on the current node, then we store at least the taskid, which knows,
     * where it is stored for all our child nodes.
     * Hence, only the MASTER node (root of tree) can reconstruct the whole eigenvector matrix.
     */
    EVRepTree evTree = initEVRepTree(treeDepth, numtasks, n);
    if (taskid == MASTER)
        assert(evTree.t[0].s[0].n == n);

    // If this task performs a split, then it stores the taskid of the right child in this array (the left child is the task itself)
    // If there is no split, then the value is -1
    int rightChild[numSplitStages];
    // If this task was splitted in the stage before, then we store here the taskid of the parent, else -1
    // More precisely, if there was a split involving this node at stage s, the we store in parent[s] the taskid of the node performing the split
    int parent[numSplitStages];
    // initialize above arrays
    #pragma omp parallel for default(shared) private(i) schedule(static)
    for (i = 0; i < numSplitStages; ++i) {
        rightChild[i] = -1;
        parent[i] = -1;
    }

    // Note, our goal is to have equally sized leaves
    int leafSize, sizeRemainder;
    if (taskid == 0) {
        leafSize = n / numtasks; // FIXME: if leaf size is too small, we have to use less nodes for computation
        sizeRemainder = n % numtasks;
    }
    MPI_Bcast(&leafSize,1,MPI_INT,MASTER,MPI_COMM_WORLD);
    MPI_Bcast(&sizeRemainder,1,MPI_INT,MASTER,MPI_COMM_WORLD);
    if (taskid == MASTER) printf("Average leaf size will be %d\n", leafSize);
    // the actual leafsize of the current task
    int nl = leafSize + (taskid < sizeRemainder ? 1 : 0); // FIXME: delete me (stored in tree)
    // helper variables
    // size of T in left resp. right subtree
    int n1,n2;

    // stage in divide tree
    int s = 0;
    for (s = 0; modulus > 1; s++) {
        assert(s < treeDepth);

        // if task is to perform a split of T (Note: in the first stage, only MASTER statisfies the condition
        if (taskid % modulus == 0) {
            // get current node in tree
            currNode = accessNode(&evTree, s, taskid);

            rightChild[s] = taskid + modulus/2;
            parent[s] = taskid; // left child will stay on this node

            n1 = currNode->left->n;
            n2 = currNode->right->n;

            if (currNode->left != currNode->right) { // an actual split is performed
                assert(n2 > 0);
                assert(currNode->left->taskid != currNode->right->taskid);

                //printf("Task %d: Splits into (Task %d: %d; Task %d: %d) in stage %d\n", taskid, taskid, n1, rightChild[s], n2, s);

                assert(s == 0 || parent[s-1] == currNode->parent->taskid);
                assert(rightChild[s] == currNode->right->taskid);

                // at each split that we perform, we have to keep track of the lost beta entry
                // save beta for later conquer phase and modify diagonal elements
                currNode->beta = E[n1-1];
                // TODO: choose more meaningful theta
                currNode->theta = 1;
                // modify last diagonal element of T1
                D[n1-1] -= currNode->theta * currNode->beta;
                // modify first diagonal element of T2
                D[n1] -= 1.0/currNode->theta * currNode->beta;

                // send size and second half of matrix
                MPI_Send(&n2, 1, MPI_INT, rightChild[s], 1, MPI_COMM_WORLD);
                MPI_Send(D+n1, n2, MPI_DOUBLE, rightChild[s], 2, MPI_COMM_WORLD);
                MPI_Send(E+n1, n2-1, MPI_DOUBLE, rightChild[s], 3, MPI_COMM_WORLD);
            } else {
                rightChild[s] = -1;
            }
        }

        // if task is receiver of a subtree in this step
        if (taskid % modulus != 0 && taskid % (modulus/2) == 0) {
            parent[s] = taskid-modulus/2; // I receive the right child produced in this stage, which I will split in the next stage

            // receive size of matrix to receive
            MPI_Recv(&n, 1, MPI_INT, parent[s], 1, MPI_COMM_WORLD, &status);

            // receive matrix
            assert(D == NULL && E == NULL);
            D = malloc(n * sizeof(double));
            E = malloc((n-1) * sizeof(double));
            MPI_Recv(D, n, MPI_DOUBLE, parent[s], 2, MPI_COMM_WORLD, &status);
            MPI_Recv(E, n-1, MPI_DOUBLE, parent[s], 3, MPI_COMM_WORLD, &status);
        }

        modulus /= 2;
    }

    /*
     * Some final remarks of the divide phase:
     * The size of the current leave is in nl.
     * The actual allocated memory is still stored in n (which will be needed in conquer phase)
     */
    MPI_Barrier(MPI_COMM_WORLD);

    /**********************
     * Compute eigenpairs of leaves using QR algorithm
     **********************/

    if (taskid == MASTER)
        printf("Apply QR algorithm on leaves ...\n");
    // TODO. make depth of tree big enough to assure that dense matrix Q of leaves can be stored (thus probably split T even on nodes itself)

    // get current leaf node in tree
    currNode = &(evTree.t[treeDepth-1].s[taskid]);
    assert(currNode->n == nl);

    // assign D to leaf node
    if (n > nl) {
        /* since we have to store all D's anyway, we don't wanna store an array that is bigger than
         * necessary. We could have shrinked D in the splitting step before, but
         * that would have probably caused even more copy operations
         */
        currNode->D = malloc(nl*nl * sizeof(double));
        memcpy(currNode->D, D, nl*sizeof(double));
        myfree(&D);
    } else {
        assert(nl == n);
        currNode->D = D;
        D = NULL;
    }

    // orthonormal where the columns are eigenvectors
    currNode->Q = malloc(nl*nl * sizeof(double));

    int ret =  LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', nl, currNode->D, E, currNode->Q, nl);
    assert(ret == 0);

    // reset D to L
    currNode->L = currNode->D;
    double* L = currNode->L; // the reason why I store extra, is because this is easier then find later on the right child, which sends it to the parent node
    currNode->D = NULL;

    // off-diagonal elements are not needed anymore
    myfree(&E);
    assert(E == NULL);

    // upper stage in the tree only needs first and last line (as explained later)
    double* Q1f = currNode->Q; // first row
    double* Q1l = currNode->Q + (nl-1)*nl; // last row

    // if there was not a single split, then we only need the eigenvectors stored in Q
    if (numSplitStages == 0)
        goto EndOfAlgorithm;

    /**********************
     * Conquer phase
     **********************/
    if (taskid == MASTER)
        printf("Start Conquer Phase ...\n");

    // sizes of Q matrices
    int nq1 = nl, nq2;
    double *Q2f = NULL, *Q2l = NULL;

    int performedMerge = 0; // If I haven't performed a merge yet, then I am not allowed to free Q1l,Q1f, because they still point to the leaf node

    // Stage s=numSplitStages-1: stage where leaves are merged (since s=0 is first split stage)
    assert(numSplitStages > 0);
    for (s = numSplitStages-1; s >= 0; s--) {
        currNode = currNode->parent;
        assert(currNode != NULL);
        // the tree is not completely balanced, so there might be tasks, that haven't performed a split at the bottom stage s

        // if task should not compute the spectral decomposition of two leaves
        if (currNode->taskid != taskid && currNode->right->taskid == taskid) {
            //printf("Task %d: Send info to %d in stage %d\n", taskid, currNode->taskid, s);
            // send eigenvalues and necessary part of eigenvectors to parent node in tree
            MPI_Ssend(&nq1, 1, MPI_INT, currNode->taskid, 4, MPI_COMM_WORLD);
            MPI_Ssend(L, nq1, MPI_DOUBLE, currNode->taskid, 5, MPI_COMM_WORLD);
            MPI_Ssend(Q1f, nq1, MPI_DOUBLE, currNode->taskid, 6, MPI_COMM_WORLD);
            MPI_Ssend(Q1l, nq1, MPI_DOUBLE, currNode->taskid, 7, MPI_COMM_WORLD);

            // this task can't be the master, so there is no work left to do for it
            if (performedMerge) { // note, that the leaf nodes don't copy elements into Q1l,Q1f
                myfree(&Q1f);
                myfree(&Q1l);
            } else {
                Q1f = NULL;
                Q1l = NULL;
            }
        }

        // if task combines two splits in this stage
        // Note: if right == left, then the tree was not splitted in this stage (single path)
        if (currNode->right != currNode->left) {
            // for all tasks, that are leaves of the current node, they can work in parallel on the root finding problem
            if (taskid >= currNode->taskid && taskid < (currNode->taskid+currNode->numLeaves)) {
                if (currNode->taskid == taskid) {
                    performedMerge = 1;
                    // get current node in tree
                    EVRepNode* leftChild = currNode->left;
                    assert(leftChild != NULL && leftChild->n == nq1 && leftChild->taskid == taskid);

                    // receive size of matrix to receive
                    MPI_Recv(&nq2, 1, MPI_INT, currNode->right->taskid, 4, MPI_COMM_WORLD, &status);
                    assert(currNode->n == nq1+nq2);
                    currNode->D = malloc(currNode->n * sizeof(double));
                    memcpy(currNode->D, leftChild->L, currNode->n*sizeof(double));

                    // receive eigenvalues and necessary part of eigenvectors from right child in tree
                    MPI_Recv(currNode->D+nq1, nq2, MPI_DOUBLE, currNode->right->taskid, 5, MPI_COMM_WORLD, &status);
                    Q2f = malloc(nq2 * sizeof(double));
                    Q2l = malloc(nq2 * sizeof(double));
                    MPI_Recv(Q2f, nq2, MPI_DOUBLE, currNode->right->taskid, 6, MPI_COMM_WORLD, &status);
                    MPI_Recv(Q2l, nq2, MPI_DOUBLE, currNode->right->taskid, 7, MPI_COMM_WORLD, &status);
                    //printf("Task %d: Conquer from (Task %d: %d; Task %d: %d)\n", taskid, taskid, nq1, currNode->right->taskid, nq2);

                    /*
                     * Compute z, where z is
                     *
                     * z = | Q1^T   0 | | e_k            |
                     *     | 0   Q2^T | | theta^-1 * e_1 |
                     *
                     * Note, I only need the last row of Q1 and the first row of Q2 in order to compute z
                     */
                    currNode->z = computeZ(Q1l, Q2f, nq1, nq2, currNode->theta);
                }

                // compute root finding in parrallel
                // compute eigenvalues lambda_1 of rank-one update: D + beta*theta* z*z^T
                // Note, we may not overwrite the diagonal elements in D with the new eigenvalues, since we need those diagonal elements to compute the eigenvectors
                rtic = omp_get_wtime();
                computeEigenvalues(currNode, mpiHandle);
                rtoc = omp_get_wtime();
                rsum += rtoc - rtic;

                if (currNode->taskid == taskid) {
                    L = currNode->L;
                    // compute normalization factors
                    computeNormalizationFactors(currNode);
//                    printVector(currNode->L, currNode->n);
//                    printVector(currNode->N, currNode->n);
//                    printVector(currNode->D, currNode->n);
//                    printVector(currNode->z, currNode->n);

                    /*
                     * It holds that T = W L W^T, where W = QU
                     * We only have to compute the first and last row of W and send it to the parent
                     *
                     * left child: the parent needs the last row of W (which is Q1 in parent) to compute z.
                     * To compute the last row of W we only need the last row of Q (last row of Q2)
                     *
                     * right child: the parent needs the first row of W (which is Q2 in parent) to compute z.
                     * To compute the first row of W we only need the first row of Q (first row of Q1)
                     *
                     * But, the parent has (if s > 1) to compute the last and first row of its W again, so it needs
                     * also the first row of its Q1 from its left child resp. the last row of its Q2 from the right child
                     */
                    if (s == 0) { // if we already reached root of tree
                        assert(taskid == MASTER);
                        // write eigenvalues into file
                        if (s < numSplitStages-1) { // note, that the leaf nodes don't copy elements into Q1l,Q1f
                            myfree(&Q1f);
                            myfree(&Q1l);
                        } else {
                            Q1f = NULL;
                            Q1l = NULL;
                        }
                        myfree(&Q2f);
                        myfree(&Q2l);

                        goto EndOfAlgorithm;
                    }

                    // compute first and last row of W
                    double* Wf = malloc((nq1+nq2) * sizeof(double)); // first line of W
                    double* Wl = malloc((nq1+nq2) * sizeof(double)); // last line of W

                    #pragma omp parallel private(i,j) // parallel region to ensure, that each thread has another array allocated for the eigenvector
                    {
                        // store i-th eigenvector of U
                        double* ev = malloc(currNode->n * sizeof(double));

                        #pragma omp for
                        for (i = 0; i < nq1+nq2; ++i) {
                            // get i-th eigenvector of U
                            getEigenVector(currNode, ev, i);

                            Wf[i] = 0;
                            for (j = 0; j < nq1; ++j)
                                Wf[i] += Q1f[j] * ev[j];
                            Wl[i] = 0;
                            for (j = 0; j < nq2; ++j)
                                Wl[i] += Q2l[j] * ev[nq1+j];
                        }

                        free(ev);
                    }
        //            if (s==0) {
        //                printVector(Wf, nq1+nq2);
        //                printVector(Wl, nq1+nq2);
        //                goto EndOfAlgorithm;
        //            }
                    if (s < numSplitStages-1) { // note, that the leaf nodes don't copy elements into Q1l,Q1f
                        myfree(&Q1f);
                        myfree(&Q1l);
                    } else {
                        Q1f = NULL;
                        Q1l = NULL;
                    }
                    myfree(&Q2f);
                    myfree(&Q2l);

                    // update variables for next iteration
                    nq1 = nq1 + nq2;
                    Q1f = Wf;
                    Q1l = Wl;
                    Wf = NULL;
                    Wl = NULL;
                }
            }
        }
    }

    /**********************
     * End of algorithm
     **********************/

    EndOfAlgorithm:

    toc = omp_get_wtime();
    if (taskid == MASTER) {
        double elapsedTime = toc-tic;
        printf("\n");
        printf("Required time to compute all eigenvalues: %f seconds\n", elapsedTime);
        printf("Required time for root finding: %f seconds; fraction: %.1f%%\n", rsum, 100*rsum/elapsedTime);
    }

    if (writeOutput) {
        if (taskid == MASTER) {
            printf("\n");
            printf("Write results to file ...\n");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        writeResults(outputfile,OD,OE,&evTree, mpiHandle, computeEV, evFile);
    }

    freeEVRepTree(&evTree);

    MPI_Barrier(MPI_COMM_WORLD);
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
    printf("    1 - Matrix will have the tridiagonal form [-1,d_i,-1] where the diagonal\n");
    printf("        elements will be evenly spaced in the interval [1,100] \n");
    printf("    2 - Eigenvalue i has the form: 2 + 2*cos((PI*i)/(n+1)) \n");
    printf("        Poisson-matrix (tridiagonal form of [-1,2-1])\n");
    printf("    If option i is used, then this option will be ignored.\n");
    printf(" -n NUM\n");
    printf("    Specify the dimension of the matrix chosen with option -s.\n");
    printf(" -e(FILENAME)\n");
    printf("    Without this option, no eigenvectors are computed, just the eigenvalues.\n");
    printf("    If you just specify the flag -e, then all eigenvectors will be computed.\n");
    printf("    If you specify additionally a filename, then he will read the indices\n");
    printf("    of the eigenvectors to compute from this file (each line one index).\n");
    printf("    Note, there is no blank between the option and the filename.\n");
    printf("\n");
}
