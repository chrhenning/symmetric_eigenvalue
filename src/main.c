#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
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

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Get_processor_name(hostname, &len);

    // we don't want to use nested parallelism, because it yields to worse results, if it is not controlled by any intelligence
    omp_set_nested(0);

    // size of tridiagonal matrix
    int n;
    // tridiagonal matrix
    double* T = NULL;

    // name of output file
    char* outputfile = NULL;

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
         * Since this program only deals with tridiagonal matrices as input matrices, we store them as a special case of
         * Intel's banded matrix scheme (https://software.intel.com/en-us/node/520871)
         * A tridiagonal matrix has one sub- and one superdiagonal. So we store it in row-major layout as an n x 3 array.
         */

        if (inputfile != NULL) { // read matrix from file
            if (readTriadiagonalMatrixFromSparseMTX(inputfile, T, &n) != 0)
                MPI_ABORT(MPI_COMM_WORLD, 2);
        } else {
            switch (usedScheme) {
            case 1:
                T = createMatrixScheme1(n);
                break;
            case 2:
                T = createMatrixScheme2(n);
                break;
            }
        }

        printf("\n");
        printf("Number of MPI tasks is: %d\n", numtasks);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("   Task %d is running on node %s, which has %d available processors.\n", taskid, hostname, omp_get_num_procs());
    MPI_Barrier(MPI_COMM_WORLD);

    // ///////////////////////////
    // Cuppen's Algorithm to obtain all eigenpairs
    // ///////////////////////////

    double tic = omp_get_wtime();

    MPI_Bcast(&n,1,MPI_INT,MASTER,MPI_COMM_WORLD);

    //free(T);


    // ///////////////////////////
    // End of algorithm
    // ///////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);
    double toc = omp_get_wtime();
    if (taskid == MASTER) {
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
