**************
*** cuppen ***
**************

The program cuppen computes the eigenpairs of symmetric tridiagonal matrices
using a parallel implementation of Cuppen's algorithm.

The program is compiled using the compiler wrapper mpiicc.

To compile the program, simply run

	make
	
There are several ways to run the program. The basic command would be the following:

	mpirun -n 1 -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm -genv OMP_NUM_THREADS 8 ./cuppens
	
If you enter this command in the console, the usage hints will be shown to you with explanations of
all options and positional arguments the program offers.

You might also use the Makefile to run the program, which simplifies the command.

For example, the following command

	mpirun -n 4 -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm -genv OMP_NUM_THREADS 8 ./cuppens -s 1 -n 100 out.txt
	
can be equivalently expressed as

	make run NUMTASKS=4 SCHEME=1 DUM=100 OUT=out.txt
	
For more information, please skim the Makefile.

Note, the program has successfully finished all its work, as soon as you see the output "Program finished successfully!".
In some cases (especially if the number of tasks gets large (> 100)) you might find a "Bad Termination" message afterwards,
which occurs within the MPI_FINALIZE environment. You can simply ignore it, if "Program finished successfully!" was output before.