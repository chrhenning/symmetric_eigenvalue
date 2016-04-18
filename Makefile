#OBJS specifies which files to compile as part of the project 
OBJS_LIBS = $(wildcard lib/*.c) $(wildcard lib/*.h) 
OBJS = $(OBJS_LIBS) $(wildcard src/*.c) $(wildcard src/*.h)

#CC specifies which compiler we're using 
CC = gcc
CC_MPI = mpiicc

#OBJ_NAME specifies the name of our exectuable 
OBJ_NAME = cuppens

MKL_MIC_ENABLE=1
MKL =    ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm
FLAGS=  -DMKL_ILP64 -qopenmp -I${MKLROOT}/include -O3

#This is the target that compiles our executable 
all : cuppen 
	
cuppen: $(OBJS)
	$(CC_MPI) $(FLAGS) $(OBJS) -o $(OBJ_NAME) $(MKL)
	
clean:
	rm -f $(OBJ_NAME)

# this standard parameters will be overriden, if you call the Makefile and assign them as parameters
NUMTASKS=4
DIM=16
OUT=out.txt
SCHEME=1

run: 
	mpirun -n $(NUMTASKS) -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm -genv OMP_NUM_THREADS 8 ./cuppens -s $(SCHEME) -n $(DIM) $(OUT)

runo: run
	cat $(OUT)
	