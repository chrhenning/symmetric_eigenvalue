#OBJS specifies which files to compile as part of the project 
OBJS = $(wildcard src/*.c) $(wildcard src/*.h)

#CC specifies which compiler we're using 
CC = gcc
CC_MPI = mpiicc

#COMPILER_FLAGS specifies the additional compilation options we're using 
# -w suppresses all warnings 
COMPILER_FLAGS =  -w -O3
COMPILER_FLAGS_CC = $(COMPILER_FLAGS) -fopenmp
COMPILER_FLAGS_CC_MPI = $(COMPILER_FLAGS) -openmp

#OBJ_NAME specifies the name of our exectuable 
OBJ_NAME = cuppens

MKL_MIC_ENABLE=1
MKL =    ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm
FLAGS=  -DMKL_ILP64 -qopenmp -I${MKLROOT}/include -O3

#This is the target that compiles our executable 
all : cuppen 
	
cuppen: $(OBJs)
	$(CC_MPI) $(FLAGS) $(OBJS) -o $(OBJ_NAME) $(MKL)
	
clean:
	rm -f $(OBJ_NAME)
