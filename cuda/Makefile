CC=g++
NVCC=nvcc
CFLAGS=-O0
INCLUDES=-I${CULA_INC_PATH} -I${CUDA_INC_PATH}
LIBPATH32=-L${CULA_LIB_PATH_32}
LIBPATH64=-L${CULA_LIB_PATH_64}

LIBS=-lm
LAPACKLIBS=-llapack_atlas -llapack -lblas -latlas
CULALIBS=-lcula -lcublas -lcudart

all: montecarlo

montecarlo:
	${NVCC} -o montecarlo montecarlo.cc $(CFLAGS) $(INCLUDES) $(LIBPATH32) $(CULALIBS) $(LIBS)

clean:
	rm -f montecarlo

