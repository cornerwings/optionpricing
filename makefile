CC=g++
NVCC=nvcc
CFLAGS=-O0
INCLUDES=-I${CULA_INC_PATH} -I${CUDA_INC_PATH}
LIBPATH32=-L${CULA_LIB_PATH_32}
LIBPATH64=-L${CULA_LIB_PATH_64}

LIBS=-lm
LAPACKLIBS=-llapack_atlas -llapack -lblas -latlas
CULALIBS=-lcula -lcublas -lcudart

all: subdirs lapacktest amopt

subdirs: 
	$(MAKE) --directory=cuda

amopt:
	${CC} -o amopt amopt.cc $(CFLAGS) $(LIBS)

culatest:
	${NVCC} -o culatest culatest.cc $(CFLAGS) $(INCLUDES) $(LIBPATH32) $(CULALIBS) $(LIBS)

lapacktest:
	${CC} -o lapacktest lapacktest.cc $(CFLAGS) $(LAPACKLIBS) $(LIBS)

clean:
	rm -f culatest amopt blastest lapacktest
	-$(MAKE) clean --directory=cuda

