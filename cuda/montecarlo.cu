#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include <culapack.h>
#include <culapackdevice.h>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))

void print_mat(float *A, int M, int N){
    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++)
            printf("%f ", A[i+j*M]);
        printf("\n");
    } 
}

void trans_mat(float *A, int M, int N){
    float *temp = (float*)malloc(M*N*sizeof(float));
    memcpy(temp, A, M*N*sizeof(float));

    int i,j;
    for(i=0; i<M; i++)
        for(j=0; j<N; j++)
            A[i*N+j] = temp[j*M+i];
}


void checkStatus(culaStatus status)
{
    if(!status)
        return;

    if(status == culaArgumentError)
        printf("Invalid value for parameter %d\n", culaGetErrorInfo());
    else if(status == culaDataError)
        printf("Data error (%d)\n", culaGetErrorInfo());
    else if(status == culaBlasError)
        printf("Blas error (%d)\n", culaGetErrorInfo());
    else if(status == culaRuntimeError)
        printf("Runtime error (%d)\n", culaGetErrorInfo());
    else
        printf("%s\n", culaGetStatusString(status));

    culaShutdown();
    exit(EXIT_FAILURE);
}


int main(){
	
    culaStatus s;
	s = culaInitialize();
	if(s != culaNoError)
	{
		printf("%s\n", culaGetStatusString(s));
	}


	int M, N;
	M = 3;
    N = 4;

    int i,j;

    float data[3][4] = { {5,6,8,4}, {3,2,5,6}, {1,4,9,2} };

   	float* A = NULL;
    float* S = NULL;
    float* U = NULL;
    float* VT = NULL;

    A = (float*)malloc(M*N*sizeof(float));

    int LDA = M;
    int LDU = M;
    int LDVT = N; 

    for(i=0; i<M; i++)
        for(j=0; j<N; j++)
            A[i+j*M] = data[i][j];

    print_mat(A,M,N); 

    S = (float*)malloc(imin(M,N)*sizeof(float));
    U = (float*)malloc(LDU*M*sizeof(float));
    VT = (float*)malloc(LDVT*N*sizeof(float));

    float* Ad = NULL;
    float* Sd = NULL;
    float* Ud = NULL;
    float* VTd = NULL;

    cudaMalloc((void**)&Ad, M*N*sizeof(float));
    cudaMalloc((void**)&Sd, imin(M,N)*sizeof(float));
    cudaMalloc((void**)&Ud, LDU*M*sizeof(float));
    cudaMalloc((void**)&VTd, LDVT*N*sizeof(float));

    cudaMemcpy(Ad, A, M*N*sizeof(float), cudaMemcpyHostToDevice);

    //SVD TEST
    s = culaDeviceSgesvd('A', 'A', M, N, Ad, LDA, Sd, Ud, LDU, VTd, LDVT);
    checkStatus(s);

    cudaMemcpy(S, Sd, imin(M,N)*sizeof(float), cudaMemcpyDeviceToHost);

    printf("************\nPrinting S\n");
    for(i=0; i<imin(M,N); i++){
        printf("%f ", S[i]);
    }
    printf("\n");

    cudaFree(Ad);
    cudaFree(Sd);
    cudaFree(Ud);
    cudaFree(VTd);

    //SVD TEST
    s = culaSgesvd('A', 'A', M, N, A, LDA, S, U, LDU, VT, LDVT);
    checkStatus(s);

    printf("************\nPrinting S\n");
    for(i=0; i<imin(M,N); i++){
        printf("%f ", S[i]);
    }
    printf("\n");

/*
    printf("************\nPrinting U\n");
    print_mat(U,LDU,M);

    printf("************\nPrinting V\n");
    trans_mat(VT,LDVT,N);
    print_mat(VT,LDVT,N);
*/

/*
    float NUM[24] = {
        1.44f, -9.96f, -7.55f,  8.34f,  7.08f, -5.45f,
       -7.84f, -0.28f,  3.24f,  8.09f,  2.52f, -5.70f,
       -4.39f, -3.24f,  6.27f,  5.28f,  0.74f, -1.19f,
        4.53f,  3.83f, -6.64f,  2.06f, -2.47f,  4.70f
    };

    float DEN[12] = {
        8.58f,  8.26f,  8.48f, -5.28f,  5.72f,  8.93f,
        9.35f, -4.43f, -0.70f, -0.26f, -7.36f, -2.52f
    };

    printf("************\nPrinting Numerator\n");
    print_mat(NUM,M,N);

    printf("************\nPrinting Denominator\n");
    print_mat(DEN,M,N);

    //LEAST SQUARES TEST
    s = culaSgels('N', 6, 4, 2, NUM, 6, DEN, 6);
	checkStatus(s);

    float *RES = (float*) malloc(4*2*sizeof(float));
    trans_mat(DEN,6,2);
    memcpy(RES,DEN,4*2*sizeof(float));
    trans_mat(RES,2,4);

    printf("************\nPrinting LS Result\n");    
    print_mat(RES,4,2);   
*/ 

culaShutdown();
}

