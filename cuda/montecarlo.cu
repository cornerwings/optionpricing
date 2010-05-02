#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "cublas.h"

//#include <cutil_inline.h>

#include <culapack.h>
#include <culapackdevice.h>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))

__device__ inline float MoroInvCNDgpu(float P){
    const float a1 = 2.50662823884f;
    const float a2 = -18.61500062529f;
    const float a3 = 41.39119773534f;
    const float a4 = -25.44106049637f;
    const float b1 = -8.4735109309f;
    const float b2 = 23.08336743743f;
    const float b3 = -21.06224101826f;
    const float b4 = 3.13082909833f;
    const float c1 = 0.337475482272615f;
    const float c2 = 0.976169019091719f;
    const float c3 = 0.160797971491821f;
    const float c4 = 2.76438810333863E-02f;
    const float c5 = 3.8405729373609E-03f;
    const float c6 = 3.951896511919E-04f;
    const float c7 = 3.21767881768E-05f;
    const float c8 = 2.888167364E-07f;
    const float c9 = 3.960315187E-07f;
    float y, z;

    if(P <= 0 || P >= 1.0f)
        return __int_as_float(0x7FFFFFFF);

    y = P - 0.5f;
    if(fabsf(y) < 0.42f){
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }else{
        if(y > 0)
            z = __logf(-__logf(1.0f - P));
        else
            z = __logf(-__logf(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if(y < 0) z = -z;
    }

    return z;
}

__global__ void NormalDistribution(float *A, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        A[i] = MoroInvCNDgpu(A[i]);
}

__global__ void CumSum(float *A, int N)
{
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    float *B = A + (offset)*N;

    float cumsum = 0;
    for(int i=0; i<N; i++){
        cumsum += B[i];
        B[i] = cumsum;
    }  
}

__global__ void SimulateAssets(float *A, int N, float sqrtofdt, float sig, float rminus, float S0)
{
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    float *B = A + (offset)*N;

    float cumsum = 0;
    for(int i=0; i<N; i++){
        cumsum += ( rminus + sig * sqrtofdt * MoroInvCNDgpu(B[i]) );
        B[i] = S0*__expf(cumsum);
    }  
}

__global__ void TransposeMatrix(float* A_o, float* A_i, int M, int N)
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xIndex < M && yIndex < N)
   {
       unsigned int index_in  = xIndex + M * yIndex;
       unsigned int index_out = yIndex + N * xIndex;
       A_o[index_out] = A_i[index_in]; 
   }
}

__global__ void InitializeMatrix(float *A, int N, float value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N)
        A[i] = value;
}

__global__ void PayOffColumn(float *P, float *S, float K, int Col, int ColLength)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<ColLength)
        P[Col*ColLength + i] = fmax(0, S[Col*ColLength + i]-K);
}  

__global__ void PayOff(float *y, float *S, float K, int Col, int ColLength)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<ColLength)
        y[i] = fmax(0, S[Col*ColLength + i]-K);
}

__global__ void FillYexY(float *yex, float *X, float *Y, float *y, float *P, float *S, float disc, int Col, int ColLength, int RowLength)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(y[i]>0){
        yex[i] = y[i];
        X[i] = S[Col*ColLength + i];

        float ndisc = 0.0;                

        int it;
        for(it=0; it<RowLength-Col; it++)
            ndisc += __powf(disc, it) * P[ (Col+1+it)*ColLength + i ];

        Y[i] = ndisc;

    }
    else{
        yex[i] = 0;
        X[i] = 0;
        Y[i] = 0;
    }
    
} 

__global__ void GenerateA(float *A, float *X, int nzcount)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<3*nzcount){
        int expi = i/nzcount;

        A[i] = __powf(X[i-expi*nzcount], expi);
    }

}

__global__ void PackNum(float *NUM, float *W, int nzcount)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    NUM[i*nzcount + i] = W[i];
}

__global__ void UpdatePayOff(float *P, float *yex, float *yco, float *y, int Col, int n, int N)
{
    int j=0;
    for(int i=0; i<n; i++){
        if(y[i]>0){
            if(yex[j] > yco[j]){
                for(int it=0; it<N; it++)
                    P[it*n + i] = 0;
                P[Col*n + i] = yex[j];
            }
            j++;
        }
    }
}

__global__ void AccumDiscount(float *A, float disc, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N)
        A[i] = __powf(disc, i);    
}
  

void UniformDistribution(float *A, int M, int N)
{
	int i,j;
	for(i=0; i<N; i++){
		for(j=0; j<M; j++){
            float randnum =  (float) rand() / RAND_MAX ;
	        if(randnum >= 1.0f || randnum <= 0.0f)
		        randnum = 0.5f;
			A[i*M+j] = randnum;
		}
	}
}

void OutputMatrix(char *fileName, float *A, int M, int N){
    FILE *fp;
    fp = fopen(fileName, "w");

    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++){
            fprintf(fp, "%f ", A[i+j*M]);
        }
        fprintf(fp, "\n");
    } 

    fclose(fp);
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

#define PATHS 20000
#define TGRID 16


float MonteCarloGPU(float S0, float K, float T, float sig, float R)
{

    int n = PATHS;
    int N = TGRID;    

    float dt = T/N;
    float sqrtofdt = sqrt(dt);
    float rminus = ( R - 0.5*pow(sig,2) ) * dt;    

    int length = n*(N-1);
    size_t size = length*sizeof(float);

    cudaError_t cudaError;

    float *dW_h = (float*) malloc( size );
    UniformDistribution(dW_h, N-1, n); 
    //OutputMatrix("dWgpu.mat", dW_h, N-1, n);

    float *Sasset;
    cudaError = cudaMalloc( (void**) &Sasset, size);
    if(cudaError != cudaSuccess){
	    printf("Memory error\n");
	    exit(-1);
    }
    cudaMemcpy(Sasset, dW_h, size, cudaMemcpyHostToDevice); 
    free(dW_h);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    SimulateAssets<<<blocksPerGrid, threadsPerBlock>>>(Sasset, N-1, sqrtofdt, sig, rminus, S0);

    float *Stemp;
    cudaError = cudaMalloc( (void**) &Stemp, size);   
    if(cudaError != cudaSuccess){
	    printf("Memory error\n");
	    exit(-1);
    } 
    cudaMemcpy(Stemp, Sasset, size, cudaMemcpyDeviceToDevice); 

    int BLOCK_DIM = 16;

    dim3 grid(N-1 / BLOCK_DIM +1 , n / BLOCK_DIM +1, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    TransposeMatrix<<< grid, threads >>>(Sasset, Stemp, N-1, n);   

    cudaFree(Stemp);    

    float *S;
    cudaError = cudaMalloc( (void**) &S, n*N*sizeof(float) );
    if(cudaError != cudaSuccess){
	    printf("Memory error\n");
	    exit(-1);
    }

    InitializeMatrix<<<blocksPerGrid, threadsPerBlock>>>(S, n, S0);
    if(cudaError != cudaSuccess){
	    printf("Memory error\n");
	    exit(-1);
    }
    cudaMemcpy(S+n, Sasset, size, cudaMemcpyDeviceToDevice);
    cudaFree(Sasset);

    float disc = exp(-R*dt);

    float *P;
    cudaError = cudaMalloc( (void**) &P, n*N*sizeof(float) );
    if(cudaError != cudaSuccess){
	    printf("Memory error\n");
	    exit(-1);
    }
    cudaMemset(P, 0, n*N*sizeof(float) );
    
    PayOffColumn<<<blocksPerGrid, threadsPerBlock>>>(P,S,K,N-1,n);

    for(int nn=N-2; nn>0; nn--){
        float *y;
        cudaMalloc( (void**) &y, n*sizeof(float) );

        PayOff<<<blocksPerGrid, threadsPerBlock>>>(y,S,K,nn,n);

        float *yex;
        float *X;
        float *Y;

        cudaError = cudaMalloc( (void**) &yex, n*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error\n");
            exit(-1);
        } 
        cudaError = cudaMalloc( (void**) &X, n*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error\n");
            exit(-1);
        } 
        cudaError = cudaMalloc( (void**) &Y, n*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error\n");
            exit(-1);
        } 
        
        FillYexY<<<blocksPerGrid, threadsPerBlock>>>(yex,X,Y,y,P,S,disc,nn,n,N-1);

        float *yex_h = (float*) malloc(n*sizeof(float));
        float *X_h = (float*) malloc(n*sizeof(float));
        float *Y_h = (float*) malloc(n*sizeof(float));

        if(yex_h == NULL || X_h == NULL || Y_h == NULL){
            printf("No CPU\n");
            exit(1);
        }

        cudaMemcpy(yex_h, yex, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(X_h, X, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Y_h, Y, n*sizeof(float), cudaMemcpyDeviceToHost);

        int nzcount = 0;
        for(int i=0; i<n; i++){
            if(yex_h[i] > 0){
                yex_h[nzcount] = yex_h[i];
                X_h[nzcount] = X_h[i];
                Y_h[nzcount] = Y_h[i];
                nzcount++;
            }
        } 

        cudaMemcpy(yex, yex_h, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(X, X_h, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Y, Y_h, n*sizeof(float), cudaMemcpyHostToDevice);

        float *A, *Aco;

        cudaError = cudaMalloc( (void**)&A, nzcount*3*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error\n");
            exit(-1);
        } 
        cudaError = cudaMalloc( (void**)&Aco, nzcount*3*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error\n");
            exit(-1);
        } 

        int ablocksPerGrid = (nzcount*3 + threadsPerBlock - 1) / threadsPerBlock;

        GenerateA<<<ablocksPerGrid, threadsPerBlock>>>(A,X,nzcount);

        cudaMemcpy( Aco, A, nzcount*3*sizeof(float), cudaMemcpyDeviceToDevice);

	culaDeviceSgels('N', nzcount, 3, 1, A, nzcount, Y, nzcount);

/*	
	//Optional calculation for Least Squares regression using SVD
	//Not suitable for GPU - as the memory available is usually less

        float *U;
        float *VT;
        float *W;

        cudaError = cudaMalloc( (void**) &U, nzcount * nzcount * sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error U %d\n", nzcount);
            exit(-1);
        } 

        cudaError = cudaMalloc( (void**) &VT, 3 * 3 * sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error VT\n");
            exit(-1);
        } 

        cudaError = cudaMalloc( (void**) &W, imin(nzcount,3) * sizeof(float) ); 
        if(cudaError != cudaSuccess){
            printf("Memory error W\n");
            exit(-1);
        } 
       

        culaDeviceSgesvd('A', 'A', nzcount, 3, A, nzcount, W, U, nzcount, VT, 3);   

        float *NUM;
        cudaError = cudaMalloc( (void**) &NUM, nzcount * 3 * sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error NUM\n");
            exit(-1);
        } 
        cudaMemset(NUM, 0, nzcount*3*sizeof(float) );

        PackNum<<<1,imin(nzcount,3)>>>(NUM, W, nzcount);
        
        float *DEN;
        cudaError = cudaMalloc( (void**) &DEN, nzcount * sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error DEN\n");
            exit(-1);
        } 

	cublasSgemm('T', 'N', nzcount, 1, nzcount, 1, U, nzcount, Y, n, 0, DEN, nzcount);

        culaDeviceSgels('N', nzcount, 3, 1, NUM, nzcount, DEN, nzcount); 

        float *b;
        cudaError = cudaMalloc( (void**)&b, 3*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error b\n");
            exit(-1);
        } 
        
	cublasSgemm('T', 'N', 3, 1, 3, 1, VT, 3, DEN, 3, 0, b, 3); 
*/

        float *yco; 
        cudaError = cudaMalloc( (void**)&yco, nzcount*sizeof(float) );
        if(cudaError != cudaSuccess){
            printf("Memory error yco\n");
            exit(-1);
        } 

        cublasSgemm('N', 'N', nzcount, 1, 3, 1, Aco,  nzcount, Y, 3, 0, yco, nzcount); 
                       
        UpdatePayOff<<<1,1>>>(P, yex, yco, y, nn, n, N);

        //cudaFree(yco);
        //cudaFree(b);
        //cudaFree(DEN);
        //cudaFree(NUM);
        //cudaFree(W);
        //cudaFree(U);
        //cudaFree(VT);
        cudaFree(y);
        cudaFree(yex);
        cudaFree(Y);
        cudaFree(X);
        cudaFree(A);
        cudaFree(Aco);

        free(yex_h);
        free(Y_h);
        free(X_h);

    }

    float price = 0.0;

    float *cumDisc;
    cudaMalloc( (void**)&cumDisc, N*sizeof(float) );

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    AccumDiscount<<<blocksPerGrid, threadsPerBlock>>>(cumDisc, disc, N);

    float *result;
    cudaMalloc( (void**)&result, n*sizeof(float) );

    cublasSgemm('N', 'N', n, 1, N, 1, P, n, cumDisc, N, 0, result, n);

    float *result_h = (float*) malloc(n*sizeof(float));

    cudaMemcpy(result_h, result, n*sizeof(float), cudaMemcpyDeviceToHost );

    for(int i=0; i<n; i++)
        price += result_h[i];
    
    price /= n;

    cudaFree(cumDisc);
    cudaFree(result);
    cudaFree(S);
    cudaFree(P);

    free(result_h);

    return price;

}

void print_mat(float *A, int M, int N){
    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++)
            printf("%f ", A[i+j*M]);
        printf("\n");
    } 
}

int main(){

    culaInitialize();
    cublasInit();

    //unsigned int hTimer;
    //cutCreateTimer(&hTimer);

    MonteCarloGPU(100,100,1,0.2,0.05);

    //cutResetTimer(hTimer);
    //cutStartTimer(hTimer);

    for(int i=0; i<5; i++){
    	float price = MonteCarloGPU(100,100,1,0.2,0.05);
    	printf("%d - %f\n", i, price);
    }

    //cutStopTimer(hTimer);
    //float timeelapsed = cutGetTimerValue(hTimer);

    //printf("time - %f\n", timeelapsed);

    cublasShutdown();
    culaShutdown();

/*
    srand(time(0));

    int n = 1000;
    int N = 16;

    int length = n*(N-1);
    size_t size = length*sizeof(float);

    float *dW = (float*) malloc( size );
    UniformDistribution(dW, n, N-1);

    float *dW_d;
    cudaMalloc( (void**) &dW_d, size);
    cudaMemcpy(dW_d, dW, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n*N + threadsPerBlock - 1) / threadsPerBlock;
    
    NormalDistribution<<<blocksPerGrid, threadsPerBlock>>>(dW_d, length); 
    
    cudaMemcpy(dW, dW_d, size, cudaMemcpyDeviceToHost);
    cudaFree(dW_d);

    OutputMatrix("dW_gpu.mat", dW, n, N-1); 

    free(dW);

    dW = (float*) malloc(size);
    
    for(int i=0; i<n; i++){
        for(int j=0; j<N-1; j++){
            dW[i*(N-1)+j] = j;
        }
    }

    OutputMatrix("init_scan.mat", dW, N-1, n);

    cudaMalloc( (void**) &dW_d, size);
    cudaMemcpy(dW_d, dW, size, cudaMemcpyHostToDevice);
    
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    CumSum<<<blocksPerGrid, threadsPerBlock>>>(dW_d, N-1);     
    
    cudaMemcpy(dW, dW_d, size, cudaMemcpyDeviceToHost);
    cudaFree(dW_d);

    OutputMatrix("scan_gpu.mat", dW, N-1, n); 


    for(int i=0; i<N-1; i++){
        for(int j=0; j<n; j++){
            dW[i*n+j] = j;
        }
    }

    float *dW_t = (float*) malloc(n*(N-1)*sizeof(float));

    float *dW_td;
    cudaMalloc( (void**) &dW_td, size);

    cudaMalloc( (void**) &dW_d, size);
    cudaMemcpy(dW_d, dW, size, cudaMemcpyHostToDevice);

    int BLOCK_DIM = 16;

    dim3 grid(n / BLOCK_DIM +1 , N-1 / BLOCK_DIM +1, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    TransposeMatrix<<< grid, threads >>>(dW_td, dW_d, n, N-1);    

    cudaMemcpy(dW_t, dW_td, size, cudaMemcpyDeviceToHost);
    cudaFree(dW_d); 
    cudaFree(dW_td);   

    OutputMatrix("scan_trans.mat", dW_t, N-1, n);

    free(dW_t);

    free(dW);
*/
    return 0;         

}

