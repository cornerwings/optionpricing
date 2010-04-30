#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

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

/*
__global__ void
matrixMul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];


        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void transpose_naive(float *odata, float* idata, int width, int height)
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xIndex < width && yIndex < height)
   {
       unsigned int index_in  = xIndex + width * yIndex;
       unsigned int index_out = yIndex + height * xIndex;
       odata[index_out] = idata[index_in]; 
   }
}
*/

__global__ void NormalDistribution(float *data, int length)
{
    unsigned int i = threadIdx.x;
    //data[i] = MoroInvCNDgpu( (float) (i+1) / length );
    data[i] = 1.0;
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

/*
#define PATHS 1000
#define TIMES 16

#define R 0.05

int MonteCarloGPU(float S0, float K, float T, float sig){

    culaStatus s;
    s = culaInitialize();
    if(s != culaNoError)
    {
	    printf("%s\n", culaGetStatusString(s));
    }



    srand(time(0));
    culaShutdown();

}
*/

#define THREADS_N 256

void write_mat(char *name, float *A, int M, int N){
    FILE *fp;
    fp = fopen(name, "w");

    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++){
            fprintf(fp, "%f ", A[i+j*M]);
        }
        fprintf(fp, "\n");
    } 

    fclose(fp);
}

int main(){
	
    culaStatus s;
	s = culaInitialize();
	if(s != culaNoError)
	{
		printf("%s\n", culaGetStatusString(s));
	}

    srand(time(0));

    int i,j,k;

    int n = 1000;
    int N = 16;

    //Option data
    float S0 = 100;
    float K = 10;
    float T = 1;
    float sig = 0.2;
    float R = 0.05;

    float dt = T/N;
    float sqrtofdt = sqrt(dt);
    float rminus = ( R - 0.5*pow(sig,2) )	* dt;

    int size = n*N;

    float *dW = (float*) malloc( size*sizeof(float) );
    memset(dW, 0, size*sizeof(float));

    //float *dW_d;
    //cutilSafeCall(cudaMalloc( (void**) &dW_d, size*sizeof(float)) );
    //cutilSafeCall(cudaMemcpy(dW_d, dW, size*sizeof(float), cudaMemcpyHostToDevice) );
    //cutilSafeCall(NormalDistribution<<<1, size>>> (dW_d, size)); 
    //cutilSafeCall(cudaMemcpy(dW, dW_d, size*sizeof(float), cudaMemcpyDeviceToHost));
    //cutilSafeCall(cudaFree(dW_d));
    write_mat("dW_gpu.mat", dW, n, N); 

/*
    float *Sasset = (float*) malloc( (N-1)*n*sizeof(float));

    for(i=0; i<n; i++){
        float cumsum = 0;
        for(j=0; j<N-1; j++){
            cumsum += (rminus + sig * sqrtofdt * MoroInvCND( (float) rand() / RAND_MAX ));
            Sasset[j+i*(N-1)] = S0*exp(cumsum);
        }
    }

    //Adding intial prices column to S
    trans_mat(Sasset, N-1, n);

    float *S = (float*) malloc(n*N*sizeof(float));
    for(i=0; i<n; i++)
        S[i] = S0;
    memcpy(S+n, Sasset, (N-1)*n*sizeof(float));

    free(Sasset);

    float disc = exp(-R*dt);

    //Initialize payoff matrix
    float *P = (float*) malloc(n*N*sizeof(float));
    memset(P,0,n*N*sizeof(float));

    for(i=0; i<n; i++)
        P[(N-1)*n + i] = MAX(0, S[(N-1)*n + i]-K);

    int nn;
    for(nn=N-2; nn>0; nn--){

        float *y = (float*) malloc(n*sizeof(float));

        for(i=0; i<n; i++)
            y[i] = MAX(0, S[nn*n +i]-K);

        float *yex = (float*) malloc(n*sizeof(float));
        float *X = (float*) malloc(n*sizeof(float));
        float *Y = (float*) malloc(n*sizeof(float));

        int nzcount = 0;

        for(i=0; i<n; i++){
            if(y[i]>0){
                yex[nzcount] = y[i];
                X[nzcount] = S[nn*n + i];

                float ndisc = 0.0;                

                int it;
                for(it=0; it<N-nn-1; it++)
                    ndisc += pow(disc, it) * P[ (nn+1+it)*n + i ];

                Y[nzcount] = ndisc;

                nzcount++;
            }
        }

        yex = (float*) realloc(yex, nzcount*sizeof(float));
        X = (float*) realloc(X, nzcount*sizeof(float));
        Y = (float*) realloc(Y, nzcount*sizeof(float));

        float *A = (float*) malloc (nzcount*3*sizeof(float));
        for(i=0; i<nzcount*3; i++){
            int exp = i/nzcount;
            A[i] = pow(X[i-exp*nzcount], exp);
        }

        float *Acopy = (float*) malloc (nzcount*3*sizeof(float));
        memcpy(Acopy, A, nzcount*3*sizeof(float));

        float* W = (float*) malloc( imin(nzcount,3) * sizeof(float));
        float* U = (float*) malloc( nzcount * nzcount * sizeof(float));
        float* VT = (float*) malloc( 3 * 3 * sizeof(float)); 

        //Take SVD
        s = culaSgesvd('A', 'A', nzcount, 3, A, nzcount, W, U, nzcount, VT, 3);
	    checkStatus(s);     

        //Pack S into a square matrix
        float *NUM = (float*) malloc(nzcount*3*sizeof(float));
        memset(NUM,0,nzcount*3*sizeof(float));
        
        for(i=0; i<imin(nzcount,3); i++)
            NUM[i*nzcount+i] = W[i];

        //Take transpose of U
        trans_mat(U, nzcount, nzcount);

        float *DEN = (float*) malloc(nzcount*sizeof(float));
        matrixMul(DEN, U, Y, nzcount, nzcount, 1);

        //Do Least Squares
        s = culaSgels('N', nzcount, 3, 1, NUM, nzcount, DEN, nzcount);
	    checkStatus(s);

        DEN = (float*) realloc(DEN, 3*sizeof(float)); 

        float *b = (float*) malloc(3* sizeof(float));
        trans_mat(VT, 3, 3);
        matrixMul(b, VT, DEN, 3, 3, 1);

        float *yco = (float*) malloc(nzcount*sizeof(float));
        matrixMul(yco, Acopy, b, nzcount, 3, 1);

        j=0;
        for(i=0; i<n; i++){
            if(y[i]>0){
                if(yex[j] > yco[j]){
                    int it;
                    for(it=0; it<N; it++)
                        P[it*n + i] = 0;
                    P[nn*n + i] = yex[j];
                }
                j++;
            }
        }

        free(y);
        free(yex);
        free(X);
        free(Y);
        free(A);
        free(U);
        free(W);
        free(VT);
        free(b);
        free(yco);

    } 

    float price = 0.0;

    float *acdis = (float*) malloc(N*sizeof(float));
    for(i=0; i<N; i++){
        acdis[i] = pow(disc, i);
    }

    float *accum = (float*) malloc(n*sizeof(float));
    matrixMul(accum, P, acdis, n, N, 1);
    
    for(i=0; i<n; i++)
        price += accum[i];

    price /= n;

    printf("Price evaluated: %f\n", price);   

    free(S);
    free(P);
*/

    culaShutdown();
}

