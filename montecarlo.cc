#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include <culapack.h>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X): (Y))

float MoroInvCND(float P){
    const float a1 = 2.50662823884;
    const float a2 = -18.61500062529;
    const float a3 = 41.39119773534;
    const float a4 = -25.44106049637;
    const float b1 = -8.4735109309;
    const float b2 = 23.08336743743;
    const float b3 = -21.06224101826;
    const float b4 = 3.13082909833;
    const float c1 = 0.337475482272615;
    const float c2 = 0.976169019091719;
    const float c3 = 0.160797971491821;
    const float c4 = 2.76438810333863E-02;
    const float c5 = 3.8405729373609E-03;
    const float c6 = 3.951896511919E-04;
    const float c7 = 3.21767881768E-05;
    const float c8 = 2.888167364E-07;
    const float c9 = 3.960315187E-07;
    float y, z;

    if(P <= 0 || P >= 1.0){
        printf("MoroInvCND(): bad parameter\n");
    }

    y = P - 0.5;
    if(fabs(y) < 0.42){
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1);
    }else{
        if(y > 0)
            z = log(-log(1.0 - P));
        else
            z = log(-log(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if(y < 0) z = -z;
    }

    return z;
}

void
matrixMul(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    unsigned int hB = wA;
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i + k * hA];
                double b = B[k + j * hB];
                sum += (a * b);
            }
            C[i + j * hA] = (float)sum;
        }
}

void randu(float *data, int m, int n){
	int i,j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			data[i*n+j] = (float) rand() / RAND_MAX;
		}
	}
}

void randn(float *data, int m, int n){
	float *udata = (float*) malloc( (m*n+1)*sizeof(float));
	randu(udata, m*n+1, 1);

	int i,j,k;

	float pi = 4. * atan(1.);	
	float square, amp, angle;	

	k = 0;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			if(k%2 == 0){
				square = -2. * log( udata[k] );
				if(square < 0.)
					square = 0.;
				amp = sqrt(square);
				angle = 2. * pi * udata[k+1];
				data[i*n+j] = amp * sin(angle);
			}
			else{
				data[i*n+j] = amp * cos(angle);
			}
			k++;
		}
	} 

	free(udata);
}

void read_mat(float *A, int M, int N){
    FILE *fp;
    fp = fopen("dW.mat", "r");

    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++){
            float f;
            fscanf(fp, "%f", &f);
            A[i+j*M] = f;
        }
    } 

    fclose(fp);
}
    

void print_mat(float *A, int M, int N){
    int i,j;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++)
            printf("%f ", A[i+j*M]);
        printf("\n");
    } 
}

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

void trans_mat(float *A, int M, int N){
    float *temp = (float*)malloc(M*N*sizeof(float));
    memcpy(temp, A, M*N*sizeof(float));

    int i,j;
    for(i=0; i<M; i++)
        for(j=0; j<N; j++)
            A[i*N+j] = temp[j*M+i];
}


//Structure to hold option related data
typedef struct _option_data{
	float initial_price;
	float strike_price;
	float time;
	float volatility;
	float interest_rate;
}option_data;

void print(float *data, int m, int n){
	int i,j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			printf("%f ", data[i*n+j]);
		}
		printf("\n");
	}
} 

void ones(float *data, int m, int n){
	int i,j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			data[i*n+j] = 1;
		}
	}
}

int randn_ok(){

	int ok = 1;
	
	int m(10), n(11);

	float *data = (float*) malloc(m*n*sizeof(float));
	randn(data,m,n);

	double sum = 0, sumsq = 0;
	
	int i,j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			sum += data[i*n+j];
			sumsq += data[i*n+j]*data[i*n+j];
		}
	}

	float N = m*n;
	
	double avg = sum/N;
	double var = sumsq/N - avg*avg;

	ok &= fabs(avg) < 2./sqrt(N); 
	ok &= fabs(var - 1.) < 2. * sqrt( 2. / N );

	return ok;
}

void brownian_inc(float *data, int m, int n, int dt){

	randn(data, m, n);
	float sqdt = sqrt(dt);

	int i,j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			data[i*n+j] *= sqdt;
		}
	}
	
}

void cumsum(float *data, int m, int n){

	int i,j;
	for(i=0; i<m; i++){
		float sum = 0;
		for(j=0; j<n; j++){
			sum+=data[i*n+j];
			data[i*n+j] = sum;
		}
	}

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

    srand(time(0));

    int i,j,k;

    int n = 10000;
    int N = 100;

	//Option data
	float S0 = 100;
	float K = 10;
	float T = 1;
	float sig = 0.2;
	float R = 0.05;

    float dt = T/N;
    float sqrtofdt = sqrt(dt);
	float rminus = ( R - 0.5*pow(sig,2) )	* dt;

/*

    printf("rminus - %f, sqrtofdt - %f\n", rminus, sqrtofdt);

    float *dW = (float*) malloc( (N-1)*n*sizeof(float));
    for(i=0; i<n; i++){
        for(j=0; j<N-1; j++){
            //dW[j+i*(N-1)] =  sqrtofdt * MoroInvCND( (float) rand() / RAND_MAX );
        }
    }

    print_mat(dW, N-1, n); 


    float *cums = (float*) malloc( (N-1)*n*sizeof(float));

    for(i=0; i<n; i++){
        float cumsum = 0;
        for(j=0; j<N-1; j++){
            cumsum += rminus + sig*sqrtofdt * MoroInvCND( (float) rand() / RAND_MAX );
            cums[j+i*(N-1)] = S0*exp(cumsum);
        }
    }

    print_mat(cums, N-1, n);
*/

    //float *dW = (float*) malloc( (N-1)*n*sizeof(float));
    //read_mat(dW, N-1, n);
    //print_mat(dW, N-1, n);

    float *Sasset = (float*) malloc( (N-1)*n*sizeof(float));

    //Packed calculation of S = S0*exp(cumsum((r - 1/2*sig^2)*dt + sig*dW));

    //float *randdata = (float *) malloc( (N-1)*n*sizeof(float));
    //randn(randdata, N-1, n);

    for(i=0; i<n; i++){
        float cumsum = 0;
        for(j=0; j<N-1; j++){
            //cumsum += (rminus + sig * dW[j+i*(N-1)]);
            cumsum += (rminus + sig * sqrtofdt * MoroInvCND( (float) rand() / RAND_MAX ));
            //cumsum += (rminus + sig * sqrtofdt * randdata[j+i*(N-1)]);
            Sasset[j+i*(N-1)] = S0*exp(cumsum);
        }
    }

    //write_mat("Sasset.mat", Sasset,N-1,n);

    //Adding intial prices column to S
    trans_mat(Sasset, N-1, n);
    //print_mat(Sasset, n, N-1);

    float *S = (float*) malloc(n*N*sizeof(float));
    for(i=0; i<n; i++)
        S[i] = S0;
    memcpy(S+n, Sasset, (N-1)*n*sizeof(float));

    free(Sasset);
    
    //write_mat("S.mat", S,n,N);

    float disc = exp(-R*dt);

    //Initialize payoff matrix
    float *P = (float*) malloc(n*N*sizeof(float));
    memset(P,0,n*N*sizeof(float));

    for(i=0; i<n; i++)
        P[(N-1)*n + i] = MAX(0, S[(N-1)*n + i]-K);

    //print_mat(P,n,N);

    int nn;
    for(nn=N-2; nn>0; nn--){

        float *y = (float*) malloc(n*sizeof(float));

        for(i=0; i<n; i++)
            y[i] = MAX(0, S[nn*n +i]-K);

        //print_mat(y,n,1);

        //fill yex,X,Y with only nonzeros from y
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

                //printf("ndisc: %f\n", ndisc);

                nzcount++;
            }
        }

        //printf("NZcount: %d\n", nzcount);

        //print_mat(yex,n,1);

        yex = (float*) realloc(yex, nzcount*sizeof(float));
        X = (float*) realloc(X, nzcount*sizeof(float));
        Y = (float*) realloc(Y, nzcount*sizeof(float));

        //print_mat(yex,nzcount,1);

        //printf("%d\n", nzcount);

        //print_mat(X,nzcount,1);
        //write_mat("Y.mat", Y,nzcount,1);

        float *A = (float*) malloc (nzcount*3*sizeof(float));
        for(i=0; i<nzcount*3; i++){
            int exp = i/nzcount;
            A[i] = pow(X[i-exp*nzcount], exp);
            //printf("%d - %f - %f\n", i, A[i], X[i-exp*nzcount]);
        }

        //char filename[10];

        //print_mat(A, nzcount, 3);
        //sprintf(filename, "A%d.mat", nn);
        //write_mat(filename, A, nzcount, 3);

        float *Acopy = (float*) malloc (nzcount*3*sizeof(float));
        memcpy(Acopy, A, nzcount*3*sizeof(float));

        float* W = (float*) malloc( imin(nzcount,3) * sizeof(float));
        float* U = (float*) malloc( nzcount * nzcount * sizeof(float));
        float* VT = (float*) malloc( 3 * 3 * sizeof(float)); 

        //Take SVD
        //trans_mat(A, nzcount, 3);
        s = culaSgesvd('A', 'A', nzcount, 3, A, nzcount, W, U, nzcount, VT, 3);
	    checkStatus(s);     

        //Pack S into a square matrix
        float *NUM = (float*) malloc(nzcount*3*sizeof(float));
        memset(NUM,0,nzcount*3*sizeof(float));
        
        for(i=0; i<imin(nzcount,3); i++)
            NUM[i*nzcount+i] = W[i];
    
        //print_mat(NUM, nzcount, 3); 
        //print_mat(U, nzcount, nzcount);

        float *tempRes1 = (float*) malloc (nzcount*3*sizeof(float));
        float *tempRes2 = (float*) malloc (nzcount*3*sizeof(float));

        matrixMul(tempRes1, NUM, VT, nzcount, 3, 3);
        matrixMul(tempRes2, U, tempRes1, nzcount, nzcount, 3);

        //write_mat("S.mat", NUM, nzcount, 3);
        //write_mat("U.mat", U, nzcount, nzcount);
        //write_mat("VT.mat", VT, 3, 3);
        //write_mat("Ares.mat", tempRes2, nzcount, 3);

        //write_mat("U.mat", U, nzcount, nzcount);
        //trans_mat(U,nzcount,nzcount);
        //write_mat("UT.mat", U, nzcount, nzcount);
        //trans_mat(U,nzcount,nzcount);
        //write_mat("UTT.mat",U, nzcount,nzcount);

        //Take transpose of U
        trans_mat(U, nzcount, nzcount);
        //trans_mat(Y, nzcount, 1);

        float *DEN = (float*) malloc(nzcount*sizeof(float));
        matrixMul(DEN, U, Y, nzcount, nzcount, 1);

        //print_mat(DEN,nzcount,1);
        //write_mat("DEN.mat", DEN, nzcount, 1);

        //Do Least Squares
        s = culaSgels('N', nzcount, 3, 1, NUM, nzcount, DEN, nzcount);
	    checkStatus(s);

        //print_mat(NUM, nzcount, 3);
        //print_mat(DEN, nzcount, 1);

        DEN = (float*) realloc(DEN, 3*sizeof(float)); 

        //print_mat(DEN, 3, 1);
        //write_mat("LS.mat", DEN, 3, 1);
        
        float *b = (float*) malloc(3* sizeof(float));
        trans_mat(VT, 3, 3);
        //write_mat("V.mat", VT, 3, 3);
        
        //trans_mat(DEN,3,1);
        matrixMul(b, VT, DEN, 3, 3, 1);

        //print_mat(b, 3, 1);
        //sprintf(filename, "B%d.mat", nn);
        //write_mat(filename, b, 3, 1);

        //trans_mat(Acopy, nzcount, 3);
        float *yco = (float*) malloc(nzcount*sizeof(float));
        matrixMul(yco, Acopy, b, nzcount, 3, 1);

        //write_mat("yco.mat", yco, nzcount, 1);
	//write_mat("yex.mat", yex, nzcount, 1);

	//write_mat("Pbef.mat", P, n, N);
	
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

        //write_mat("P.mat", P,n,N);

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

        //break;

    } 
    
    //print_mat(P,n,N);

    float price = 0.0;

    float *acdis = (float*) malloc(N*sizeof(float));
    for(i=0; i<N; i++){
        acdis[i] = pow(disc, i);
    }

    float *accum = (float*) malloc(n*sizeof(float));
    matrixMul(accum, P, acdis, n, N, 1);

    //print_mat(accum, n, 1);
    
    for(i=0; i<n; i++)
        price += accum[i];

    price /= n;

    printf("Price evaluated: %f\n", price);   

    free(S);
    free(P); 

/*

    //cumsum logic test
    
    int dW2D[3][4] = {{1,2,3,4},{4,5,6,7},{7,8,9,10}};

    for(i=0; i<3; i++){
        for(j=0; j<4; j++){
            printf("%d ", dW2D[i][j]);
        }
        printf("\n");
    }


    for(i=0; i<4; i++){
        int sum = 0;
        for(j=0; j<3; j++){
            sum += dW2D[j][i];
            dW2D[j][i] = sum;
        }
    }

    for(i=0; i<3; i++){
        for(j=0; j<4; j++){
            printf("%d ", dW2D[i][j]);
        }
        printf("\n");
    }
*/    

/*    
    float Amat[6] = {1,2,3,4,5,6};
    float Bmat[3] = {3,4,5};

    float Cmat[2];

    matrixMul(Cmat, Amat, Bmat, 2, 3, 1);

    for(i=0; i<2; i++)
        printf("%f \n", Cmat[i]);
*/ 

/*
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

    //SVD TEST
    s = culaSgesvd('A', 'A', M, N, A, LDA, S, U, LDU, VT, LDVT);
	checkStatus(s);

    printf("************\nPrinting S\n");
    for(i=0; i<imin(M,N); i++){
        printf("%f ", S[i]);
    }
    printf("\n");


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

