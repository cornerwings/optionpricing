#include <stdlib.h>
#include <stdio.h>


extern void sgesvd( char* jobu, char* jobvt, int* m, int* n, float* a,
                int* lda, float* s, float* u, int* ldu, float* vt, int* ldvt,
                float* work, int* lwork, int* info );

extern void print_matrix( char* desc, int m, int n, float* a, int lda );

void read_mat(float *A, int M, int N){
    FILE *fp;
    fp = fopen("A.mat", "r");

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

/* Parameters */
#define M 540
#define N 3
#define LDA M
#define LDU M
#define LDVT N

/* Main program */
int main() {
        /* Locals */
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        float wkopt;
        float* work;
        /* Local arrays */
        float s[N], u[LDU*M], vt[LDVT*N];
        float a[LDA*N];

	read_mat(a, LDA, N);

        printf( " SGESVD Example Program Results\n" );
        lwork = -1;
        sgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
         &info );
        lwork = (int)wkopt;
        work = (float*)malloc( lwork*sizeof(float) );

        sgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
         &info );

        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }

        print_matrix( "Singular values", 1, n, s, 1 );

        //print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
	write_mat("U.mat", u, ldu, m);

        //print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
	write_mat("V.mat", vt, ldvt, n);

        free( (void*)work );
        exit( 0 );
} 


void print_matrix( char* desc, int m, int n, float* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}

