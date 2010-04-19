#include <stddef.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" double dnrm2_(int *n,double *x, int* incx);
extern "C" void   dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, 
double *b, int *ldb, int *info );
extern "C" void   dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern "C" void   dgemm_(char *transa, char *transb,
                       int *m, int *n, int *k, double *alpha, double *a, int 
*lda, double *b, int *ldb,
                       double *beta, double *c, int *ldc );

int main( ){

       int n=100, k=1;
       double A[n*n], b[n*k], Acopy[n*n], bcopy[n*k];
       int ipiv[n], info;
       int i;
       double normr, normb;

       int ione=1, int01;
       char charN = 'N';
       double dpone = 1.e0, dmone = -1.e0;

       for(i=0;i<n*n;i++)
               A[i] = ((double) rand()) / ((double) RAND_MAX) - 0.5;
       for(i=0;i<n*k;i++)
               b[i] = ((double) rand()) / ((double) RAND_MAX) - 0.5;


       int01 = n*k; dcopy_(&int01,b,&ione,bcopy,&ione);
       int01 = n*n; dcopy_(&int01,A,&ione,Acopy,&ione);

       dgesv_(&n, &k, A, &n, ipiv, b, &n, &info);

       normb = dnrm2_(&n,bcopy,&ione);
       dgemm_ (&charN, &charN, &n, &k, &n, &dpone, Acopy, &n, b, &n, &dmone, 
bcopy, &n);
       normr = dnrm2_(&n,bcopy,&ione);
       printf("\tDGESV [n=%d,k=%d] \t\t(check:%e)\n",
               n,k,normr/normb);

       return 0;

}

