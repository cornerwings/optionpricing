#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

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

void simulate_assets(option_data *option, int points, int paths){

	//Other constants
	float dt = (float)option->time/points;
	float disc = exp(-option->interest_rate*dt);

	float *binc = (float *) malloc( (points-1)*paths*sizeof(float));
	brownian_inc(binc, points-1, paths, dt);

	float cadd = ( option->interest_rate - 0.5*pow(option->volatility,2) )	* dt;

	int i,j;
	for(i=0; i<points-1; i++){
		float sum = 0;
		for(j=0; j<paths; j++){
			float temp = cadd + option->volatility*binc[i*paths+j];
			sum += temp;
			binc[i*paths+j] = option->initial_price * exp(sum);
		}
	}
			
}
	
int main(){

	//Allocating memory for a new option
	option_data *option = (option_data*)malloc(sizeof(option_data));

	//Option data
	option->initial_price = 100;
	option->strike_price = 100;
	option->time = 1;
	option->volatility = 0.2;
	option->interest_rate = 0.05;

	//Number of exercise points
	int points = 16;

	//Number of monte carlo paths to simulate
	int paths = 1000;

	assert(randn_ok()==1);

	simulate_assets(option, points, paths);

	float A[]={1,4,7,2,5,8,3,6,9};
	cumsum(A,3,3);
	print(A,3,3);

	return 0;
}
