#include <cula.h>
#include <stdio.h>

int main(){
	culaStatus s;
	s = culaInitialize();
	if(s != culaNoError)
	{
	printf("%s\n", culaGetStatusString(s));
	/* ... Error Handling ... */
	}
	/* ... Your code ... */
	culaShutdown();
}


