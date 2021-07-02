#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main (int argc, char *argv[]) {

	int nthreads, tid;

	/* Fork a team of threads giving them their own copies of variables */
	#if 0
	#pragma omp parallel private(nthreads, tid)
	 {

		 /* Obtain thread number */
		 tid = omp_get_thread_num();
		 printf("Hello World from thread = %d\n", tid);

		 /* Only master thread does this */
		 if (tid == 0) 
		 {
			  nthreads = omp_get_num_threads();
			  printf("Number of threads = %d\n", nthreads);
		 }

	 } /* All threads join master thread and disband */
    printf("\n");

    #endif

    int threads = omp_get_max_threads();

   #pragma omp parallel for
	for(int i=0; i<4; i++){
		/* Obtain thread number */
		tid = omp_get_thread_num();
		printf("1st layer HI from thread = %d out of %d\n", tid, threads);

		for(int k=0; k<3; k++){
			printf("2nd layer HI from thread = %d out of %d\n", tid, threads);

		}

	}

	

}