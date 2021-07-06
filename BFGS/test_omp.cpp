#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


void report_num_threads(int level)
{
    #pragma omp single
    {
        printf("Level %d: number of threads in the team - %d\n",
                  level, omp_get_num_threads());
    }
 }


int main (int argc, char *argv[]) {

	int threads, tid;

	//int threads = omp_get_max_threads();


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

    #if 0
   	#pragma omp parallel for
	for(int i=0; i<4; i++){
		/* Obtain thread number */
		tid = omp_get_thread_num();
		printf("1st layer HI from thread = %d out of %d\n", tid, threads);

		for(int k=0; k<3; k++){
			printf("2nd layer HI from thread = %d out of %d\n", tid, threads);

		}

	}

	#endif

    #if 1
	

    #pragma omp parallel for 
    for(int i = 0; i < 4; ++i)
    {
  
		threads = omp_get_num_threads();
		tid = omp_get_thread_num();

		printf("1st layer, thread = %d out of %d threads\n", tid, threads);

        #pragma omp parallel 
        {
        	tid = omp_get_thread_num();
        	int max_threads = omp_get_max_threads();
			int threads = omp_get_num_threads();
			printf("2nd layer, thread = %d out of %d threads\n", tid, threads);

        }
    }

    #endif

	//omp_set_nested(2);

    #if 0

	int max_threads = omp_get_max_threads();
	int threads = omp_get_num_threads();

    #pragma omp parallel for
	for(int i=0; i<4; i++){
		/* Obtain thread number */
		tid = omp_get_thread_num();
		printf("i = %d, 1st layer HI from thread = %d out of max %d and threads %d\n", i, tid, max_threads, threads);

		max_threads = omp_get_max_threads();
		threads = omp_get_max_threads();

		#pragma omp parallel for
		for(int k=0; k<3; k++){
			printf("k = %d, 2nd layer HI from thread = %d out of max %d and threads %d\n", k, tid, max_threads, threads);

		}
	}

	#endif

   	#if 0
	omp_set_nested(1);
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0)
            omp_set_num_threads(4);       /* line A */
        else
            omp_set_num_threads(6);       /* line B */

        /* The following statement will print out
         *
         * 0: 2 4
         * 1: 2 6
         *
         * omp_get_num_threads() returns the number
         * of the threads in the team, so it is
         * the same for the two threads in the team.
         */
        printf("%d: %d %d\n", omp_get_thread_num(),
               omp_get_num_threads(),
               omp_get_max_threads());

        /* Two inner parallel regions will be created
         * one with a team of 4 threads, and the other
         * with a team of 6 threads.
         */
        #pragma omp parallel
        {
        	//printf("Inner before master: %d\n", omp_get_num_threads());
        	
            #pragma omp master
            {
                /* The following statement will print out
                 *
                 * Inner: 4
                 * Inner: 6
                 */
                printf("Inner: %d\n", omp_get_num_threads());
            }
            //omp_set_num_threads(7);      /* line C */
        }

         /* Again two inner parallel regions will be created,
         * one with a team of 4 threads, and the other
         * with a team of 6 threads.
         *
         * The omp_set_num_threads(7) call at line C
         * has no effect here, since it affects only
         * parallel regions at the same or inner nesting
         * level as line C.
         */

        #pragma omp parallel
        {
            printf("count me.\n");
        }
    }

    #endif




	

}