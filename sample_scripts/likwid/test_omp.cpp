#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>


//#include <Eigen/Dense>
#include <iostream>

using namespace std;
//using namespace Eigen;

void report_num_threads(int level)
{
    #pragma omp single
    {
        printf("Level %d: number of threads in the team - %d\n",
                  level, omp_get_num_threads());
    }
 }

 int compute(int n){

    int fk = 0;
    int fn = 1;
    int tmp;

    for(int i=0; i<n; i++){

        tmp = fn + fk;
        fk = fn;
        fn = tmp;
    }

    return fn;

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


    // OPENMP TASKING
    #if 0

    int dim_th = 4;
    int loop_dim = 16;

    double theta = 2;

    double eps = 0.5;

    Eigen::MatrixXd hessUpper = Eigen::MatrixXd::Zero(dim_th, dim_th);

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i = Eigen::MatrixXd::Zero(3,dim_th);
    Eigen::MatrixXd f_i_j = Eigen::MatrixXd::Zero(4,loop_dim);

    std::cout << "f_i_i :\n" << f_i_i << std::endl;

    #pragma omp parallel
    #pragma omp single
    {

    // compute f(theta) only once.
    #pragma omp task 
    { 
    f_i_i.row(1) = theta*theta * Eigen::VectorXd::Ones(dim_th).transpose(); 
    }

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){

            #pragma omp task 
            { 
            double theta_forw_i = theta+k*eps;
            f_i_i(0,i) = theta_forw_i*theta_forw_i; 
            }

            # pragma omp task
            { 
            double theta_back_i = theta-k*eps;
            f_i_i(2,i) = theta_back_i*theta_back_i; 
            }

        
        // symmetric only compute upper triangular part
        } else if(j > i) {

            #pragma omp task 
            { 
            double theta_forw_i_j = theta+k*eps+j*eps;
            f_i_j(0,k) = theta_forw_i_j; 
            }

            #pragma omp task 
            { 
            double theta_forw_i_back_j = theta+k*eps-j*eps;
            f_i_j(1,k) = theta_forw_i_back_j; 
            }

            #pragma omp task 
            { 
            double theta_back_i_forw_j = theta-i*eps+j*eps;
            f_i_j(2,k) = theta_back_i_forw_j; 
            }

            #pragma omp task 
            { 
            double theta_back_i_j = theta-i*eps-j*eps;
            f_i_j(3,k) = theta_back_i_j; 
            }            
        }

    }

    #pragma omp taskwait

    std::cout << "f_i_i \n" << f_i_i << std::endl;
    std::cout << "f_i_j \n" << f_i_j << std::endl;



    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){
            hessUpper(i,i) = (f_i_i(0,i) - 2 * f_i_i(1,i) + f_i_i(2,i))/(eps*eps);

        } else if(j > i){
            //std::cout << "k = " << k << ", i = " << i << ", j = " << j << std::endl;
            hessUpper(i,j) = (f_i_j(0,k) + f_i_j(1,k) + f_i_j(2,k) + f_i_j(3,k)) / (4*eps*eps);
        }

    }

    } // end omp

    std::cout << "hess Upper \n" << hessUpper << std::endl;

    int n = 5;

    double* b;
    double* x;

    b = new double [n*n];
    x = new double [n*n];

    /* Set right hand side to Identity. */
    for (int i = 0; i < n*n; i++) {
        if(i % (n+1) == 0){
            b[i] = 1.0;
        } else{
            b[i] = 0;
        }
    } 

    std::cout << "b = ";

    for (int i = 0; i < n*n; i++) {
        std::cout << b[i] << " ";
    }

    std::cout << std::endl;

    #endif


    /*double* array;
    array = new double[8];
    const int n = 2;
    const int m = 4;
    for(int i = 0; i < 8; ++i) array[i] = i*1.0;
    cout << "Column-major:\n" << Map<Matrix<double,n,m> >(array) << endl;*/


    #if 1

    double get_time = -omp_get_wtime();
	
    #pragma omp parallel 
    //for(int i = 0; i < 4; ++i)
    {
  
		threads = omp_get_num_threads();
		tid = omp_get_thread_num();

		printf("1st layer, thread = %d out of %d threads\n", tid, threads);
        sleep(1);


        #pragma omp parallel 
        {
        	tid = omp_get_thread_num();
        	int max_threads = omp_get_max_threads();
			int threads = omp_get_num_threads();
			printf("2nd layer, thread = %d out of %d threads\n", tid, threads);
            
            //sleep(2);

            // add computational task to see threads working
            int fn = compute(1000000000);
            //std::cout << "pi : " << pi << std::endl;


        }
    }

    get_time += omp_get_wtime();

    std::cout << "time passed : " << get_time << std::endl;

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