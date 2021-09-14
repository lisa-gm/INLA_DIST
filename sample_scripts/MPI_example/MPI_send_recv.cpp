// super simple MPI send & receive 

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <iostream>

#include <unistd.h>  // for sleep()


#define WORKTAG 1
#define DIETAG 2

using namespace std;

#if 0 
void worker()
{
	int rank;
	//result_t result;
	//work_t work;

	double result;
	double work;

	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (;;) {
		MPI_Recv(&work, sizeof(double), MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		/* Check the tag of the received message */
		if (status.MPI_TAG == DIETAG) {
			return;
		}

		result = 2*work;
		sleep(1);

		printf("result of work %d on %d : %f\n", result, rank, result);
		MPI_Send(&result, sizeof(double), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
}

void master()
{
	int ntasks, rank;
	double result;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &ntasks); /* #processes in application */
	init_work(10);

	/*
	 * Seed the workers.
	 */
	work_t *work;

	for (rank = 1; rank < ntasks; ++rank) {
		work = get_next_work_request();

		printf("sending work %d\n", work->pos);
		MPI_Send(work, /* message buffer */
			sizeof(work_t), /* one data item */

			MPI_CHAR, /* data item is a struct */
			rank, /* destination process rank */
			WORKTAG, /* user chosen message tag */
			MPI_COMM_WORLD); /* always use this */
	}

	/*
	 * Receive a result from any worker and dispatch a new work
	 * until work requests have been exhausted.
	 */
	result_t res;
	work = get_next_work_request();

	while (work != NULL) {
		MPI_Recv(&res, /* message buffer */

		sizeof(result_t), /* one data item .. */
		MPI_CHAR, /* of a struct */
		MPI_ANY_SOURCE, /* receive from any sender */
		MPI_ANY_TAG, /* any type of message */
		MPI_COMM_WORLD, /* always use this */
		&status); /* received message info */
		workarray[res.pos].y = res.y;

		printf("sending work %d\n", work->pos);
		MPI_Send(work, sizeof(work_t), MPI_CHAR, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);
		work = get_next_work_request();
	}

	/*
	 * Receive results for pending work requests.
	 */
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Recv(&res,sizeof(result_t),MPI_CHAR,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		workarray[res.pos].y = res.y;
	}

	/*
	 * Tell all the workers to exit.
	 */
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Send(0, 0, MPI_CHAR, rank, DIETAG, MPI_COMM_WORLD);
	}

	/* Print the results */
	print_work();
}

#endif

#if 1
double master(double* number, int n){

	for(int i=0; i<n; i++){
		MPI_Send(&number[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD);
	}

	double result[n];

	MPI_Status statuses[n];
	MPI_Request requests[n];
	int num_requests = 0;

	for(int i=0; i<n; i++){
		/*MPI_Recv(&number[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);*/
		MPI_Irecv(&result[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD,
             requests+num_requests);
		num_requests++;
	}

	MPI_Waitall(num_requests, requests, statuses);

	double sum = 0;
	for(int i=0; i<n; i++){
		sum += result[i];
	}

    return sum;

}

void worker(){

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double number1;
	
	/*MPI_Recv(&number1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("Process %d received number %f from process 0\n",
           rank, number1);

    number1 += 3;
   	MPI_Send(&number1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);*/

	MPI_Status status;

	// infinite loop thats just waiting for a receives
	for (;;) {
		MPI_Recv(&number1, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		/* Check the tag of the received message */
		if (status.MPI_TAG == DIETAG) {
			cout << "Process " << rank << " received DIETAG." << endl;
			return;
		}

	    number1 += 3;

		sleep(1);
	    printf("Process %d received number %f from process 0\n", rank, number1);

   		MPI_Send(&number1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

}

#endif

int main(int argc, char *argv[]) {
	// Unique rank is assigned to each process in a communicator
	int rank;

	// Total number of ranks
	int size;

	// The machine we are on
	char name[80];

	// Length of the machine name
	int length;

	// Initializes the MPI execution environment
	MPI_Init(&argc, &argv);

	// Get this process' rank (process within a communicator)
	// MPI_COMM_WORLD is the default communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get the total number ranks in this communicator
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = size-1;

	if(rank == 0){
		cout << "number of MPI workers = " << n << endl;
	}

	if(rank > 0){
		worker();
	}

	double number[n];
	double check_sum=0;
	double sum;

	if (rank == 0) {

		for(int i=0; i<n; i++){
		    number[i] = i+2;
		}

		sum = master(number, n);

		for(int i=0; i<n; i++){
			check_sum += number[i]+3;
		}

		printf("Process %d received the following sum : %f. Should be %f\n",
	         rank, sum, check_sum);	

		for(int i=0; i<n; i++){
			MPI_Send(&number[i], 1, MPI_DOUBLE, i+1, 2, MPI_COMM_WORLD);
		}


	} 

	// Terminate MPI execution environment
	MPI_Finalize();

  return 0;
}

	#if 0 

	int n = size-1;

	if(rank == 0){
		cout << "number of MPI workers = " << n << endl;
	}

	double number[n];
	double result[n];
	double check_sum=0;
	double sum = 0;

	if (rank == 0) {

		for(int i=0; i<n; i++){
		    number[i] = i+2;
		}

		for(int i=0; i<n; i++){
			check_sum += number[i]+3;
		}

		for(int i=0; i<n; i++){
			MPI_Send(&number[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD);
		}

		MPI_Status statuses[n];
		MPI_Request requests[n];
		int num_requests = 0;

		for(int i=0; i<n; i++){
			/*MPI_Recv(&number[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD,
	             MPI_STATUS_IGNORE);*/
			MPI_Irecv(&result[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD,
	             requests+num_requests);
			num_requests++;
		}

		MPI_Waitall(num_requests, requests, statuses);

		for(int i=0; i<n; i++){
			sum += result[i];
		}

		printf("Process %d received the following sum : %f. Should be %f\n",
	         rank, sum, check_sum);	

	} else {
		double number1;
		MPI_Recv(&number1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
	             MPI_STATUS_IGNORE);
	    printf("Process %d received number %f from process 0\n",
	           rank, number1);

	    number1 += 3;
	   	MPI_Send(&number1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

	}
	}

	#endif