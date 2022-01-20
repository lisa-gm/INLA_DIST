#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include "BFGS.cpp"
#include "PostTheta.cpp"
#include "Model.cpp"

#include <mpi.h>

#include <Eigen/Core>

typedef Eigen::VectorXd Vect;


//#include <likwid-marker.h>

using namespace std;

int main(int argc, char* argv[])
{
	// Unique rank is assigned to each process in a communicator
	int rank;

	// Total number of ranks
	int size;

	// Initializes the MPI execution environment
	MPI_Init(&argc, &argv);

	// Get this process' rank (process within a communicator)
	// MPI_COMM_WORLD is the default communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get the total number ranks in this communicator
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = size - 1;
	//int n = 5;

	Vect theta(4);
	theta = 3*Vect::Ones(4);

	#if 0
	if(rank == 0){
		double *theta_array; // = (double*)malloc(theta.size() * sizeof(double));
		theta_array = theta.data();

		cout << "theta : " << theta.transpose() << endl;
		cout << "theta size = " << theta.size() << endl;

		MPI_Send(theta_array, theta.size(), MPI_DOUBLE, 0+1, 0, MPI_COMM_WORLD);

		//cout << "got after send. number = " << number  << endl;


	} else {

		MPI_Status status;

		double *theta_array = (double*)malloc(theta.size() * sizeof(double));
		MPI_Recv(theta_array, theta.size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		Vect theta2(theta.size());
		theta2 = Eigen::Map<Vect>(theta_array,theta.size());

		cout << "theta2 : " << theta2.transpose() << endl;

	}

	MPI_Finalize(); /* cleanup MPI */

	#endif

	#if 1
	if(rank > 0){	

		// WORKERS
		// construct independent models	

		Model* model;
		model = new Model(n, theta.size());
		model->ready();
	}

	if(rank == 0){

		// MASTER
		PostTheta* fun;
		fun = new PostTheta(n);

		int iter = 1;
		BFGS solver(iter);

		solver.minimize(fun, theta);

		// close workers.
		// TODO: can i send a null?
		double* theta_array = theta.data();
		for(int i=0; i<n; i++){
			std::cout << "Sending out DIETAGS to rank " << i+1 << std::endl;
			MPI_Send(theta_array, theta.size(), MPI_DOUBLE, i+1, 2, MPI_COMM_WORLD);
		}
		
	}



	MPI_Finalize(); /* cleanup MPI */

	#endif

	#if 0 


	//LIKWID_MARKER_INIT;
	//LIKWID_MARKER_START("fThetaComputation");


	// set up MPI

	if (myrank == 0) {

	} // wait until done.




	} else {
		worker  // 
	}

	//LIKWID_MARKER_STOP("fThetaComputation");
	//LIKWID_MARKER_CLOSE;
	
	#endif


}