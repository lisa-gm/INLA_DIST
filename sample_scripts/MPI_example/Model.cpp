#ifndef Model__
#define Model__

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"
#include <unistd.h>  // for sleep()

#include <Eigen/Core>

#include "Solver.cpp"

typedef Eigen::VectorXd Vector;


#define WORKTAG 1
#define DIETAG 2

using namespace std;

// construct model here, call solver from here
class Model {
	private:
		int n;
		int theta_size;

		Vector theta;
		double* theta_array;

		double f_theta;
		Solver* solver;

	public: 
		Model(int n_, int theta_size_) : n(n_), theta_size(theta_size_) {
			cout << "Model constructor." << endl;
			solver = new Solver();

			theta_array = (double*)malloc(theta_size * sizeof(double));   
		}

		void ready(){

		cout << "entered ready function." << endl;

		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		MPI_Status status;

		// infinite loop thats just waiting for receives
		for (;;) {
			MPI_Recv(theta_array, theta_size, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			/* Check the tag of the received message */
			if (status.MPI_TAG == DIETAG) {
				cout << "Process " << rank << " received DIETAG." << endl;
				return;
			}

			// map to eigen vector format
			theta = Eigen::Map<Vector>(theta_array,theta_size);

			cout << "received theta in model : " << theta.transpose() << endl;

			f_theta = solver->factorize(theta);
			// number1 += 3;

			sleep(1);
		    printf("Process %d received number %f from process 0\n", rank, theta(0));

	   		MPI_Send(&f_theta, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}

	} // end ready() function




}; // end class


#endif