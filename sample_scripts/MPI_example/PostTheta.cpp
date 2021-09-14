#ifndef PostTheta__
#define PostTheta__

// dummy class for PostTheta 

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#include <Eigen/Core>

#include "Model.cpp"


typedef Eigen::VectorXd Vector;

using namespace std;

class PostTheta {
	private:
		int n;

		Vector theta;
		Vector theta_loc;
		double* theta_loc_array;

		double f_theta;


	public: 
		PostTheta(int n_) : n(n_){

			// call constructor for Solver objects for different ranks.
			cout << "constructor PostTheta." << endl;
		}

		double combined_eval(Vector theta){

			cout << "different theta evaluations." << endl;
			double f_theta_list[n];

			//double number[n];

			for(int i=0; i<n; i++){
				Vector theta_loc = theta+i*Vector::Ones(theta.size());
				theta_loc_array = theta_loc.data();

				MPI_Send(theta_loc_array, theta_loc.size(), MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD);

			}

			MPI_Status statuses[n];
			MPI_Request requests[n];
			int num_requests = 0;

			cout << "sent done. waiting for receive." << endl;

			for(int i=0; i<n; i++){
				/*MPI_Recv(&number[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD,
		             MPI_STATUS_IGNORE);*/
				MPI_Irecv(&f_theta_list[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD,
		             requests+num_requests);
				num_requests++;
			}

			MPI_Waitall(num_requests, requests, statuses);

			double sum = 0;
			for(int i=0; i<n; i++){
				sum += f_theta_list[i];
			}

			double check_sum = 0;
			for(int i=0; i<n; i++){
				check_sum += 2*(theta(0)+i);
			}

			cout << "check sum : " << check_sum << endl;
			return sum;
		}

};

#endif