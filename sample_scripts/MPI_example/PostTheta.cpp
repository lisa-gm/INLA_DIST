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

using namespace std;
using namespace Eigen;

class PostTheta {
	private:
		int n;
		int k;

		int mpi_size;
		int num_workers;

		int num_requests_send;
		int num_requests_recv;

		Vect theta;
		Vect theta_loc;
		double* theta_loc_array;

		double f_theta;


	public: 
		PostTheta(int n_) : n(n_){

			MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
			num_workers = mpi_size - 1;

			// call constructor for Solver objects for different ranks.
			cout << "constructor PostTheta." << endl;
		}

		// set up such that number of processes and n NOT the same. 
		double combined_eval(Vect theta){

			cout << "different theta evaluations." << endl;
			double f_theta_list[n];

			//double number[n];

			#if 1

			int sum = 1;

			int num_jobs = 5;
			VectorXi counter_list = VectorXi::LinSpaced(num_jobs,1,num_jobs);
			std::cout << "counter_list = " << counter_list.transpose() << std::endl;

			bool work_left = true;

			while(work_left == true){

				k = min((int)counter_list.size(), num_workers);
				std::cout << "num workers = " << num_workers << ", dim(counter_list) = " << counter_list.size() << ", k = " << k << std::endl;

				MPI_Request request_send[k];
				MPI_Status  status_send[k];
				num_requests_send = 0;

				MPI_Request request_recv[k];
				MPI_Status  status_recv[k];
				num_requests_recv = 0;

			
				for(int i=0; i<k; i++){
					Vect theta_loc = theta+counter_list[i]*Vect::Ones(theta.size());
					theta_loc_array = theta_loc.data();

					MPI_Isend(theta_loc_array, theta_loc.size(), MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD, request_send+num_requests_send);	
					//std::cout << "num requests : " << num_requests_send << std::endl;
					num_requests_send++;			
				}

				for(int i=0; i<k; i++){
					std::cout << "counter_list[" << i << "] = " << counter_list[i] << std::endl;
					MPI_Irecv(&f_theta_list[counter_list[i]], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD, \
			             request_recv+num_requests_recv);
					num_requests_recv++;
				}

				MPI_Waitall(num_requests_send, request_send, status_send);
				MPI_Waitall(num_requests_recv, request_recv, status_recv);

				//MPI_Waitall(num_requests, request, statuses);
				
				VectorXi temp = counter_list.tail(counter_list.size()-k); 
				counter_list.resize(counter_list.size()-k);				
				counter_list = temp;
				std::cout << "counter list : " << counter_list.transpose() << std::endl;

				if(counter_list.size() == 0){
					work_left = false;
				}


			}

			std::cout << "while loop done." << std::endl;

			#endif


			#if 0

			std::cout << "In PostTheta, n = " << n << std::endl;
			
			for(int i=0; i<n; i++){
				Vect theta_loc = theta+i*Vect::Ones(theta.size());
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

			#endif


			return sum;

		}


};

#endif