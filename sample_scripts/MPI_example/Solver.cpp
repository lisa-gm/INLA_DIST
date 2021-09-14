#ifndef Solver__
#define Solver__

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Core>

typedef Eigen::VectorXd Vector;

using namespace std;

// dummy solver class to be called by model class through workers
class Solver {
	private:
		int n;
		Vector theta;
		double f_theta;

	public: 
		Solver(){
			cout << "Solver constructor." << endl;
		}

		double factorize(Vector theta){
			f_theta = 2*theta(0);

			return f_theta;
		}




};


#endif