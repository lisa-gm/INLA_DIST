// dummy class for BFGS solver

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Core>

typedef Eigen::VectorXd Vector;

using namespace std;

#include "PostTheta.cpp"

// ideally we are NOT adding/changing anything here.
class BFGS {
	private:
		int n;
		Vector theta;
		double f_theta;
		PostTheta* fun;
		int iter;

	public:
		BFGS(int iter_) : iter(iter_){
			cout << "constructor BFGS." << endl;
		}

		void minimize(PostTheta* fun, Vector theta){
			cout << "in minimize function." << endl;

			for(int i=0; i<iter; i++){
				cout << "\niter : " << i << endl;
				f_theta = fun->combined_eval(theta);
				cout << "f_theta : " << f_theta << endl;
				theta = f_theta*Vector::Ones(theta.size());
			}
		}




};