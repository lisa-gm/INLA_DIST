#ifndef SOLVER_H
#define SOLVER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace std;
using namespace Eigen;


// Base Solver class
class Solver {
	public:
		// pure virtual function providing interface framework.
		virtual void symbolic_factorization(SpMat& Q, int& init) = 0;
		virtual void factorize(SpMat& Q, double& log_det) = 0;
		virtual void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det) = 0;
		virtual void selected_inversion(SpMat&Q, Vector& inv_diag) = 0;
   
   protected:
		SpMat Q; 			// should it be Q or &Q?

		int init;

		Vector rhs;
		Vector sol;
		double log_det;
};



#endif