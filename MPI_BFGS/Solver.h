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

#include "mpi.h"

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace std;
using namespace Eigen;

 /**
 * @brief abstract base solver class to enable to be able two switch between solvers 
 * (current options: PARDISO & GPU) at run time.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 */
class Solver {
	public:
		// pure virtual function providing interface framework.
		virtual void symbolic_factorization(SpMat& Q, int& init) = 0;
		virtual void factorize(SpMat& Q, double& log_det) = 0;
		virtual void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det) = 0;
		virtual void selected_inversion(SpMat&Q, Vector& inv_diag) = 0;

		// "simple inversion" function for small matrices. exists already in pardiso.
   
   protected:
   		int init;			/**< flag indicating if symbolic factorisation
   								 was already performed or not             */
   		double log_det;		/**< log determinant of Q 					  */

		SpMat Q; 			/**< sparse precision matrix Q. Eigen format. */
		Vector rhs;			/**< right-hand side, solving Q*sol = rhs     */
		Vector sol;			/**< solution vector, solving Q*sol = rhs     */
};



#endif