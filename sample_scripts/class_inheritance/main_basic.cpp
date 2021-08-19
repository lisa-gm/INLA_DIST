
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include "PardisoSolver.h"
#include "RGFSolver.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace std;
using namespace Eigen;
  
 
int main(int argc, char* argv[]) {

    if(argc != 1 + 1){
    	cout << "wrong input length." << endl;
    	exit(1);
    }

	int ns = 3;

	Solver* solver;
	string solver_type = argv[1];

	int threads = omp_get_max_threads();
	cout << "number of available threads = " << threads << endl;

	if(solver_type == "PARDISO"){
		solver = new PardisoSolver;
	} else if(solver_type == "RGF"){
		solver = new RGFSolver(ns);
	} else {
		cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << endl;
	}

	int n = 5;
	SpMat Q(5,5); Q.setIdentity();

	solver->symbolic_factorization(Q);

	#if 0
	RGFSolver  R_solver(ns);

	P_solver.setWidth(5);
	P_solver.setHeight(7);

	int a = 5;

	// Print the area of the object.
	cout << "Total P_solver area: " << P_solver.getArea(a) << endl;

	R_solver.setWidth(5);
	R_solver.setHeight(7);

	// Print the area of the object.
	cout << "Total R_solver area: " << R_solver.getArea(a) << endl; 


	int n = 5;
	SpMat Q(5,5); Q.setIdentity();

	P_solver.symbolic_factorization(Q);
	R_solver.symbolic_factorization(Q);

	P_solver.factorize(Q);
	R_solver.factorize(Q);

	Vector x(n);
	P_solver.factorize_solve(Q,x);
	std::cout << "x : " << x.transpose() << std::endl;
	R_solver.factorize_solve(Q,x);
	std::cout << "x : " << x.transpose() << std::endl;

	Vector inv_diag(n);
	P_solver.selected_inversion(Q,inv_diag);
	std::cout << "inv diag : " << inv_diag.transpose() << std::endl;
	R_solver.selected_inversion(Q,inv_diag);
	std::cout << "inv diag : " << inv_diag.transpose() << std::endl;
	#endif

   return 0;
}


