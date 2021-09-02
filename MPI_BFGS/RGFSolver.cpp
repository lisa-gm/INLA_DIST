#include "RGFSolver.h"

RGFSolver::RGFSolver(){
   	std::cout << "constructing RGF solver." << std::endl;
}

void RGFSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION()." << std::endl;
}

void RGFSolver::factorize(SpMat& Q, double& log_det) {
	log_det = 1;
	std::cout << "Placeholder FACTORIZE()." << std::endl;

}

void RGFSolver::factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det) {
	sol.setOnes();
	std::cout << "Placeholder FACTORIZE_SOLVE()." << std::endl;
}

void RGFSolver::selected_inversion(SpMat& Q, Vector& inv_diag) {
	inv_diag = 5*Vector::Ones(inv_diag.size());
	std::cout << "Placeholder SELECTED_INVERSION()." << std::endl;
}


/*
RGFSolver::~RGFSolver(){

}*/


