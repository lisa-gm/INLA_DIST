// RGFSolver_dummy

#include "RGFSolver_dummy.h"


RGFSolver::RGFSolver(size_t ns, size_t nt, size_t nb, size_t no) : ns_t(ns), nt_t(nt), nb_t(nb), no_t(no){
   	
   	std::cout << "constructing dummy RGF solver." << std::endl;

   	n = ns_t*nt_t + nb_t;

}

// currently not needed !!
void RGFSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION() not needed for RGF." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void RGFSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol) {

	std::cout << "in dummy RGF FACTORIZE()." << std::endl;

	log_det = 0;
}

void RGFSolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){

	std::cout << "in dummy RGF FACTORIZE WITH CONSTRAINTS()." << std::endl;

	log_det = 0;

}


void RGFSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve) {

	std::cout << "in dummy RGF FACTORIZE_SOLVE()." << std::endl;

	log_det = 0;
  	sol = Vect::Ones(n);

}

void RGFSolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){

	std::cout << "in dummy RGF FACTORIZE_SOLVE with constraints()." << std::endl;

	log_det = 0;
	sol = Vect::Ones(n);



}


// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void RGFSolver::selected_inversion(SpMat& Q, Vect& inv_diag) {

	std::cout << "in dummy RGF SELECTED_INVERSION()." << std::endl;
	inv_diag = Vect::Ones(n);

}


void RGFSolver::selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){
	
	std::cout << "in dummy RGF SELECTED_INVERSION with constraints()." << std::endl;
	inv_diag = Vect::Ones(n);
}




RGFSolver::~RGFSolver(){
    //std::cout << "Derived destructor called." << std::endl;
}



