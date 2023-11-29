// BTASolver_dummy

#include "BTASolver_dummy.h"


BTASolver::BTASolver(size_t ns, size_t nt, size_t nb, size_t no) : ns_t(ns), nt_t(nt), nb_t(nb), no_t(no){
   	
   	std::cout << "constructing dummy BTA solver." << std::endl;

   	n = ns_t*nt_t + nb_t;

}

// currently not needed !!
void BTASolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION() not needed for BTA." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void BTASolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol) {

	std::cout << "in dummy BTA FACTORIZE()." << std::endl;

	log_det = 0;
}

void BTASolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){

	std::cout << "in dummy BTA FACTORIZE WITH CONSTRAINTS()." << std::endl;

	log_det = 0;

}


void BTASolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve) {

	std::cout << "in dummy BTA FACTORIZE_SOLVE()." << std::endl;

	log_det = 0;
  	sol = Vect::Ones(n);

}

void BTASolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){

	std::cout << "in dummy BTA FACTORIZE_SOLVE with constraints()." << std::endl;

	log_det = 0;
	sol = Vect::Ones(n);



}


// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void BTASolver::selected_inversion(SpMat& Q, Vect& inv_diag) {

	std::cout << "in dummy BTA SELECTED_INVERSION()." << std::endl;
	inv_diag = Vect::Ones(n);

}


void BTASolver::selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){
	
	std::cout << "in dummy BTA SELECTED_INVERSION with constraints()." << std::endl;
	inv_diag = Vect::Ones(n);
}




BTASolver::~BTASolver(){
    //std::cout << "Derived destructor called." << std::endl;
}



