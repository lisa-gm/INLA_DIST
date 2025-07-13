// PardisoSolver_dummy

#include "PardisoSolver_dummy.h"


PardisoSolver::PardisoSolver(int MPI_rank_) : MPI_rank(MPI_rank_){
   	
   	std::cout << "constructing dummy Pardiso solver." << std::endl;

}

// currently not needed !!
void PardisoSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "SYMBOLIC_FACTORIZATION()." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void PardisoSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol) {

	std::cout << "in dummy Pardiso FACTORIZE()." << std::endl;

	log_det = 0;
}

void PardisoSolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){
	
    std::cout << "in dummy Pardiso FACTORIZE_w_CONSTR()." << std::endl;

	log_det = 0;
}

void PardisoSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve) {
    n = Q.rows();

	std::cout << "in dummy Pardiso FACTORIZE_SOLVE()." << std::endl;

	log_det = 0;
  	sol = Vect::Ones(n);

}

void PardisoSolver::fused_factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatCholForwardSolve, double& t_condLatBackwardSolve){
	n = Q.rows();
	
	std::cout << "in dummy Pardiso FUSED_FACTORIZE_SOLVE()." << std::endl;

	log_det = 0;
  	sol = Vect::Ones(n);
}


void PardisoSolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){
	n = Q.rows();

    std::cout << "in dummy Pardiso FACTORIZE_SOLVE_w_CONSTR()()." << std::endl;

	log_det = 0;
  	sol = Vect::Ones(n);

}

// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void PardisoSolver::selected_inversion(SpMat& Q, Vect& inv_diag) {
    n = Q.rows();

	std::cout << "in dummy Pardiso SELECTED_INVERSION()." << std::endl;
	inv_diag = Vect::Ones(n);

}

void PardisoSolver::selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){
    n = Q.rows();

	std::cout << "in dummy Pardiso SELECTED_INVERSION_w_CONSTR()." << std::endl;
	inv_diag = Vect::Ones(n);  
}



PardisoSolver::~PardisoSolver(){
    //std::cout << "Derived destructor called." << std::endl;
}
