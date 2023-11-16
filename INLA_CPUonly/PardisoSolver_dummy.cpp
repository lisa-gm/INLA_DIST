// PardisoSolver_dummy

#include "PardisoSolver_dummy.h"


PardisoSolver::PardisoSolver(size_t ns, size_t nt, size_t nb, size_t no) : ns_t(ns), nt_t(nt), nb_t(nb), no_t(no){
   	
   	std::cout << "constructing dummy Pardiso solver." << std::endl;

   	n = ns_t*nt_t + nb_t;

}

// currently not needed !!
void PardisoSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "SYMBOLIC_FACTORIZATION()." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void PardisoSolver::factorize(SpMat& Q, double& log_det) {

	std::cout << "in dummy Pardiso FACTORIZE()." << std::endl;

	log_det = 0;
}

void PardisoSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det) {

	#ifdef PRINT_MSG
	std::cout << "in dummy Pardiso FACTORIZE_SOLVE()." << std::endl;
	#endif

	log_det = 0;
  	sol = Vect::Ones(n);

}

// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void PardisoSolver::selected_inversion(SpMat& Q, Vect& inv_diag) {

	std::cout << "in dummy Pardiso SELECTED_INVERSION()." << std::endl;
	inv_diag = Vect::Ones(n);

}



PardisoSolver::~PardisoSolver(){
    //std::cout << "Derived destructor called." << std::endl;
}
