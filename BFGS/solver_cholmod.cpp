
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>


using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::CholmodSimplicialLDLT  <SpMat > Solver;
typedef Eigen::VectorXd Vector;


void log_det_cholmod(SpMat *A, double *log_det)
{

	Solver solver;
	solver.analyzePattern(*A);
	solver.factorize(*A);

	*log_det = solver.logDeterminant();

	//std::cout << "solution vector u : " << *u << std::endl;

}

void solve_cholmod(SpMat *A, Vector *f, Vector& u, double *log_det)
{

	Solver solver;
	solver.analyzePattern(*A);
	solver.factorize(*A);

	*log_det = solver.logDeterminant();

	u = solver.solve(*f);

	//std::cout << "solution vector u : " << *u << std::endl;

}

void inv_diagonal_cholmod(SpMat* Q, Vector& vars){

		MatrixXd Q_dense = MatrixXd(*Q);
		//std::cout << "Q dense :\n" << Q_dense << std::endl; 

		MatrixXd Q_inv = Q_dense.inverse();		
		//std::cout << "Q inv :\n" << Q_inv << std::endl; 

		vars = Q_inv.diagonal();


}

void compute_inverse_cholmod(MatrixXd& Q, MatrixXd& Q_inv){

		Q_inv = Q.inverse();		
		//std::cout << "Q inv :\n" << Q_inv << std::endl; 


}


