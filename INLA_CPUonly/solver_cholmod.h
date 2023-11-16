#ifndef __solver_cholmod
#define __solver_cholmod

#include <math.h>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>


using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::CholmodSimplicialLDLT  <SpMat > Solver;
typedef Eigen::VectorXd Vector;


void log_det_cholmod(SpMat *A, double *log_det);

void solve_cholmod(SpMat *A, Vector *f, Vector& u, double *log_det);

void inv_diagonal_cholmod(SpMat* Q, Vector& vars);

void compute_inverse_cholmod(MatrixXd& Q, MatrixXd& Q_inv);

#endif
