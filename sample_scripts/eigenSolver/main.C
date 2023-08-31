#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>


#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/SparseCholesky>

#include "EigenCholSolver.h"

using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRmMat;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;


int main(int argc, char *argv[]) {

int nrows = 2000;
int ncols = 2000;

// generate sparse matrix A with random sparsity pattern
MatrixXd A_dense = MatrixXd::Random(nrows,ncols);
A_dense = A_dense*A_dense.transpose();

//std::cout << "A: \n" << A_dense << std::endl;

 for(int i=0; i<nrows; i++){
    for(int j=0; j<ncols; j++){
        if(A_dense(i,j) < 0.4){
            A_dense(i,j) = 0;
        }
    }
 }

//std::cout << "A_dense:\n" << A_dense << std::endl;
SpMat A = A_dense.sparseView();

int fakeRank = 0;
Solver *solverQ;

solverQ = new EigenCholSolver(fakeRank);

double log_det;
double time_factorize;
solverQ->factorize(A, log_det, time_factorize);

Vect rhs = Vect::Ones(nrows);
Vect sol(nrows);
double t_condLatChol; double t_condLatSolve;
solverQ->factorize_solve(A, rhs, sol, log_det, t_condLatChol, t_condLatSolve);
//printf("log det factorize solve = %f\n", log_det);

Vect invDiag(nrows);
solverQ->selected_inversion(A, invDiag);
//std::cout << "invDiag = " << invDiag.transpose() << std::endl;

exit(1);

MatrixXd Ainv(A_dense.rows(), A_dense.cols());
solverQ->compute_full_inverse(A_dense, Ainv);
std::cout << "diag(Ainv) = " << Ainv.diagonal().transpose() << std::endl;

delete solverQ;

}