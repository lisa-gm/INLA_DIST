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


using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRmMat;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;


int main(int argc, char *argv[]) {

int nrows = 4;
int ncols = 4;

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

 MatrixXd V = MatrixXd::Random(ncols,1);
 std::cout << "V: " << V.transpose() << std::endl;
 MatrixXd S = 2.0*MatrixXd::Random(1, ncols);
 std::cout << "V: " << S << std::endl;

 std::cout << "A_dense:\n" << A_dense << std::endl;

SpMat A = A_dense.sparseView();

for (int k=0; k<A.outerSize(); ++k)
  for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
  {
    std::cout << it.value() << " ";
    std::cout << it.row() << " ";   // row index
    std::cout << it.col() << " ";   // col index (here it is equal to k)
    std::cout << it.index() << std::endl; // inner index, here it is equal to it.row()

    std::cout << "V.row(" << it.row() << "): " << V.row(it.row()) << std::endl;
    std::cout << "S.col(" << it.col() << "): " << S.col(it.col()) << std::endl;

    it.valueRef() = V.row(it.row())*S.col(it.col());
  }

std::cout << "A:\n" << MatrixXd(A) << std::endl;
std::cout << "V*S = \n" << V*S << std::endl;
std::cout << "A - V*S = \n" << A - V*S << std::endl;



}