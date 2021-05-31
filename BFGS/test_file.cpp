// test file to try out specific calls

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>


using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::VectorXd Vector;


int main(int argc, char* argv[]){


	printf("hello\n");


	std::cout << pow(2,2) << std::endl;
	int ns = 8;
	int nb = 2;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(10, 10).sparseView(); 
	std::cout << Q_b << std::endl;

	std::cout << Q_b.leftCols(5).topRows(5) << std::endl;

	SpMat Qx(ns+nb, ns+nb);
	//Qx.setZero();

	for(int i=ns; i<(ns+nb); i++){
		Qx.insert(i,i) = 5;
	}

	std::cout << Qx << std::endl;

	int nnz = Qx.nonZeros();

	Qx.makeCompressed();
	Map<SparseMatrix<double> > Qs(12,12,nnz,Qx.outerIndexPtr(), // read-write
                               Qx.innerIndexPtr(),Qx.valuePtr());


	std::cout << Qs << std::endl;


	return 1;

}