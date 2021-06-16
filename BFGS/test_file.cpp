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

#include <armadillo>
#include "../read_write_functions.cpp"



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
	SpMat Qs = Map<SparseMatrix<double> >(12,12,nnz,Qx.outerIndexPtr(), // read-write
                               Qx.innerIndexPtr(),Qx.valuePtr());


	std::cout << Qs << std::endl;
	std::cout << Qs.rows() << std::endl;
	std::cout << Qs.cols() << std::endl;

	std::string filename = "/home/x_gaedkelb/b_INLA/data/ns6252/y_11646_1.dat";
	int n_row = 11646;
	int n_col = 6252;

	arma::mat X(n_row, n_col);
    X.load(filename, arma::raw_ascii);
    X.submat(0,0,10,0).print();
        

	std::string g1_file = "/home/x_gaedkelb/b_INLA/data/ns42/g1_42.dat";
    SpMat g1 = read_sym_CSC(g1_file);
    g1 = g1.block(0,0,10,10);
    std::cout << "g1\n" << g1 << std::endl;

	SparseMatrix<double> Qv(13,13);         // default is column major
	Qv.reserve(60);
	//for each i,j such that v_ij != 0

	for (int k=0; k<g1.outerSize(); ++k)
	  for (SparseMatrix<double>::InnerIterator it(g1,k); it; ++it)
	  {
	    Qv.insert(it.row(),it.col()) = it.value();                 
	  }

	  Qv.makeCompressed();     


    std::cout << "Qv\n" << Qv << std::endl;



	return 1;

}