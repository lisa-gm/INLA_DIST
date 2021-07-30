// test file to try out specific calls

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/Eigenvalues> 


#include <armadillo>
#include "../read_write_functions.cpp"

#include "copy_solver_pardiso.cpp"
#include "solver_cholmod.cpp"


using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::VectorXd Vector;


int main(int argc, char* argv[]){


	printf("hello\n");


	//std::cout << pow(2,2) << std::endl;
	int ns = 8;
	int nb = 2;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(10, 10).sparseView(); 
	//std::cout << Q_b << std::endl;

	//std::cout << Q_b.leftCols(5).topRows(5) << std::endl;

	/*SpMat Qx(ns+nb, ns+nb);
	//Qx.setZero();

	for(int i=ns; i<(ns+nb); i++){
		Qx.insert(i,i) = 5;
	}

	//std::cout << Qx << std::endl;

	int nnz = Qx.nonZeros();

	Qx.makeCompressed();
	SpMat Qs = Map<SparseMatrix<double> >(12,12,nnz,Qx.outerIndexPtr(), // read-write
                               Qx.innerIndexPtr(),Qx.valuePtr());*/


	//std::cout << Qs << std::endl;
	//std::cout << Qs.rows() << std::endl;
	//std::cout << Qs.cols() << std::endl;

	/*std::string filename = "/home/x_gaedkelb/b_INLA/data/ns6252/y_11646_1.dat";
	int n_row = 11646;
	int n_col = 6252;

	arma::mat X(n_row, n_col);
    X.load(filename, arma::raw_ascii);
    //X.submat(0,0,10,0).print(); */
        

	std::string g1_file = "/home/x_gaedkelb/b_INLA/data/synthetic/ns42/g1_42.dat";
	file_exists(g1_file);
    SpMat g1 = read_sym_CSC(g1_file);
    //g1 = g1.block(0,0,10,10);
    //std::cout << "g1\n" << g1 << std::endl;

	/*SparseMatrix<double> Qv(13,13);         // default is column major
	Qv.reserve(60);
	//for each i,j such that v_ij != 0

	for (int k=0; k<g1.outerSize(); ++k)
	  for (SparseMatrix<double>::InnerIterator it(g1,k); it; ++it)
	  {
	    Qv.insert(it.row(),it.col()) = it.value();                 
	  }

	  Qv.makeCompressed();   */  


    //std::cout << "Qv\n" << Qv << std::endl;

    //std::cout << "g1\n" << g1 << std::endl;

    // extract lower triangular part
    //std::cout << "g1 lower \n" << g1.triangularView<Lower>() << std::endl;

    //MatrixXd g1_dense = MatrixXd(g1);

    // compute Eigenvalues symmetric matrix 
	//SelfAdjointEigenSolver<MatrixXd> es(g1_dense);
	/*cout << "The eigenvalues of g1 are:" << endl << es.eigenvalues() << endl;
	cout << "The min eigenvalue of g1 is:" << endl << es.eigenvalues().minCoeff() << endl;*/

	/*if((es.eigenvalues().minCoeff()) <= 0){
		std::cout << "Matrix not positive definite only considering diagonal values!! " << std::endl;
		Vector diag_g1 = g1_dense.diagonal();
		g1_dense = diag_g1.asDiagonal();
    	std::cout << g1_dense << std::endl;
	}

	Vector inv_diag(g1.rows());

    std::cout << "before inv diagonal pardiso call " << std::endl;
    inv_diagonal_pardiso(&g1, inv_diag);

    std::cout << " inv diag pardiso  : " << inv_diag.transpose() << std::endl;
    // std::cout << " inv diag cholmod  : " << inv_diag_cholmod.transpose() << std::endl;*/

    Vector f(g1.rows()); f.setOnes();

    std::cout << "before inv diagonal pardiso call " << std::endl;


	int omp_get_thread_num();
	int threads = omp_get_max_threads();


    #pragma omp parallel for
    for(int i=0; i<3; i++){
    	printf("Thread rank: %d out of %d threads.\n", omp_get_thread_num(), threads);

		double log_det_det;
    	log_det_pardiso(&g1, log_det_det);
    	std::cout << "log det det : " << log_det_det << std::endl;

    	Vector inv_diag(g1.rows());
    	inv_diagonal_pardiso(&g1, inv_diag);

    	double log_det_solve;
    	Vector u(g1.rows()); u.setOnes();
    	solve_pardiso(&g1, &f, u, &log_det_solve);
    	std::cout << "log det solve : " << log_det_solve << std::endl;

    }

    /*std::cout << " inv diag pardiso  : " << inv_diag.transpose() << std::endl;


    //double log_det_det;
    std::cout << "before log det pardiso call " << std::endl;
    log_det_pardiso(&g1, log_det_det);

    double log_det_det_cholmod;
    log_det_cholmod(&g1, &log_det_det_cholmod);

	//Vector inv_diag_cholmod(g1.rows());    
    //inv_diag_cholmod(&g1, inv_diag_cholmod);

    std::cout << " log_det   : " << log_det_solve << std::endl;
    Vector f_back = g1*u;
    std::cout << " A*u : " << f_back.transpose() << std::endl;

    std::cout << " log_det pardiso  : " << log_det_det << std::endl;
    std::cout << " log_det cholmod  : " << log_det_det_cholmod << std::endl;
 
	int omp_get_thread_num();

	int threads = omp_get_max_threads();

 	#pragma omp parallel for 
 	for(int i = 0; i < 4; ++i)
 	{
 		printf("Thread rank: %d out of %d threads.\n", omp_get_thread_num(), threads);
 	}
 	*/

	return 1;

}