// generate test matrices BTA block inversion

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
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>

using Eigen::MatrixXd;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::VectorXd Vect;


// hardcode very simple test case
SpMat gen_test_mat_base1(){
	SpMat Q(5,5);
	int n = Q.cols();

	Vect d(5);
	Vect offD(3);
	Vect denseD(4);

	d      << 2,4,7,4,5;
	offD   << -0.2, -0.05, -0.1;
	denseD << -0.4, -0.3, -0.2, -0.1;

	std::cout << "d      = " << d.transpose() << std::endl;
	std::cout << "offD   = " << offD.transpose() << std::endl;
	std::cout << "denseD = " << denseD.transpose() << std::endl;


	// insert elements
	for(int i=0; i<n; i++){
		Q.insert(i, i) = d[i];

		if(i < n-1){
			Q.insert(n-1, i)= denseD[i];
			Q.insert(i, n-1)= denseD[i];
		}  

		if(i < n-2){
			Q.insert(i,i+1) = offD[i];
			Q.insert(i+1,i) = offD[i];
		}

	}

	Q.makeCompressed();
	return Q;
}


SpMat gen_test_mat_base2(){
	SpMat Q(7,7);
	int n = Q.cols();

	Vect d(7);
	Vect offD(5);
	Vect denseD(6);

	d      << 8,2,4,7,4,5,11;
	offD   << -0.8, -0.3, -0.2, -0.05, -0.1;
	denseD << -0.2, -0.5, -0.4, -0.3, -0.2, -0.1;

	std::cout << "d      = " << d.transpose() << std::endl;
	std::cout << "offD   = " << offD.transpose() << std::endl;
	std::cout << "denseD = " << denseD.transpose() << std::endl;


	// insert elements
	for(int i=0; i<n; i++){
		Q.insert(i, i) = d[i];

		if(i < n-1){
			Q.insert(n-1, i)= denseD[i];
			Q.insert(i, n-1)= denseD[i];
		}  

		if(i < n-2){
			Q.insert(i,i+1) = offD[i];
			Q.insert(i+1,i) = offD[i];
		}

	}

	Q.makeCompressed();
	return Q;
}

SpMat gen_test_mat_base3(int ns, int nt, int nb){

	//int nb = 2;
	int n  = ns*nt + nb;
	MatrixXd Q_d(n,n);
	Q_d.setZero();

	MatrixXd A1(ns, ns);
	MatrixXd A2(ns, ns);

	if(ns == 2){
		A1 << 10, -1,
			  -1, 10;

		A2 << -0.5, -0.5,
			  -0.5, -0.5;
	} else if(ns == 3){

		A1 << 10, -1, -0.5,
			  -1, 10, -1,
			  -0.5, -1, 10;

		A2 << -0.5, -0.5, -0.5,
			  -0.5, -0.5, -0.5,
			  -0.5, -0.5, -0.5;
	} else {
		printf("Invalid choice ns. Valid choices are 2 & 3.\n");
		exit(1);
	}

	SpMat Q_upper(ns*nt, ns*nt);

	for(int i=0; i<nt; i++){
		int ii = i*ns;
		//printf("ii = %d, ii+ns = %d\n", ii,ii+ns);
		Q_d.block(ii,ii,ns,ns) = A1;
		if(i < nt-1){
			Q_d.block(ii,ii+ns, ns, ns) = A2;
			Q_d.block(ii+ns,ii, ns, ns) = A2.transpose();
		}
	}

	MatrixXd A3 = 0.5*(MatrixXd::Random(nb, ns*nt) - MatrixXd::Ones(nb, ns*nt));
	Q_d.block(ns*nt,0,nb,ns*nt) = A3;
	Q_d.block(0,ns*nt,ns*nt,nb) = A3.transpose();

	MatrixXd A4(nb,nb);

	if(nb == 1){
		A4 << n;
	} else if(nb ==2) {
		A4 << n, -0.5,
			  -0.5, n;
	} else if (nb == 3){
		A4 << n, 0, 0,
			  0, n, 0,
			  0, 0, n;
	} else {
		printf("invalid size nb, admissible choices: 1,2 & 3.!\n");
	}

	Q_d.block(ns*nt, ns*nt, nb, nb) = A4;


	std::cout << "Q :\n" << Q_d << std::endl;


	SpMat Q = Q_d.sparseView();


	return Q;



}

// nss : size of additional spatial field
SpMat gen_test_mat_base4(int ns, int nt, int nss, int nb){

    int n = ns*nt + nss + nb;

	//int nb = 2;
	MatrixXd Q_d(n,n);
	Q_d.setZero();

	MatrixXd A1(ns, ns);
	MatrixXd A2(ns, ns);

	if(ns == 2){
		A1 << n, -1,
			  -1, n;

		A2 << -0.5, -0.5,
			  -0.5, -0.5;
	} else if(ns == 3){

		A1 << n, -1, -0.5,
			  -1, n, -1,
			  -0.5, -1, n;

		A2 << -0.5, -0.5, -0.5,
			   0,   -0.5, -0.5,
			  -0.5,  0  , -0.5;
	} else {
		printf("Invalid choice ns. Valid choices are 2 & 3.\n");
		exit(1);
	}

	SpMat Q_upper(ns*nt, ns*nt);

	for(int i=0; i<nt; i++){
		int ii = i*ns;
		//printf("ii = %d, ii+ns = %d\n", ii,ii+ns);
		Q_d.block(ii,ii,ns,ns) = A1;
		if(i < nt-1){
			Q_d.block(ii,ii+ns, ns, ns) = A2;
			Q_d.block(ii+ns,ii, ns, ns) = A2.transpose();
		}
	}

	MatrixXd A3 = 0.5*(MatrixXd::Random(nss, ns*nt) - MatrixXd::Ones(nss, ns*nt));
	Q_d.block(ns*nt,0,nss,ns*nt) = A3;
	Q_d.block(0,ns*nt,ns*nt,nss) = A3.transpose();

    MatrixXd A33 = MatrixXd::Random(nss, nss);
    A33 = A33 * A33.transpose();
    for(int i=0; i<nss; i++){
        A33(i,i) = n+1;
    }

    Q_d.block(ns*nt, ns*nt, nss, nss) = A33;
    
    std::cout << "Qd : \n" << Q_d << std::endl;

    MatrixXd A4 = -0.1*MatrixXd::Ones(nb, ns*nt+nss);
    Q_d.block(ns*nt+nss,0, nb, ns*nt+nss) = A4;
    Q_d.block(0, ns*nt+nss, ns*nt+nss, nb) = A4.transpose();


	MatrixXd A44(nb,nb);

	if(nb == 1){
		A44 << n;
	} else if(nb ==2) {
		A44 << n, -0.5,
			  -0.5, n;
	} else if (nb == 3){
		A44 << n, -0.05, -0.02,
			  -0.05, n, -0.01,
			  -0.02, -0.01, n;
	} else {
		printf("invalid size nb, admissible choices: 1,2 & 3.!\n");
	}

	Q_d.block(ns*nt+nss, ns*nt+nss, nb, nb) = A44;
   


	std::cout << "Q :\n" << Q_d << std::endl;


	SpMat Q = Q_d.sparseView();


	return Q;



}


// nss : size of additional spatial field
SpMat gen_test_mat_base4_prior(int ns, int nt, int nss){

    int n = ns*nt + nss;

	//int nb = 2;
	MatrixXd Q_d(n,n);
	Q_d.setZero();

	MatrixXd A1(ns, ns);
	MatrixXd A2(ns, ns);

	if(ns == 2){
		A1 << n, -1,
			  -1, n;

		A2 << -0.5, -0.5,
			  -0.5, -0.5;
	} else if(ns == 3){

		A1 << n, -1, -0.5,
			  -1, n, -1,
			  -0.5, -1, n;

		A2 << -0.5, -0.5, -0.5,
			   0,   -0.5, -0.5,
			  -0.5,  0  , -0.5;
	} else {
		printf("Invalid choice ns. Valid choices are 2 & 3.\n");
		exit(1);
	}

	SpMat Q_upper(ns*nt, ns*nt);

	for(int i=0; i<nt; i++){
		int ii = i*ns;
		//printf("ii = %d, ii+ns = %d\n", ii,ii+ns);
		Q_d.block(ii,ii,ns,ns) = A1;
		if(i < nt-1){
			Q_d.block(ii,ii+ns, ns, ns) = A2;
			Q_d.block(ii+ns,ii, ns, ns) = A2.transpose();
		}
	}

    MatrixXd A33 = MatrixXd::Random(nss, nss);
    A33 = A33 * A33.transpose();
    for(int i=0; i<nss; i++){
        A33(i,i) = n+1;
    }

    Q_d.block(ns*nt, ns*nt, nss, nss) = A33;
    
	std::cout << "Q prior :\n" << Q_d << std::endl;


	SpMat Q = Q_d.sparseView();


	return Q;



}