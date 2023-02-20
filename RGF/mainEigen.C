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
#include <unsupported/Eigen/SparseExtra>

#include <armadillo>
#include "generate_testMat_selInv.cpp"
#include "../read_write_functions.cpp"

#include "RGF.H"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

#define PRINT_MSG

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

void construct_Q_spatial(SpMat& Qs, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2){

	// Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
	// return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
	double exp_theta1 = exp(theta[1]);
	double exp_theta2 = exp(theta[2]);
	//double exp_theta1 = -3;
	//double exp_theta2 = 1.5;

	Qs = pow(exp_theta1,2)*(pow(exp_theta2, 4) * c0 + 2*pow(exp_theta2,2) * g1 + g2);

	#ifdef PRINT_MSG
		/*std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
		std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;
		std::cout << "c0 : \n" << c0.block(0,0,10,10) << std::endl;
        std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;*/
    #endif

	// extract triplet indices and insert into Qx
} 


void construct_Q_spat_temp(SpMat& Qst, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2){

	//std::cout << "theta : " << theta.transpose() << std::endl;

	double exp_theta1 = exp(theta[1]);
	double exp_theta2 = exp(theta[2]);
	double exp_theta3 = exp(theta[3]);

	/*double exp_theta1 = exp(-5.594859);
	double exp_theta2 = exp(1.039721);
	double exp_theta3 = exp(3.688879);*/

	//std::cout << "exp(theta) : " << exp(theta[0]) << " " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << " " << std::endl;	

	// g^2 * fem$c0 + fem$g1
	SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

	 // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
		SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

		// g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
		SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

		#ifdef PRINT_MSG
			/*std::cout << "theta u : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;
		std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
		std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;
		std::cout << "q1s : \n" << q1s.block(0,0,10,10) << std::endl;
        std::cout << "q2s : \n" << q2s.block(0,0,10,10) << std::endl;
        std::cout << "q3s : \n" << q3s.block(0,0,10,10) << std::endl;*/
		#endif

		// assemble overall precision matrix Q.st
		Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + 2*exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

		//std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
}


void construct_Q(SpMat& Q, int ns, int nt, int nb, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

	double exp_theta0 = exp(theta[0]);
	int nu = ns*nt;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

	if(ns > 0){
		SpMat Qu(nu, nu);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(Qu, theta, c0, g1, g2, g3, M0, M1, M2);
		} else {	
			construct_Q_spatial(Qu, theta, c0, g1, g2);
		}	

		//Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
		// construct Qx from Qs values, extend by zeros 
		size_t n = ns*nt + nb;
		SpMat Qx(n,n);         // default is column major			

		int nnz = Qu.nonZeros();
		Qx.reserve(nnz);

		for (int k=0; k<Qu.outerSize(); ++k)
		  for (SparseMatrix<double>::InnerIterator it(Qu,k); it; ++it)
		  {
		    Qx.insert(it.row(),it.col()) = it.value();                 
		  }

		//Qs.makeCompressed();
		//SpMat Qx = Map<SparseMatrix<double> >(ns+nb,ns+nb,nnz,Qs.outerIndexPtr(), // read-write
        //                   Qs.innerIndexPtr(),Qs.valuePtr());

		for(int i=nu; i<(n); i++){
			Qx.coeffRef(i,i) = 1e-5;
		}

		Qx.makeCompressed();

		#ifdef PRINT_MSG
			//std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
			//std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;
		#endif

		Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

		#ifdef PRINT_MSG
			std::cout << "exp(theta0) : " << exp_theta0 << std::endl;
			std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;

			std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
			//std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
			std::cout << "theta : \n" << theta.transpose() << std::endl;

		#endif
	}

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}


/* ===================================================================== */

int main(int argc, char* argv[])
{

#if 0

    /*
    int ns=1;
    int nt=6;
    int nb=1;
    int no=0;

    int n = ns*nt+nb;

    //SpMat Q = gen_test_mat_base1();
    SpMat Q = gen_test_mat_base2();
    std::cout << "Q: \n" << Q << std::endl;
    */

    int ns=2;
    int nt=3;
    int nb=1;
    int n = ns*nt + nb;

    SpMat Q = gen_test_mat_base3(ns, nt, nb);

    Vect rhs(n);
    rhs.setOnes(n);



#else

    if(argc != 1 + 6){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        std::cerr << "[string:solver_type]        RGF or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    std::cout << "ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    //size_t no = atoi(argv[4]);
    std::string no_s = argv[4];
    // to be filled later
    size_t no;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    //std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nb);

    std::string base_path = argv[5];    

    std::string solver_type = argv[6];
    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("RGF") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << std::endl;
        exit(1);
    }

    /* ---------------- read in matrices ---------------- */

    // dimension hyperparamter vector
    int dim_th;

    // spatial component
    SpMat c0; 
    SpMat g1; 
    SpMat g2;

    // spatial-temporal parts
    SpMat g3;
    SpMat M0;
    SpMat M1;
    SpMat M2;

    // data component / fixed effects
    MatrixXd B;
    SpMat Ax; 
    Vect y;

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        std::cout << "total number of observations : " << no << std::endl;
      
        B = read_matrix(B_file, no, nb);

        // std::cout << "y : \n"  << y << std::endl;    
        // std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

        std::cout << "spatial model." << std::endl;

        dim_th = 3;

        // check spatial FEM matrices
        std::string c0_file       =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file       =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file       =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);

        // doesnt require no to be read, can read no from Ax
        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();
        std::cout << "total number of observations : " << no << std::endl;


        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;
        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        std::cout << "spatial-temporal model. Reading in matrices." << std::endl;

        dim_th = 4;

        // files to construct Q.u depending on HYPERPARAMETERS theta
        std::string c0_file      =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file      =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file      =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);
        std::string g3_file      =  base_path + "/g3_" + ns_s + ".dat";
        file_exists(g3_file);

        std::string M0_file      =  base_path + "/M0_" + nt_s + ".dat";
        file_exists(M0_file);
        std::string M1_file      =  base_path + "/M1_" + nt_s + ".dat";
        file_exists(M1_file);
        std::string M2_file      =  base_path + "/M2_" + nt_s + ".dat";
        file_exists(M2_file);  

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);
        g3 = read_sym_CSC(g3_file);

        M0 = read_sym_CSC(M0_file);
        //arma::mat(M0).submat(0,0,nt-1,nt-1).print();
        M1 = read_sym_CSC(M1_file);
        //arma::mat(M1).submat(0,0,nt-1,nt-1).print();
        M2 = read_sym_CSC(M2_file);
        //arma::mat(M2).submat(0,0,nt-1,nt-1).print();

        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();
        std::cout << "total number of observations : " << no << std::endl;

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
        exit(1);
    }

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);


    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);
    Vect theta_prior(dim_th);

	if(nt == 1){
	    theta << -1.5,-5,-2;
	    //theta.print();
  	} else {
	    theta << 5, -10, 2.5, 1;
	    //theta = {3, -5, 1, 2};
	    //theta.print();
  	}

  	std::cout << "Constructing precision matrix Q. " << std::endl;

    size_t n = ns*nt + nb;
  	SpMat Q(n,n);
  	construct_Q(Q, ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax);
    //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;

    Vect rhs(n);
    double exp_theta = exp(theta[0]);
	rhs = exp_theta*Ax.transpose()*y;

#endif


#if 0

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    size_t nnz = Q_lower.nonZeros();

    //std::cout << "Q_lower before writing :\n" << Q_lower << std::endl;
    std::string Q_file = "Q_lower_" + to_string(Q_lower.rows()) + "_" + to_string(Q_lower.cols()) + ".mtx";
    // saving Q to file in mtx format
    std::cout << "writing Q_lower to file in mtx format." << std::endl;
    //Eigen::saveMarket(Q, Q_file);
    Eigen::saveMarket(Q_lower, Q_file);

    /*
    // read in mtx matrix file
    SpMat Q_test(Q_lower.rows(), Q_lower.cols());
    Eigen::loadMarket(Q_test, Q_file);
    //std::cout << "Q_lower after reloading :\n" << Q_test << std::endl;
    std::cout << "norm(Q_test - Q_lower) = " << (Q_test - Q_lower).norm() << std::endl;
    */

#endif


#if 1

    size_t i; // iteration variable

	// =========================================================================== //
	std::cout << "Converting Eigen Matrices to CSR format. " << std::endl;

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    size_t nnz = Q_lower.nonZeros();

    size_t* ia; 
    size_t* ja;
    T* a; 
    T *b;
  	T *x;
    T *invDiag;

  	b        = new T[n];
  	x        = new T[n];
    invDiag = new T[n];

    // allocate memory
    ia = new long unsigned int [n+1];
    ja = new long unsigned int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    double t_factorise;
	double t_solve;
    double t_inv;
	RGF<T> *solver;

	time_t rawtime;
	struct tm *timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf ("The current date/time is: %s\n",asctime(timeinfo));

	solver = new RGF<T>(ns, nt, nb);

	t_factorise = get_time(0.0);
	//solver->solve_equation(GR);
	double flops_factorize = solver->factorize(ia, ja, a);
	t_factorise = get_time(t_factorise);

	double log_det = solver->logDet(ia, ja, a);
	printf("logdet: %f\n", log_det);

  	// assign b to correct format
  	for (int i = 0; i < n; i++){
	    b[i] = rhs[i];
	    //printf("%f\n", b[i]);
  	}

  	t_solve = get_time(0.0); 
    double flops_solve = solver->solve(ia, ja, a, x, b, 1);
  	t_solve = get_time(t_solve);
  	printf("flops solve:     %f\n", flops_solve);

	printf("Residual norm.           : %e\n", solver->residualNorm(x, b));
	printf("Residual norm normalized : %e\n", solver->residualNormNormalized(x, b));

  	// create file with solution vector
  	/*
    std::string sol_x_file_name = "x_sol_RGF_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
  	std::ofstream sol_x_file(sol_x_file_name,    std::ios::out | std::ios::trunc);

	for (i = 0; i < n; i++) {
		sol_x_file << x[i] << std::endl;
		// sol_x_file << x[i] << std::endl; 
	}

  sol_x_file.close();
  */


    // true inv diag from Eigen
    //SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> solverQ;
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Q);

   if(solverQ.info()!=Success) {
     cout << "Oh: Very bad" << endl;
   }

   SpMat L = solverQ.matrixL();
   if(n < 20){
        std:cout << "L: \n" << MatrixXd(L) << std::endl;
    }

   SpMat eye(n,n);
   eye.setIdentity();

   SpMat inv_Q = solverQ.solve(eye);

   if(n < 20){
        std::cout << "inv(Q):\n" << MatrixXd(inv_Q) << std::endl;
    }

    t_inv = get_time(0.0);
    double flops_invDiag = solver->RGFdiag(ia, ja, a, invDiag);
    t_inv = get_time(t_inv);
    printf("flops inv:      %f\n", flops_invDiag);



    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve     time: %lg\n",t_solve);
    printf("RGF inv Diag  time: %lg\n",t_inv);

    // print/write diag 
    /*
    string sel_inv_file_name = "sel_inv_RGF_ns"+to_string(ns)+"_nt"+to_string(nt)+"_nb"+ to_string(nb) + "_no" + to_string(no) +".dat";
    cout << sel_inv_file_name << endl;
    ofstream sel_inv_file(sel_inv_file_name,    ios::out | ::ios::trunc);

    for (int i = 0; i < n; i++){
        sel_inv_file << invDiag[i] << endl;
    }

    sel_inv_file.close();
    cout << "after writing file " << endl;
    */

    Vect invDiag_vec(n);
    // assign b to correct format
    for (int i = 0; i < n; i++){
        invDiag_vec[i] = invDiag[i];
        //printf("%f\n", b[i]);
    }

   //cout << "Q:\n" << Q << endl;

    if(n < 20){
        cout << "\ninvDiag BTA       : " << invDiag_vec.transpose() << std::endl;
        cout << "Eigen diag(inv_Q) : " << inv_Q.diagonal().transpose() << endl;
    } else {
        cout << "\ninvDiag BTA[1:20]       : " << invDiag_vec.head(20).transpose() << std::endl;
        cout << "Eigen diag(inv_Q)[1:20] : " << inv_Q.diagonal().head(20).transpose() << endl;
    }

  
  // free memory
  delete solver;
  delete[] ia;
  delete[] ja;
  delete[] a;
  delete[] b;
  delete[] x;

  #endif

    
  return 0;


  }