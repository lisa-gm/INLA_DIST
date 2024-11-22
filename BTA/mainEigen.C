#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SparseExtra>   // includes saveMarket

#include <armadillo>
#include "generate_testMat_selInv.cpp"
#include "../read_write_functions.cpp"
#include "helper_functions.h"

#include "BTA.H"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

#define PRINT_MSG
//#define RECORD_TIMES

#if 0
// ******************* 
#define SINGLE_PREC
typedef float T;
#define assign_T(val);
//typedef CPX T;
//#define assign_T(val) CPX(val, 0.0)
// *******************  
#else
// ******************* 
#define DOUBLE_PREC
typedef double T;
#define assign_T(val) val
// ******************* 
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
        //double t_kron = get_time(0.0);
		Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));
        //t_kron = get_time(t_kron);
        //printf("time KroneckerProductSparse: %f\n", t_kron);

		//std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
}

#if 0
void construct_Q(SpMat& Q, int ns, int nt, int nb, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

	double exp_theta0 = exp(theta[0]);
	int nu = ns*nt;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;

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

	} else {
        Q = Q_b + exp_theta0 * Ax.transpose() * Ax;
    }

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

#ifdef PRINT_MSG
			std::cout << "exp(theta0) : " << exp_theta0 << std::endl;
			//std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
			std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
			std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
			std::cout << "theta : \n" << theta.transpose() << std::endl;

#endif

}
#endif


void construct_Q(SpMat& Q, int ns, int nt, int nss, int nb, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

    int n = ns*nt + nss + nb;
    SpMat Qx(n,n);

    if(ns > 0 && nt == 1){
        printf("in construct Q. only spatial field. dummy. not implemented.\n");
    }
    else if(ns > 0 && nt > 1 && nss == 0){
        SpMat Qst(ns*nt, ns*nt);
        construct_Q_spat_temp(Qst, theta, c0, g1, g2, g3, M0, M1, M2);
		
		int nnz = Qst.nonZeros();
		Qx.reserve(nnz);

		for (int k=0; k<Qst.outerSize(); ++k){
		  for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
		  {
		    Qx.insert(it.row(),it.col()) = it.value();                 
		  }
        }

    } else if(ns > 0 && nt > 0 && nss > 0){
        SpMat Qst(ns*nt, ns*nt);
        std::cout << "theta:           " << theta.transpose() << std::endl;
        std::cout << "theta(seq(0,3)): " << theta(seq(0,3)).transpose() << std::endl;
        Vect theta_spat_temp = theta(seq(0,3));
        construct_Q_spat_temp(Qst, theta_spat_temp, c0, g1, g2, g3, M0, M1, M2);

        size_t nnz_Qst = Qst.nonZeros();
        Qx.reserve(nnz_Qst);

        for (int k=0; k<Qst.outerSize(); ++k){
            for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
            {
                Qx.insert(it.row(),it.col()) = it.value();                 
            }
        }

        SpMat Qs(nss, nss);
        // need to be careful about what theta values are accessed!! now dimension larger
        Vect theta_spat = theta(seq(3,5));
        construct_Q_spatial(Qs, theta_spat, c0, g1, g2);

        // insert entries of Qs
        for (int k=0; k<Qs.outerSize(); ++k){
            for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
            {
            Qx.insert(it.row()+ns*nt,it.col()+ns*nt) = it.value();                 
            }
        }

    } 

    //std::cout << "dim(Ax) = " << Ax.rows() << " " << Ax.cols() << ", dim(Qx) = " << Qx.rows() << " " << Qx.cols() << std::endl;

    for(int i=ns*nt+nss; i < n; i++){
        //printf("i = %d\n", i);
		// CAREFUL 1e-3 is arbitrary choice!!
		Qx.insert(i,i) = 1e-3;
	}	

    Qx.makeCompressed();

    double exp_theta0 = exp(theta[0]);
    Q = Qx + exp_theta0 * Ax.transpose()*Ax;

}

void call_EigenSolver(SpMat& Q, Vect& rhs, MatrixXd& Qinv, Vect& sol, double& log_det){

    int n = Q.rows();
    // true inv diag from Eigen
    //SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> solverQ;
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Q);

   if(solverQ.info()!=Success) {
     cout << "Oh: Very bad" << endl;
   }

    sol = solverQ.solve(rhs);

    SpMat L = solverQ.matrixL();

   // compute log sum by hand
   log_det = 0.0;
   for(int i = 0; i<n; i++){
        log_det += log(L.coeff(i,i));
   }
   log_det *=2.0;
   
    SpMat eye(n,n);
    eye.setIdentity();

    Qinv = solverQ.solve(eye);
    if(n < 25){
        MatrixXd inv_Q_dense = MatrixXd(Qinv.triangularView<Lower>());
        std::cout << "inv(Q)\n" << inv_Q_dense << std::endl;
    }


}


/* ===================================================================== */

int main(int argc, char* argv[])
{

size_t i; // iteration variable
std::string valueType;

#ifdef DOUBLE_PREC
    printf("Template T is double.\n");
    valueType = "double";
#endif

#ifdef SINGLE_PREC
    printf("Template T is float.\n");
    valueType = "single";
#endif

#if 0 // dummy example

    std::string solver_type = "BTA";

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

    int ns=3;
    int nss=0;
    int nt=4;
    int nb=2;
    int n = ns*nt + nb;

    SpMat Q = gen_test_mat_base3(ns, nt, nb);

    Vect rhs = Vect::Random(n);
    //rhs.setOnes(n);

#else

    if(argc != 1 + 7){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nss nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points of ADD. SPATIAL FIELD " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        std::cerr << "[string:solver_type]        BTA or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb = atoi(argv[4]);
    std::cout << "ns = " << ns << ", nt = " << nt << ", nss = " << nss << ", nb = " << nb << std::endl;
    //size_t no = atoi(argv[4]);
    std::string no_s = argv[5];
    // to be filled later
    size_t no;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 
    // set nt = 0 for regression case
    if(ns == 0){
        nt = 0;
    }

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nss_s = std::to_string(nss);
    std::string nb_s = std::to_string(nb);
    //std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nss + nb);

    std::string base_path = argv[6];    

    std::string solver_type = argv[7];
    // check if solver type is neither PARDISO nor BTA :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("BTA") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nBTA" << std::endl;
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
        std::string B_file        =  base_path + "/A_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        std::cout << "total number of observations : " << no << std::endl;
      
        B = read_matrix(B_file, no, nb);
        Ax = B.sparseView();

        // std::cout << "y : \n"  << y << std::endl;    
        // std::cout << "B : \n" << B << std::endl;
        std::cout << "t(B)*B = " << B.transpose() * B << std::endl;

    } else if(ns > 0 && nt == 1 && nss == 0){

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

    } else if(ns > 0 && nt > 1){

        if(nss == 0){
            dim_th = 4;
        } else {
            dim_th = 6;
        }

        printf("spatial-temporal model");
        if(nss > 0){
            printf(" with add. spatial field");
        }
        printf(".\n");
    

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

        //std::cout << "total number of observations : " << no << std::endl;
        std::cout << "read in all matrices." << std::endl;

    } else {
        std::cout << "invalid parameters : ns nt nss !!" << std::endl;
        exit(1);
    }

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    y = read_matrix(y_file, no, 1);

    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);
    Vect theta_prior(dim_th);
    
    //printf("dim(theta) = %ld, ns = %ld, nt = %ld\n", dim_th, ns, nt);

    if(ns == 0 && nt == 0){
        theta << 2;
        std::cout << "theta : " << theta.transpose() << std::endl;
    } else if(ns > 0 && nt == 1){
	    theta << -1.5,-5,-2;
	    //theta.print();
    } else if(nt > 1 && ns > 0 && nss == 0){

        //theta << 5, -10, 2.5, 1;
        theta << 1.386294, -5.882541,  1.039721,  3.688879; // equals 4,0,0,0 in param scale     
        //theta << -1.998039, -9.828957,  1.981187,  8.288427;   
        //std::cout << "theta : " << theta.transpose() << std::endl;
        // final theta for temperature example
        //theta << -1.28985974,  3.93801236, -8.38111330, -4.48324016;
	    //theta = {3, -5, 1, 2};
  	} else {
        theta << 1.386796, -4.434666, 0.6711493, 1.632289, -5.058083, 2.664039;
  	}

    std::cout << "theta: " << theta.transpose() << std::endl;

    printf("# threads: %d\n", omp_get_max_threads());

#if 0

    int nx = ns*nt + nss;

    SpMat Qx(nx, nx);

    double t_Qx_factorise;
    BTA<T> *solver_Qx;
    solver_Qx = new BTA<T>(ns, nt, nss);

    double log_det_Qx;

    for(int c=0; c<1; c++){
        //theta = theta + Vect::Random(theta.size());
        std::cout << "\niter = " << c << ". Constructing precision matrix Qx. theta : " << theta.transpose() << std::endl;

        if(nss == 0){
            construct_Q_spat_temp(Qx, theta, c0, g1, g2, g3, M0, M1, M2);
        } else {
            SpMat Qst(ns*nt, ns*nt);
            std::cout << "theta:           " << theta.transpose() << std::endl;
            std::cout << "theta(seq(0,3)): " << theta(seq(0,3)).transpose() << std::endl;
            Vect theta_spat_temp = theta(seq(0,3));
            construct_Q_spat_temp(Qst, theta_spat_temp, c0, g1, g2, g3, M0, M1, M2);

            size_t nnz_Qst = Qst.nonZeros();
            Qx.reserve(nnz_Qst);

            for (int k=0; k<Qst.outerSize(); ++k){
                for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
                {
                    Qx.insert(it.row(),it.col()) = it.value();                 
                }
            }

            SpMat Qs(nss, nss);
            // need to be careful about what theta values are accessed!! now dimension larger
            Vect theta_spat = theta(seq(3,5));
            construct_Q_spatial(Qs, theta_spat, c0, g1, g2);

            // insert entries of Qs
            for (int k=0; k<Qs.outerSize(); ++k){
                for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
                {
                Qx.insert(it.row()+ns*nt,it.col()+ns*nt) = it.value();                 
                }
            }

        }

        //SpMat epsId(nx,nx);
        //epsId.setIdentity();
        //epsId = 1e-4*epsId;
        //Qx = Qx + epsId;

        // only take lower triangular part of A
        SpMat Qx_lower = Qx.triangularView<Lower>(); 
        size_t nnz_Qx  = Qx_lower.nonZeros();

        /*std::string Qx_fileName = "Qx_" + to_string(n) + ".txt";
        write_sym_CSC_matrix(Qx_fileName, Qx_lower);
        exit(1);*/

#if 0
        SpMat Qx_first = Qx.block(0,0,ns,ns);
        std::cout << "nnz(Q) = " << Qx_first.nonZeros() << std::endl;
        //write_sym_CSC_matrix(Qx_firstBlock_file, Qx_first);
        std::string Qx_firstBlock_file = "Qst_firstBlock_" + to_string(ns) + "_" + to_string(ns) + ".mtx";
        Eigen::saveMarket(Qx_first, Qx_firstBlock_file);
        exit(1);
#endif

        Qx_lower.makeCompressed();

        size_t* ia_Qx; 
        size_t* ja_Qx;
        T* a_Qx; 

        // allocate memory
        ia_Qx = new long unsigned int [nx+1];
        ja_Qx = new long unsigned int [nnz_Qx];
        a_Qx  = new double [nnz_Qx];

        for (i = 0; i < nx+1; ++i){
            ia_Qx[i] = Qx_lower.outerIndexPtr()[i]; 
        }  

        for (i = 0; i < nnz_Qx; ++i){
            ja_Qx[i] = Qx_lower.innerIndexPtr()[i];
        }  

        for (i = 0; i < nnz_Qx; ++i){
            a_Qx[i] = Qx_lower.valuePtr()[i];
        }

        t_Qx_factorise = get_time(0.0);
        //solver->solve_equation(GR);
        double flops_Qx_factorize = solver_Qx->factorize_noCopyHost(ia_Qx, ja_Qx, a_Qx, log_det_Qx);
        printf("no Cpy to Host logdet       : %f\n", log_det_Qx);

        double flops_Qx_factorize = solver_Qx->factorize(ia_Qx, ja_Qx, a_Qx);
        log_det_Qx = solver_Qx->logDet(ia_Qx, ja_Qx, a_Qx);

        t_Qx_factorise = get_time(t_Qx_factorise);

        printf("logdet       : %f\n", log_det_Qx);
        printf("time chol(Qx): %lg\n",t_Qx_factorise);

        T *invDiag_Qx = new T[nx];
        solver_Qx->BTAdiag(ia_Qx, ja_Qx, a_Qx, invDiag_Qx);

        Vect invDiag_Qx_vec(nx);
        for(int i=0; i<nx; i++){
            invDiag_Qx_vec[i] = invDiag_Qx[i];
        }

        //printf("norm(invDiag_Qx)      : %f\n", invDiag_Qx_vec.norm());
        
        T *invQa = new T[Qx_lower.nonZeros()];
        solver_Qx->BTAselInv(ia_Qx, ja_Qx, a_Qx, invQa);

        SpMat invQx_lower = Eigen::Map<Eigen::SparseMatrix<double> >(nx,nx,Qx_lower.nonZeros(),Qx_lower.outerIndexPtr(), // read-write
                               Qx_lower.innerIndexPtr(),invQa);

        // TODO: more efficient way to do this?
        //SpMat invQx_new = invQ_new_lower.selfadjointView<Lower>();

        //printf("norm(invDiag_full_Qx) : %f\n", invQx_lower.diagonal().norm());

        //std::cout << "norm(diag(invQ_new) - diag(invDiag)) = " << (invQx_lower.diagonal() - invDiag_Qx_vec).norm() << std::endl;

        //Vect invDiag_Qx(nx);
        //solver_Qx->BTAdiag(ia_Qx, ja_Qx, a_Qx, invDiag_Qx);

        delete[] ia_Qx;
        delete[] ja_Qx;
        delete[] a_Qx;

        delete[] invDiag_Qx;
        delete[] invQa;

    }

    delete solver_Qx;

#endif // end construct Qx


#endif // end dummy example or reading in matrices


#if 1
    int n = ns*nt + nss + nb;
    size_t nnz;

    SpMat Q(n,n);
    SpMat Q_lower(n,n);
    Vect rhs(n);

    double exp_theta = exp(theta[0]);
    rhs = exp_theta*Ax.transpose()*y;
    std::cout << "\nConstructing precision matrix Qxy. " << std::endl; 

    double t_constructQ = - omp_get_wtime();
    construct_Q(Q, ns, nt, nss, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax);        
    t_constructQ += omp_get_wtime();
    printf("time spent construct Q :  %f\n", t_constructQ);
    std::cout << "Q : \n" << Q.block(0,0,6,6) << std::endl;
    
    // =========================================================================== //
    std::cout << "Converting Eigen Matrices to CSR format. " << std::endl;

    // only take lower triangular part of A
    Q_lower = Q.triangularView<Lower>(); 
    nnz    = Q_lower.nonZeros();
    printf("nnz(Q_lower) = %ld\n", nnz);

    size_t* ia; 
    size_t* ja;
    T* a; 
    T *b;
    T *x;
    T *x2;

    T* invQa;

    b        = new T[n];
    x        = new T[n];
    x2       = new T[n];

    // allocate memory
    ia = new long unsigned int [n+1];
    ja = new long unsigned int [nnz];
    a  = new T [nnz];

    Q_lower.makeCompressed();

    for (i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    // cast as double or f
    for (i = 0; i < nnz; ++i){
        a[i] = (T) Q_lower.valuePtr()[i];
    }

    for(i = 0; i < n; i++){
        b[i] = (T) rhs[i];
    }

    double t_factorise;
    double t_solve;

    // *** pin GPU & combine with appropriate cores *** //
    int GPU_rank = 0;
    cudaSetDevice(GPU_rank);
    int numa_node = topo_get_numNode(GPU_rank);

    int* hwt = NULL;
    int hwt_count = read_numa_threads(numa_node, &hwt);
    pin_hwthreads(1, &hwt[omp_get_thread_num()]);
    std::cout<<"Pinning GPU & hw threads. GPU rank : "<<GPU_rank <<", tid: "<<omp_get_thread_num()<<", NUMA domain ID: "<<numa_node;
    std::cout<<", hwthreads: " << hwt[omp_get_thread_num()] << std::endl;
    // *********************************************** //

    printf("call BTA constructor. nt = %ld\n", nt); 
    BTA<T> *solver;
    solver = new BTA<T>(ns, nt, nss+nb, GPU_rank);

    int m = 3;
    Vect t_factorize_vec(m-1);
    T log_det;

    double t_firstStageFactor;
    double t_secondStageForwardPass;
    double t_secondStageBackwardPass1;
    double t_firstSecondStage;
    double t_secondStageBackwardPass2;

    double flops_factorize;

#ifdef RECORD_TIMES
    std::string log_file_name = "log_file_factorize_solve_" + solver_type + "_MAGMAnative_ns" + std::to_string(ns) + "_nt" + std::to_string(nt) + "_nb" + std::to_string(nb) + "_" + std::to_string(omp_get_max_threads()) + ".txt";
    std::ofstream log_file(log_file_name);
    log_file << "iter t_firstStageFactor t_secondStageForwardPass t_secondStageBackwardPass t_total_solveCPU t_firstSecondStage t_SecondStageBackPass t_total_solveHybrid" << std::endl;
    log_file.close();
#endif

    for(int iter=0; iter<m; iter++){
        printf("\niter = %d\n", iter);

        // t_factorise = get_time(0.0);
        // flops_factorize = solver->factorize_noCopyHost(ia, ja, a, log_det);
        // t_factorise = get_time(t_factorise);
        // printf("log det noCopyHost: %f\n", log_det);
        // printf("time factorize noCopyHost: %f\n", t_factorise);

        t_factorise = get_time(0.0);
        //solver->solve_equation(GR);
        flops_factorize = solver->factorize(ia, ja, a, t_firstStageFactor);
        //log_det = solver->logDet(ia, ja, a);
        t_factorise = get_time(t_factorise);
        //printf("time factorize:             %f\n", t_factorise);

        t_solve = get_time(0.0); 
        double flops_solve = solver->solve(ia, ja, a, x, b, 1, t_secondStageForwardPass, t_secondStageBackwardPass1);
        t_solve = get_time(t_solve);

        double res_norm_normalized = solver->residualNormNormalized(x, b);
        if(res_norm_normalized > 1e-10){
            printf("\nResidual norm.:           %e\n", solver->residualNorm(x, b));
            printf("Residual norm normalized: %e\n", res_norm_normalized); 
            printf("Residual norm too large. Exiting. \n");
            exit(1);
        }

        // Vect sol(n);
        // for(int i=0; i<n; i++){
        //     sol[i] = x[i];
        // }

        double t_factoriseSolve = get_time(0.0);
        //solver->solve_equation(GR);
        flops_factorize = solver->factorizeSolve(ia, ja, a, x2, b, 1, t_firstStageFactor, t_secondStageBackwardPass1);
        //log_det = solver->logDet(ia, ja, a);
        t_factoriseSolve = get_time(t_factoriseSolve);

        double res_norm_normalized2 = solver->residualNormNormalized(x2, b);
        if(res_norm_normalized2 > 1e-10){
            printf("\nResidual norm.:           %e\n", solver->residualNorm(x2, b));
            printf("Residual norm normalized: %e\n", res_norm_normalized2); 
            printf("Residual norm too large. Exiting. \n");
            exit(1);
        }

        // Vect sol2(n);
        // for(int i=0; i<n; i++){
        //     sol2[i] = x2[i];
        // }

        //printf("time chol(Q): %lg\n",t_factorise);
        printf("time factorize:            %f\n",t_factorise);
        printf("time solve:                %f\n", t_solve);
        printf("time factorizeSolve:       %f\n", t_factoriseSolve);

#if 0
        T *x_new = new T[n];

        t_factorise = get_time(0.0);
        flops_factorize = solver->factorizeSolve(ia, ja, a, x_new, b, 1, t_firstSecondStage, t_secondStageBackwardPass2);
        t_factorise = get_time(t_factorise);
        log_det = solver->logDet(ia, ja, a);

        Vect x_new_vec(n);
        Vect x_vec(n);

        for(int i=0; i<n; i++){
            x_new_vec[i] = x_new[i];
            x_vec[i]     = x[i];
        }
        std::cout << "norm(x-x_new) = " << (x_vec - x_new_vec).norm() << std::endl;

        printf("log det factorizeSolve   : %f\n", log_det);
        printf("time factorizeSolve      : %f\n", t_factorise);
#endif

#ifdef RECORD_TIMES
        // ========================================================================== 
        //iter t_firstStageFactor t_secondStageForwardPass t_secondStageBackwardPass t_total_solveCPU t_firstSecondStage t_SecondStageBackPass t_total_solveHybrid
        std::ofstream log_file(log_file_name, std::ios_base::app | std::ios_base::out);
        log_file << iter << " " << t_firstStageFactor << " " << t_secondStageForwardPass << " " << t_secondStageBackwardPass1 << " " << t_firstStageFactor + t_secondStageForwardPass + t_secondStageBackwardPass1 << " ";
        log_file << t_firstSecondStage << " " << t_secondStageBackwardPass2 << " " << t_firstSecondStage+t_secondStageBackwardPass2 << std::endl;
        log_file.close(); 
        // ========================================================================== //
#endif

  
#if 1

        T *invDiag;
        invDiag  = new T[n];

        double t_invDiag;
        t_invDiag = get_time(0.0);
        double flops_invDiag = solver->BTAdiag(ia, ja, a, invDiag);
        t_invDiag = get_time(t_invDiag);
        double log_detBTAdiag = solver->logDet(ia, ja, a);

        if(n < 25){
            printf("\nBTAinvDiag: ");
            for(i=0; i<n; i++){
                printf(" %f", invDiag[i]);
            }
            printf("\n");
        }

        //printf("computed BTAdiag\n");
        //printf("flops inv:      %f\n", flops_invDiag);

        //solver->init_supernode()
        invQa = new T[nnz];
        //printf("before BTAselinv\n");
        double flops_invQa = solver->BTAselInv(ia, ja, a, invQa);

        //printf("before logDetselInv\n");
        // T log_detBTAselInv = solver->logDet(ia, ja, a);

        // double* invQa_d = new double[nnz];
        // for(int i=0; i<nnz; i++){
        //     invQa_d[i] = (double) invQa[i];
        // }

        // if(n < 25){
        //     printf("invQa : ");
        //     for(int i=0; i<nnz; i++){
        //         printf(" %f", invQa[i]);
        //     }
        //     printf("\n");
        // }

        //printf("computed BTAselInv\n");

        // SpMat invQ_new_lower = Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,Q_lower.outerIndexPtr(), // read-write
        //                         Q_lower.innerIndexPtr(),invQa_d);


        // if(n < 25){
        //     std::cout << "invQ_new:\n" << MatrixXd(invQ_new_lower) << std::endl;
        // }

    // TODO: more efficient way to do this?
        // SpMat invQ_new = invQ_new_lower.selfadjointView<Lower>();

        // Vect invDiag_vec(n);
        // for(int i=0; i<n; i++){
        //     invDiag_vec[i] = invDiag[i];
        // }

        // std::cout << "norm(diag(invQ_new)) = " << invQ_new.diagonal().norm() << std::endl;
        // std::cout << "norm(invDiag))       = " << invDiag_vec.norm() << std::endl;    
        // std::cout << "norm(diag(invQ_new) - diag(invDiag)) = " << (invQ_new.diagonal() - invDiag_vec).norm() << std::endl;
        
        // check with Eigen solver for small test cases
        // if(n < 5000){
        //     MatrixXd Qinv_eigen(n,n);
        //     Vect sol_eigen(n);
        //     double log_det_eigen;

        //     // call Eigen solver
        //     call_EigenSolver(Q, rhs, Qinv_eigen, sol_eigen, log_det_eigen);
        //     std::cout << "norm(x_BTA - x_Eigen)             = " << (sol_eigen - sol).norm() << std::endl;
        //     std::cout << "norm(log_det_BTA - log_det_Eigen) = " << (log_det_eigen - log_det) << std::endl;
        //     std::cout << "norm(diag(invQ_new) - diag(invEigen)) = " << (invQ_new.diagonal() - Qinv_eigen.diagonal()).norm() << std::endl;

        // }

    } // end loop over m

#if 0
    // create file with inv Diag vector
    std::string invDiag_file_name = "invDiag_BTA_" + valueType + "_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
  	std::ofstream invDiag_file(invDiag_file_name,    std::ios::out | std::ios::trunc);

	for (i = 0; i < n; i++) {
		invDiag_file << invDiag[i] << std::endl;
		// sol_x_file << x[i] << std::endl; 
	}
    invDiag_file.close();
#endif 
    /*
    t_invBlks = get_time(0.0);
    double flops_invBlks = solver->BTAinvBlks(ia, ja, a, invBlks);
    t_invBlks= get_time(t_invBlks);
    std::cout << "diff Log Dets : " << log_detBTAdiag - log_detBTAselInv << std::endl;

    printf("BTA factorise time: %lg\n",t_factorise);
    printf("BTA solve     time: %lg\n",t_solve);
    printf("BTA inv Diag  time: %lg\n",t_invDiag);
    printf("BTA inv Blks  time: %lg\n",t_invBlks);
    */

    // now assemble invBlks to correct sparse matrix -> column major -> iterate through columns
    // careful with block structure, need to be alternating betwen diagonal & off diagonal dense blocks
    //delete[] invDiag;

#endif

  
    // free memory
    delete solver;

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] invQa;

    delete[] x;
    delete[] b;

    //} // end if

    #endif
        
    return 0;


  }

