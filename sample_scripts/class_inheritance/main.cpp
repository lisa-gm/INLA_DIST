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

#include <optional>
#include <armadillo>

#include "../../read_write_functions.cpp"

#include "PardisoSolver.h"
#include "RGFSolver.h"

//#define PRINT_MSG

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;

/* ===================================================================== */

/** spatial temporal model : SPDE discretisation. DEMF(1,2,1) model.*/
void construct_Q_spat_temp(Vector& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3, SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Qst){
    double exp_theta1 = exp(theta[1]);
    double exp_theta2 = exp(theta[2]);
    double exp_theta3 = exp(theta[3]);

    int nu = g1.rows()*M0.rows();

    // g^2 * fem$c0 + fem$g1
    SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

     // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
    SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

    // g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
    SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

    #ifdef PRINT_MSG
        std::cout << "theta u : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;

        /*std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
        std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl; 

        std::cout << "q1s : \n" << q1s.block(0,0,10,10) << std::endl;
        std::cout << "q2s : \n" << q2s.block(0,0,10,10) << std::endl;
        std::cout << "q3s : \n" << q3s.block(0,0,10,10) << std::endl;*/
    #endif

    // assemble overall precision matrix Q.st
    Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + 2*exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));
    //std::cout << "Qst : \n" << Qst->block(0,0,10,10) << std::endl;

}

  
/** construct precision matrix. Calls spatial, spatial-temporal, etc. appropriately. */
void construct_Q(int ns, int nt, int nb, Vector& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3, SpMat& M0, SpMat& M1, SpMat& M2, SpMat& AxTAx, SpMat& Q){
    double exp_theta0 = exp(theta[0]);
    //double exp_theta = exp(3);

    int nu = ns*nt;
    int n = nu + nb;

    SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
    /*std::cout << "Q_b " << std::endl;
    std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/
    //Q_b = 1e-5*Q_b.setIdentity();

    if(ns > 0){
        SpMat Qu(nu, nu);
        // TODO: find good way to assemble Qx

        if(nt > 1){
            construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, Qu);

        } else {    
            std::cout << "nt must be greater 1!" << std::endl;
            exit(1);
        }   

        //Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
        // construct Qx from Qs values, extend by zeros 
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

        Q =  Qx + exp_theta0 * AxTAx;

        #ifdef PRINT_MSG
            std::cout << "exp(theta0) : \n" << exp_theta0 << std::endl;
            std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
        #endif

        } else {
            std::cout << "ns must be greater than zero!" << std::endl;
            exit(1);
        }

    /*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
    std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

    #ifdef PRINT_MSG
        std::cout << "Q  dim : " << Q->rows() << " "  << Q->cols() << std::endl;
        std::cout << "Q : \n" << Q->block(0,0,10,10) << std::endl;
        std::cout << "theta : \n" << theta.transpose() << std::endl;

    #endif 

}

/** Assemble right-handside. Could compute Ax^T*y once, and only multiply with appropriate exp_theta.*/
void construct_b(Vector& theta, SpMat Ax, Vector y, Vector *rhs){
    double exp_theta = exp(theta[0]);
    //double exp_theta = exp(3);

    *rhs = exp_theta*Ax.transpose()*y;

}

// ================================ MAIN ===================================== //

int main(int argc, char* argv[])
{

	if(argc != 1 + 6){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;
        
        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;
        std::cerr << "[string:solver_type]        RGF or PARDISO " << std::endl;



        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    size_t no = atoi(argv[4]);

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nb);

    std::string base_path   = argv[5];

    std::string solver_type = argv[6];
    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("RGF") != 0){
		std::cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << std::endl;
		exit(1);
    }

    // dimension hyperparamter initialised to 1
    int dim_th = 1;

    /* ---------------- read in matrices ---------------- */

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
    SpMat AxTAx;
    Vector y;

    if(ns == 0 && nt == 0){
        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

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

        Ax = readCSC(Ax_file);
        AxTAx = Ax.transpose() * Ax;

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        std::cout << "spatial-temporal model." << std::endl;

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
        AxTAx = Ax.transpose() * Ax;

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
        exit(1);
    }


    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    y = read_matrix(y_file, no, 1);


    /* ----------------------- initialise random theta -------------------------------- */

    Vector theta(dim_th);
    Vector theta_prior(dim_th);

    int n;

    // initialise theta
    if(ns == 0 && nt == 0){
        n = nb;
        // Initial guess
        theta[0] = 3;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else if(ns > 0 && nt == 1){
        n = ns + nb;
        //theta << 1, -1, 1;
        theta << 1, -2, 2;
        theta_prior << 0, 0, 0;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else {
        n = ns*nt + nb;

        // =========== synthetic data set =============== //
        /*std::cout << "using SYNTHETIC DATASET" << std::endl;        
        theta_prior << 1.4, -5.9,  1,  3.7;  // here exact solution
        std::cout << "theta original     : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        //theta << 1.4, -5.9,  1,  3.7; 
        theta << 1, -3, 1, 3;
        //theta << 0.5, -1, 2, 2;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;*/

        // =========== temperature data set =============== //
        std::cout << "using TEMPERATURE DATASET" << std::endl; 
        theta_prior << -0.294769, -5.670050, -3.452297,  5.627084;       // EU only (solution from INLA)
        //theta_original << 5, -10, 2.5, 1;
        std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        theta << -0.2, -2, -2, 3;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;


    } // end initialise theta


    // ================================ initialise SOLVER ======================================= //
	Solver* solver;

	int threads = omp_get_max_threads();
	cout << "number of available threads = " << threads << endl;

	if(solver_type == "PARDISO"){
		solver = new PardisoSolver;
	} else if(solver_type == "RGF"){
		solver = new RGFSolver(ns);
	} else {
		cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << endl;
	}

    // ============================= construct Q & initialise =============================== //

	double log_det;

	SpMat Q(n,n);
    construct_Q(ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, AxTAx, Q);
    //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;

    Vector rhs(n);
    construct_b(theta, Ax, y, &rhs);
    //std::cout << "b : \n" << rhs.head(10) << std::endl;

    Vector sol(n);
    Vector inv_diag(n);

    // ============================= call SOLVER =============================== //

	solver->factorize(Q, log_det);
	std::cout << "log det : " << log_det << std::endl;

	solver->factorize_solve(Q, rhs, sol, log_det);
   	std::cout << "log det : " << log_det << std::endl;

   	solver->selected_inversion(Q, inv_diag);
    std::cout << "inv diag : " << inv_diag.head(10).transpose() << std::endl;




} // end main
