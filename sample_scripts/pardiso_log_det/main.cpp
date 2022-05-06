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

#include <armadillo>

#include "read_write_functions.cpp"
#include "PardisoSolver.h"

//#include <likwid.h>


using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

//#define PRINT_MSG

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
			std::cout << "theta : " << theta.transpose() << std::endl;

		#endif
	}

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}


void update_mean_constr(MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W){

    // now that we have V = Q^-1*t(Dxy), compute W = Dxy*V
    W = D*V;
    std::cout << "W = " << W << std::endl;
    // U = W^-1*V^T, W is spd and small
    // TODO: use proper solver ...
    MatrixXd U = W.inverse()*V.transpose();
    //std::cout << "U = " << U << std::endl;

    Vect c = D*sol - e;
    sol = sol - U.transpose()*c;

    std::cout << "sum(sol) = " << (D*sol).sum() << std::endl;

}


/* ===================================================================== */

int main(int argc, char* argv[])
{

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

    std::cout << "========= New PARDISO main call ===========" << std::endl;

    size_t i; // iteration variable

    //std::cout << "reading in example. " << std::endl;

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
        //std::cout << "total number of observations : " << no << std::endl;

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
	    theta << 4.000000, -1.344954, -2.960279, -2.613706;
	    //theta = {3, -5, 1, 2};
	    //theta.print();
  	}

  	//std::cout << "Constructing precision matrix Q. " << std::endl;
    SpMat Qst(ns*nt, ns*nt);
    construct_Q_spat_temp(Qst, theta, c0, g1, g2, g3, M0, M1, M2);

    size_t n = ns*nt + nb;
  	SpMat Q(n,n);
  	construct_Q(Q, ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax);
    //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;

    Vect rhs(n);
    double exp_theta = exp(theta[0]);
	rhs = exp_theta*Ax.transpose()*y;

    Vect sol(n);
    Vect sol_constr(n);

    // =========================================================================== //
    // initialize solvers

    int MPI_rank = 0;

    Solver* solverQst;
    Solver* solverQ;

    solverQst = new PardisoSolver(MPI_rank);
    solverQ   = new PardisoSolver(MPI_rank);

    bool constr = true;
    int num_constr = 1;

    Vect e = Vect::Zero(num_constr);
    MatrixXd Dx(num_constr, ns*nt);
    Dx.row(0) << MatrixXd::Ones(num_constr,ns*nt);
    MatrixXd V1(ns*nt, num_constr);

    double log_det_Qst;
    solverQst->factorize_w_constr(Qst, constr, Dx, log_det_Qst, V1);

    MatrixXd W1(num_constr, num_constr);
    // here sol is equal to zero?
    Vect mu_x = Vect::Zero(ns*nt);
    update_mean_constr(Dx, e, mu_x, V1, W1);

    double log_det_Q;
    MatrixXd Dxy(num_constr,n);
    Dxy.row(0) << Dx , MatrixXd::Zero(num_constr,nb);

    MatrixXd V2(n,num_constr);
    solverQ->factorize_solve_w_constr(Q, rhs, constr, Dxy, log_det_Q, sol, V2);

    std::cout << "sol(1:10) = " << sol.head(10).transpose() << std::endl;

    MatrixXd W2(num_constr, num_constr);
    update_mean_constr(Dxy, e, sol, V2, W2);

    std::cout << "sol(1:10) = " << sol.head(10).transpose() << std::endl;
    std::cout << "Dxy       = " << Dxy.block(0,0,1,10) << std::endl;

    MatrixXd DxyDxyT = Dxy*Dxy.transpose();
    // .logDeterminant() is in cholmod module, requires inclusion of all of cholmod ...
    // W = D*Q^-1*t(D), want log(sqrt(1/det(W)) = - 0.5 * log(det(W)) 
    double log_det_Q_constr = 0.5 * log_det_Q - 0.5 * log(DxyDxyT.determinant()) - 0.5 * log(W2.determinant());
    std::cout << "constrained log det = " << log_det_Q_constr << std::endl;

    // to compute remaining terms, if e = 0, this will be zero.
    Vect DxyMu = Dxy*sol; // => TODO: this is not right ... but what is?
    double DxyMuWinvDxyMu = DxyMu.transpose()*W2.inverse()*DxyMu;
    // log(pi(Dx | x) = -0.5 log(|A*A^T|)), why no other term?
    double val = sol.transpose()*Q*sol + DxyMuWinvDxyMu;
    std::cout << "constrained val     = " << val << std::endl;

    delete solverQst;
    delete solverQ;


#if 0

    for(int j=0; j<5; j++){
        std::cout << "" << std::endl;
        
        Solver* solverQ;
        Solver* solverQst;

        int MPI_rank = 0;

        solverQ   = new PardisoSolver(MPI_rank);
        solverQst = new PardisoSolver(MPI_rank);

        for(int i=0; i<2; i++){
            std::cout << "outer iter = " << j << ", inner iter = " << i << std::endl;


            double log_det_Qst;
            double log_det_Q;


            #pragma omp parallel
            #pragma omp single
            {

            #pragma omp task
            {
            solverQ->factorize(Qst, log_det_Qst);
            //std::cout << "log det Qst = " << log_det_Qst << std::endl;
            }

            #pragma omp task
            {
            solverQst->factorize_solve(Q, rhs, sol, log_det_Q);
            //std::cout << "log det Q   = " << log_det_Q << std::endl;
            }

            } // end pragma omp parallel section
        }

        // =========================================================================== //
        delete solverQst;
        delete solverQ;
    }

#endif

    
    
  return 0;


  }
