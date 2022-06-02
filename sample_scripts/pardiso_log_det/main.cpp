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


void update_mean_constr(MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W, MatrixXd& U, Vect& updated_sol){

    // now that we have V = Q^-1*t(Dxy), compute W = Dxy*V
    W = D*V;
    //std::cout << "W = " << W << std::endl;
    // U = W^-1*V^T, W is spd and small
    // TODO: use proper solver ...
    U = W.inverse()*V.transpose();
    //std::cout << "U = " << U << std::endl;

    Vect c = D*sol - e;
    updated_sol = sol - U.transpose()*c;

    std::cout << "sum(updated_sol) = " << (D*updated_sol).sum() << std::endl;

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

    size_t nu = ns*nt;

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


#if 1
    // =========================================================================== //
    // set constraints
    std::cout << "setting up constraints." << std::endl;
    bool constr = true;
    int num_constr = 1;

    // Dx*u = e, and accordingly Dxy*[u,b] = e (constraints only random effects, patch with zeros)
    Vect e = Vect::Zero(1);
    //Vect e = Vect::Ones(1);

    MatrixXd Dst = 0.001*MatrixXd::Ones(1,nu);
    // TODO: better way to do this?!
    /*SpMat D_Mat = KroneckerProductSparse<SpMat, SpMat>(M0, c0);
    D_st = D_Mat.diagonal();  */
    //std::cout << "Dst(-10)  : " << Dst << std::endl; //.block(0,nu-10, 1, nu-1)

    MatrixXd Dxy(num_constr, n);
    Dxy << Dst, MatrixXd::Zero(1,nb);
    //std::cout << "Dxy(-10) : " << Dxy << std::endl; //.block(0,n-10, 1, n-1)

    // =========================================================================== //
    // initialize solvers

    int MPI_rank = 0;

    Solver* solverQst;
    Solver* solverQ;

    solverQst = new PardisoSolver(MPI_rank);
    solverQ   = new PardisoSolver(MPI_rank);

#endif


#if 0
    // ========================================================================== //
    // generate constraints with 3D normal for testing 

    // generate covariance matrix
    int m = 12;
    MatrixXd M = MatrixXd::Random(m,m);
    MatrixXd Cov = M*M.transpose();
    MatrixXd Prec = Cov.inverse();
    //std::cout << "Cov = \n" << Cov << std::endl;
    //std::cout << "eigenvalues(Cov) = " <<  Cov.eigenvalues().real().transpose() << std::endl;
    Vect mu = Vect::Random(m,1);
    std::cout << "mu = " << mu.transpose() << std::endl;

    Vect rhs_s = Prec*mu;

    // set constraints
    bool constr = true;
    int num_constr = 1;

    int b = 2;
    MatrixXd Dx(num_constr, m);
    Dx << MatrixXd::Ones(num_constr,m-b), MatrixXd::Zero(num_constr, b);
    std::cout << "Dx = " << Dx << std::endl;
    Vect e = Vect::Zero(1);
#endif


#if 0
    // ========================================================================== //
    // compute pi(x | \theta, Dx = e )
    MatrixXd Vst(nu, num_constr);

    double log_det_Qst;
    solverQst->factorize_w_constr(Qst, constr, Dst, log_det_Qst, Vst);

    MatrixXd Wst(num_constr, num_constr);
    MatrixXd Ust(num_constr, nu);
    Vect mu_st = Vect::Zero(nu);  // zero mean
    Vect constr_mu_st(nu);
    update_mean_constr(Dst, e, mu_st, Vst, Wst, Ust, constr_mu_st);

    // constr_mu_st will by definition satisfy Ax = e, hence choose x = constr_mu_st
    //Vect x_st = constr_mu_st;
    Vect x_st = Vect::Zero(nu);

    //std::cout << "Wx = " << Wx << ", inv(invW) = " << invW.inverse() << std::endl;

    std::cout << "constr_mu = " << constr_mu_st.transpose() << std::endl;
    //std::cout << "mu        = " << mu_x.transpose() << std::endl;

    
    // log(pi(x)) 
    double log_pi_x_st    = - 0.5*nu*log(2*M_PI) + 0.5*log_det_Qst - 0.5*(x_st-mu_st).transpose()*Qst*(x_st-mu_st);
    // log(pi(Ax|x)) 
    MatrixXd DstDstT = Dst*Dst.transpose();
    std::cout << "DstDstT = " << DstDstT << std::endl;
    // .logDeterminant() is in cholmod module, requires inclusion of all of cholmod ...
    // W = D*Q^-1*t(D), want log(sqrt(1/det(W)) = - 0.5 * log(det(W)) 
    double log_pi_Ax_x_st = - 0.5*log(DstDstT.determinant());
    // log(pi(Ax)), W1 is covariance matrix
    double log_pi_Ax_st   = - 0.5*Dst.rows()*log(2*M_PI) - 0.5*log(Wst.determinant()) - 0.5*(Dst*x_st - Dst*mu_st).transpose()*Wst.inverse()*(Dst*x_st - Dst*mu_st);

    double total = log_pi_x_st + log_pi_Ax_x_st - log_pi_Ax_st;
    //std::cout << - 0.5*m*log(2*M_PI) - (- 0.5*Dx.rows()*log(2*M_PI)) << " " << - 0.5*(m-Dx.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val Bayes = " << total << std::endl;  

    // ================================================================================================== //
    // compute control, gets very inaccurate very quickly as dimension increases 
    // ... always check that eigenvalue decomposition is reasonable -> identify zero eigenvalues ...

    // compute constrained mean and covariance
    MatrixXd Qst_d = MatrixXd(Qst);
    MatrixXd Cov = Qst_d.inverse();
    MatrixXd invW = (Dst*Cov*Dst.transpose()).inverse();
    std::cout << "Wst = " << Wst << ", W = " << Dst*Cov*Dst.transpose() << std::endl;
    std::cout << "inv(Wst) = " << Wst.inverse() << ", invW = " << invW << std::endl;
    Vect constr_mu = mu_st - Cov*Dst.transpose()*invW*(Dst*mu_st - e);
    std::cout << "norm(constr_mu - constr_mu_st) = " << (constr_mu - constr_mu_st).norm() << std::endl;
    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    MatrixXd constr_Cov = Cov - Cov*Dst.transpose()*invW*Dst*Cov;
    //std::cout << "constr_Cov = \n" << constr_Cov << std::endl;

    EigenSolver<MatrixXd> es(constr_Cov);
    MatrixXd V = es.eigenvectors().real();
    //cout << "Eigenvectors = " << endl << V << endl;

    Vect eivals = es.eigenvalues().real();
    std::cout << "eigenvalues(constr_Cov) = " <<  eivals.transpose() << std::endl;
    //std::cout << "V*D*V^T = " << endl << V*eivals.asDiagonal()*V.transpose() << std::endl;
          
    // identify non-zero eigenvalues
    double log_sum_st = 0;
    double prod_st = 1;
    Vect invD = Vect::Zero(nu);
    for(int i=0; i<nu; i++){
        if(eivals[i] > 1e-7){
            log_sum_st += log(eivals[i]);
            prod_st *= eivals[i];
            invD[i] = 1/eivals[i];
        }
    }

    MatrixXd invD_mat = invD.asDiagonal();
    //std::cout << "invD = " << invD_mat << std::endl;

    //printf("sum(eivals)     = %f\n", log(prod));
    //printf("log_sum(eivals) = %f\n", log_sum);

    // compute log pi(x | Ax = e), evaluated at x, make sure x satisfies condition ...
    //Vect x = Vect::Zero(m);
    //Vect x = constr_mu;
    MatrixXd pInv = V*invD.asDiagonal()*V.transpose();
    //std::cout << "pInv = \n" << pInv << std::endl;
    //std::cout << "Cov = \n" << V*eivals.asDiagonal()*V.transpose() << std::endl;
    std::cout << "(V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() = " << (V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() << std::endl;
    std::cout << "norm(x_st - constr_mu) = " << (x_st - constr_mu).norm() << std::endl;
    double temp = (x_st - constr_mu).transpose()*pInv*(x_st - constr_mu);
    std::cout << "temp = " << temp << std::endl;
    double log_val = - 0.5*(nu-Dst.rows())*log(2*M_PI) - 0.5*log_sum_st - 0.5*temp;
    std::cout << - 0.5*nu*log(2*M_PI) - (- 0.5*Dst.rows()*log(2*M_PI)) << " " << - 0.5*(nu-Dst.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val direct = " << log_val << "\n" << std::endl;
#endif


#if 1
    // ========================================================================== //
    // compute constrained pi(x | \theta, y, Dx = e )

    Vect mu_xy(n);
    MatrixXd Vxy(n, num_constr);
    double log_det_Q;
    solverQ->factorize_solve_w_constr(Q, rhs, constr, Dxy, log_det_Q, mu_xy, Vxy);
    //std::cout << "norm(Q*Vxy - t(Dxy)) = " << (Q*Vxy - Dxy.transpose()).norm() << std::endl;
    //std::cout << "norm(Q*mu_xy - rhs) = " << (Q*mu_xy - rhs).norm() << std::endl;

    MatrixXd Wxy(num_constr, num_constr);
    MatrixXd Uxy(num_constr, n);
    Vect constr_mu_xy(n);
    update_mean_constr(Dxy, e, mu_xy, Vxy, Wxy, Uxy, constr_mu_xy);
    std::cout << "constr_mu_xy = " << constr_mu_xy.transpose() << std::endl;

    // constr_mu_st will by definition satisfy Ax = e, hence choose x = constr_mu_xy
    //Vect x_xy = constr_mu_xy;
    Vect x_xy = Vect::Zero(n);

    // log(pi(x)) 
    std::cout << "log det Q = " << log_det_Q << std::endl;
    std::cout << "- 0.5*(x_xy-mu_xy).transpose()*Q*(x_xy-mu_xy) = " << - 0.5*(x_xy-mu_xy).transpose()*Q*(x_xy-mu_xy) << std::endl;
    double log_pi_x    = - 0.5*n*log(2*M_PI) + 0.5*log_det_Q - 0.5*(x_xy-mu_xy).transpose()*Q*(x_xy-mu_xy);
    // log(pi(Ax|x)) 
    MatrixXd DxyDxyT = Dxy*Dxy.transpose();
    // .logDeterminant() is in cholmod module, requires inclusion of all of cholmod ...
    // W = D*Q^-1*t(D), want log(sqrt(1/det(W)) = - 0.5 * log(det(W)) 
    double log_pi_Ax_x = - 0.5*log(DxyDxyT.determinant());
    // log(pi(Ax)), W1 is covariance matrix
    double log_pi_Ax   = - 0.5*Dxy.rows()*log(2*M_PI) - 0.5*log(Wxy.determinant()) - 0.5*(Dxy*x_xy - Dxy*mu_xy).transpose()*Wxy.inverse()*(Dxy*x_xy - Dxy*mu_xy);

    double total_xy = log_pi_x + log_pi_Ax_x - log_pi_Ax;
    //std::cout << - 0.5*m*log(2*M_PI) - (- 0.5*Dx.rows()*log(2*M_PI)) << " " << - 0.5*(m-Dx.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val Bayes cond = " << total_xy << std::endl;  

    // ================================================================================================== //
    // compute control, can get very inaccurate very quickly as dimension increases !!! pseudo-inverse not numerically stable ...    
    // compute constrained mean and covariance
    MatrixXd Q_d = MatrixXd(Q);
    MatrixXd Cov_xy = Q_d.inverse();
    MatrixXd invWxy = (Dxy*Cov_xy*Dxy.transpose()).inverse();
    std::cout << "Wxy = " << Wxy << ", W = " << Dxy*Cov_xy*Dxy.transpose() << std::endl;
    std::cout << "inv(Wxy) = " << Wxy.inverse() << ", invWxy = " << invWxy << std::endl;
    Vect constr_mu_xy2 = mu_xy - Cov_xy*Dxy.transpose()*invWxy*(Dxy*mu_xy - e);
    std::cout << "norm(constr_mu_xy - constr_mu_xy2) = " << (constr_mu_xy - constr_mu_xy2).norm() << std::endl;
    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    MatrixXd constr_Cov_xy = Cov_xy - Cov_xy*Dxy.transpose()*invWxy*Dxy*Cov_xy;
    //std::cout << "constr_Cov = \n" << constr_Cov << std::endl;

    EigenSolver<MatrixXd> es_xy(constr_Cov_xy);
    MatrixXd EV_xy = es_xy.eigenvectors().real();
    //cout << "Eigenvectors = " << endl << V << endl;

    Vect eivals_xy = es_xy.eigenvalues().real();
    std::cout << "eigenvalues(constr_Cov_xy) = " <<  eivals_xy.transpose() << std::endl;
    //std::cout << "V*D*V^T = " << endl << V*eivals.asDiagonal()*V.transpose() << std::endl;
          
    // identify non-zero eigenvalues
    double log_sum = 0;
    double prod = 1;
    Vect invD_xy = Vect::Zero(n);
    for(int i=0; i<n; i++){
        if(eivals_xy[i] > 1e-7){
            log_sum += log(eivals_xy[i]);
            prod *= eivals_xy[i];
            invD_xy[i] = 1/eivals_xy[i];
        }
    }

    MatrixXd invDxy_mat = invD_xy.asDiagonal();
    //std::cout << "invD = " << invD_mat << std::endl;

    printf("sum(eivals)     = %f\n", log(prod));
    printf("log_sum(eivals) = %f\n", log_sum);

    // compute log pi(x | Ax = e), evaluated at x, make sure x satisfies condition ...
    //Vect x = Vect::Zero(m);
    //Vect x = constr_mu;
    MatrixXd pInv_xy = EV_xy*invD_xy.asDiagonal()*EV_xy.transpose();
    //std::cout << "pInv = \n" << pInv << std::endl;
    //std::cout << "Cov = \n" << V*eivals.asDiagonal()*V.transpose() << std::endl;
    std::cout << "(V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() = " << (EV_xy*eivals_xy.asDiagonal()*EV_xy.transpose() - constr_Cov_xy).norm() << std::endl;
    double temp_xy = (x_xy - constr_mu_xy2).transpose()*pInv_xy*(x_xy - constr_mu_xy2);
    std::cout << "temp = " << temp_xy << std::endl;
    double log_val_xy = - 0.5*(n-Dxy.rows())*log(2*M_PI) - 0.5*log_sum - 0.5*temp_xy;
    std::cout << - 0.5*n*log(2*M_PI) - (- 0.5*Dxy.rows()*log(2*M_PI)) << " " << - 0.5*(n-Dxy.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val direct = " << log_val_xy << std::endl;

#endif

#if 0

    // =========================================================================== //
    // initialize solvers

    int MPI_rank = 0;

    Solver* solverQst;
    Solver* solverQ;

    solverQst = new PardisoSolver(MPI_rank);
    solverQ   = new PardisoSolver(MPI_rank);

    //SpMat sPrec = Prec.sparseView();

    //Vect e = Vect::Zero(num_constr);
    //MatrixXd Dx(num_constr, ns*nt);
    //Dx.row(0) << MatrixXd::Ones(num_constr,ns*nt);
    //MatrixXd V1(m, num_constr);

    //double log_det_Qst;
    //std::cout << "sPrec = \n" << sPrec << std::endl;
    //std::cout << "Dx = \n" << Dx << std::endl;
    //solverQst->factorize_w_constr(sPrec, constr, Dx, log_det_Qst, V1);

    MatrixXd Q_d = MatrixXd(Q);
    MatrixXd Cov = Q_d.inverse();

    int m = n;

    double log_det_Q;
    Vect mu_xy(m);
    MatrixXd V_xy(m,num_constr);
    solverQ->factorize_solve_w_constr(Q, rhs, constr, Dxy, log_det_Q, mu_xy, V_xy);

    //std::cout << "norm(V1 - V2) = " << (V1-V2).norm() << ", norm(mu - mu_xy) = " << (mu - mu_xy).norm() << std::endl;


    MatrixXd W_xy(num_constr, num_constr);
    MatrixXd Ux(num_constr, m);
    // here sol is equal to zero?
    //Vect mu_x = Vect::Zero(m);
    Vect constr_mu(m);
    update_mean_constr(Dxy, e, mu_xy, V_xy, W_xy, Ux, constr_mu);

    //Vect x = constr_mu;
    Vect x = Vect::Zero(m);

    //std::cout << "W1 = " << W1 << ", inv(invW) = " << invW.inverse() << std::endl;

    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    //std::cout << "mu        = " << mu.transpose() << std::endl;

    
    // log(pi(x)) 
    double log_pi_x    = - 0.5*m*log(2*M_PI) + 0.5*log_det_Q - 0.5*(x-mu_xy).transpose()*Q*(x-mu_xy);
    // log(pi(Ax|x)) 
    MatrixXd DxyDxyT = Dxy*Dxy.transpose();
    // .logDeterminant() is in cholmod module, requires inclusion of all of cholmod ...
    // W = D*Q^-1*t(D), want log(sqrt(1/det(W)) = - 0.5 * log(det(W)) 
    double log_pi_Ax_x = - 0.5*log(DxyDxyT.determinant());
    // log(pi(Ax)), W1 is covariance matrix
    double log_pi_Ax   = - 0.5*Dxy.rows()*log(2*M_PI) - 0.5*log(W_xy.determinant()) - 0.5*(Dxy*x - Dxy*mu_xy).transpose()*W_xy.inverse()*(Dxy*x - Dxy*mu_xy);

    double total = log_pi_x + log_pi_Ax_x - log_pi_Ax;
    //std::cout << - 0.5*m*log(2*M_PI) - (- 0.5*Dxy.rows()*log(2*M_PI)) << " " << - 0.5*(m-Dxy.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val Bayes = " << total << std::endl;    


    // ========================================================================================== //
    // compute using eigenvalues
    MatrixXd invW = (Dxy*Cov*Dxy.transpose()).inverse();
    std::cout << "W_xy = " << W_xy << ", W = " << Dxy*Cov*Dxy.transpose() << std::endl;
    std::cout << "inv(W_xy) = " << W_xy.inverse() << ", invW = " << invW << std::endl;
    Vect constr_mu_c = mu_xy - Cov*Dxy.transpose()*invW*(Dxy*mu_xy - e);
    std::cout << "norm(constr_mu - constr_mu_c) = " << (constr_mu - constr_mu_c).norm() << std::endl;
    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    MatrixXd constr_Cov = Cov - Cov*Dxy.transpose()*invW*Dxy*Cov;
    //std::cout << "constr_Cov = \n" << constr_Cov << std::endl;


    EigenSolver<MatrixXd> es(constr_Cov);
    MatrixXd V = es.eigenvectors().real();
    //cout << "Eigenvectors = " << endl << V << endl;

    Vect eivals = es.eigenvalues().real();
    std::cout << "eigenvalues(constr_Cov) = " <<  eivals.transpose() << std::endl;
    //std::cout << "V*D*V^T = " << endl << V*eivals.asDiagonal()*V.transpose() << std::endl;
          
    // identify non-zero eigenvalues
    double log_sum = 0;
    double prod = 1;
    Vect invD = Vect::Zero(m);
    for(int i=0; i<m; i++){
        if(eivals[i] > 1e-7){
            log_sum += log(eivals[i]);
            prod *= eivals[i];
            invD[i] = 1/eivals[i];
        }
    }

    MatrixXd invD_mat = invD.asDiagonal();
    //std::cout << "invD = " << invD_mat << std::endl;

    printf("sum(eivals)     = %f\n", log(prod));
    printf("log_sum(eivals) = %f\n", log_sum);

    // compute log pi(x | Ax = e), evaluated at x, make sure x satisfies condition ...
    //Vect x = Vect::Zero(m);
    //Vect x = constr_mu;
    MatrixXd pInv = V*invD.asDiagonal()*V.transpose();
    //std::cout << "pInv = \n" << pInv << std::endl;
    //std::cout << "Cov = \n" << V*eivals.asDiagonal()*V.transpose() << std::endl;
    std::cout << "(V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() = " << (V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() << std::endl;
    double temp = (x - constr_mu).transpose()*pInv*(x - constr_mu);
    std::cout << "temp = " << temp << std::endl;
    double log_val = - 0.5*(m-Dxy.rows())*log(2*M_PI) - 0.5*log_sum - 0.5*temp;
    std::cout << - 0.5*m*log(2*M_PI) - (- 0.5*Dxy.rows()*log(2*M_PI)) << " " << - 0.5*(m-Dxy.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val direct = " << log_val << "\n" << std::endl;


#endif


#if 0
    // =========================================================================== //
    // initialize solvers

    int MPI_rank = 0;

    Solver* solverQst;
    Solver* solverQ;

    solverQst = new PardisoSolver(MPI_rank);
    solverQ   = new PardisoSolver(MPI_rank);

    SpMat sPrec = Prec.sparseView();

    //Vect e = Vect::Zero(num_constr);
    //MatrixXd Dx(num_constr, ns*nt);
    //Dx.row(0) << MatrixXd::Ones(num_constr,ns*nt);
    //MatrixXd V1(m, num_constr);

    //double log_det_Qst;
    //std::cout << "sPrec = \n" << sPrec << std::endl;
    //std::cout << "Dx = \n" << Dx << std::endl;
    //solverQst->factorize_w_constr(sPrec, constr, Dx, log_det_Qst, V1);

    double log_det_Q;
    Vect mu_xy(m);
    MatrixXd V_xy(m,num_constr);
    solverQ->factorize_solve_w_constr(sPrec, rhs_s, constr, Dx, log_det_Q, mu_xy, V_xy);

    //std::cout << "norm(V1 - V2) = " << (V1-V2).norm() << ", norm(mu - mu_xy) = " << (mu - mu_xy).norm() << std::endl;


    MatrixXd W_xy(num_constr, num_constr);
    MatrixXd Ux(num_constr, m);
    // here sol is equal to zero?
    //Vect mu_x = Vect::Zero(m);
    Vect constr_mu2(m);
    update_mean_constr(Dx, e, mu_xy, V_xy, W_xy, Ux, constr_mu2);

    //Vect x = constr_mu2;
    Vect x = Vect::Zero(m);

    //std::cout << "W1 = " << W1 << ", inv(invW) = " << invW.inverse() << std::endl;

    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    //std::cout << "mu        = " << mu.transpose() << std::endl;

    
    // log(pi(x)) 
    double log_pi_x    = - 0.5*m*log(2*M_PI) + 0.5*log_det_Q - 0.5*(x-mu_xy).transpose()*Prec*(x-mu_xy);
    // log(pi(Ax|x)) 
    MatrixXd DxDxT = Dx*Dx.transpose();
    // .logDeterminant() is in cholmod module, requires inclusion of all of cholmod ...
    // W = D*Q^-1*t(D), want log(sqrt(1/det(W)) = - 0.5 * log(det(W)) 
    double log_pi_Ax_x = - 0.5*log(DxDxT.determinant());
    // log(pi(Ax)), W1 is covariance matrix
    double log_pi_Ax   = - 0.5*Dx.rows()*log(2*M_PI) - 0.5*log(W_xy.determinant()) - 0.5*(Dx*x - Dx*mu_xy).transpose()*W_xy.inverse()*(Dx*x - Dx*mu_xy);

    double total = log_pi_x + log_pi_Ax_x - log_pi_Ax;
    //std::cout << - 0.5*m*log(2*M_PI) - (- 0.5*Dx.rows()*log(2*M_PI)) << " " << - 0.5*(m-Dx.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val Bayes = " << total << std::endl;    


    // ========================================================================================== //
    // compute using eigenvalues
    MatrixXd invW = (Dx*Cov*Dx.transpose()).inverse();
    std::cout << "W_xy = " << W_xy << ", W = " << Dx*Cov*Dx.transpose() << std::endl;
    std::cout << "inv(W_xy) = " << W_xy.inverse() << ", invW = " << invW << std::endl;
    Vect constr_mu = mu - Cov*Dx.transpose()*invW*(Dx*mu_xy - e);
    //std::cout << "norm(constr_mu - constr_mu_st) = " << (constr_mu - constr_mu_st).norm() << std::endl;
    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    MatrixXd constr_Cov = Cov - Cov*Dx.transpose()*invW*Dx*Cov;
    //std::cout << "constr_Cov = \n" << constr_Cov << std::endl;


    EigenSolver<MatrixXd> es(constr_Cov);
    MatrixXd V = es.eigenvectors().real();
    //cout << "Eigenvectors = " << endl << V << endl;

    Vect eivals = es.eigenvalues().real();
    std::cout << "eigenvalues(constr_Cov) = " <<  eivals.transpose() << std::endl;
    //std::cout << "V*D*V^T = " << endl << V*eivals.asDiagonal()*V.transpose() << std::endl;
          
    // identify non-zero eigenvalues
    double log_sum = 0;
    double prod = 1;
    Vect invD = Vect::Zero(m);
    for(int i=0; i<m; i++){
        if(eivals[i] > 1e-7){
            log_sum += log(eivals[i]);
            prod *= eivals[i];
            invD[i] = 1/eivals[i];
        }
    }

    MatrixXd invD_mat = invD.asDiagonal();
    //std::cout << "invD = " << invD_mat << std::endl;

    printf("sum(eivals)     = %f\n", log(prod));
    printf("log_sum(eivals) = %f\n", log_sum);

    // compute log pi(x | Ax = e), evaluated at x, make sure x satisfies condition ...
    //Vect x = Vect::Zero(m);
    //Vect x = constr_mu;
    MatrixXd pInv = V*invD.asDiagonal()*V.transpose();
    //std::cout << "pInv = \n" << pInv << std::endl;
    //std::cout << "Cov = \n" << V*eivals.asDiagonal()*V.transpose() << std::endl;
    std::cout << "(V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() = " << (V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() << std::endl;
    double temp = (x - constr_mu).transpose()*pInv*(x - constr_mu);
    std::cout << "temp = " << temp << std::endl;
    double log_val = - 0.5*(m-Dx.rows())*log(2*M_PI) - 0.5*log_sum - 0.5*temp;
    std::cout << - 0.5*m*log(2*M_PI) - (- 0.5*Dx.rows()*log(2*M_PI)) << " " << - 0.5*(m-Dx.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val direct = " << log_val << "\n" << std::endl;

#endif

#if 0
    double log_det_Q;
    MatrixXd Dxy(num_constr,n);
    Dxy.row(0) << Dx , MatrixXd::Zero(num_constr,nb);

    MatrixXd V2(n,num_constr);
    solverQ->factorize_solve_w_constr(Q, rhs, constr, Dxy, log_det_Q, sol, V2);

    std::cout << "sol(1:10) = " << sol.head(10).transpose() << std::endl;

    MatrixXd W2(num_constr, num_constr);
    MatrixXd Uxy(num_constr, n);
    update_mean_constr(Dxy, e, sol, V2, W2, Uxy);


    // ============= evaluate at new mean ================ //
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


    // ============== update marginal variances =============== //
    Vect inv_diag(n);
    solverQ->selected_inversion(Q, inv_diag);
    std::cout << "inv_diag(1:10) = " << inv_diag.head(10).transpose() << std::endl;

    // \Sigma* = Q^-1 - V*W^-1*V^T => get diag(V*W^-1*V^T)
    // TODO: update such that only diagonal entries are computed ...
    Vect update1 = (V2*W2.inverse()*V2.transpose()).diagonal();
    Vect update2(n);
     
    for(int i = 0; i<n; i++){
        update2[i] = V2.row(i)*Uxy.col(i);
    }

    std::cout << "dim(V2) = " << V2.rows() << " " << V2.cols() << ", dim(Uxy) = " << Uxy.rows() << " " << Uxy.cols() << std::endl;
    //std::cout << "update1 = " << update1.head(10).transpose() << std::endl;
    std::cout << "update2 = \n" << (V2*Uxy).diagonal().head(10).transpose() << "\n" << update2.head(10).transpose() << std::endl;

    std::cout << "norm(update1 - update2) = " << (update1 - update2).norm() << std::endl;
    inv_diag = inv_diag - update1;
    //std::cout << "inv_diag(1:10) = " << inv_diag.head(10).transpose() << std::endl;
    //std::cout << "(V2*W2.inverse()*V2.transpose())[1:10,1:10] : \n" << (V2*W2.inverse()*V2.transpose()).block(0,0,10,10) << std::endl;

    delete solverQst;
    delete solverQ;

#endif


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
