#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <functional> // to pass functions as arguments in other functions

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SparseExtra>   // includes saveMarket

#include <armadillo>
#include "generate_testMat_st_s_field.cpp"
#include "../../read_write_functions.cpp"

#include "../../RGF/RGF.H"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

using namespace std;

//#define PRINT_MSG

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

void construct_Q_spatial(SpMat& Qs, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2){

	// Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
	// return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
	double exp_theta1 = exp(theta[0]);
	double exp_theta2 = exp(theta[1]);
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


void construct_Q_spat_temp(SpMat& Qst, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2){

	std::cout << "theta : " << theta.transpose() << std::endl;

	double exp_theta1 = exp(theta[0]);
	double exp_theta2 = exp(theta[1]);
	double exp_theta3 = exp(theta[2]);

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
		Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

		//std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
}


void construct_Q(SpMat& Q, int ns, int nt, int nss, int nb, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

	double exp_theta0 = exp(theta[0]);
	int nst = ns*nt;
    int nu  = nst + nss;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

	if(ns > 0){
		SpMat Qst(nst, nst);
        SpMat Qss(nss, nss);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(Qst, theta(seq(1,3)), c0, g1, g2, g3, M0, M1, M2);
            if(nss > 0){
                construct_Q_spatial(Qss, theta(seq(4,5)), c0, g1, g2);
            }
		} else {	
			construct_Q_spatial(Qst, theta(seq(1,2)), c0, g1, g2);
		}	

        std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
        std::cout << "Qss : \n" << Qss.block(0,0,10,10) << std::endl;

		//Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
		// construct Qx from Qs values, extend by zeros 
		size_t n = ns*nt + nss + nb;
		SpMat Qx(n,n);         // default is column major			

		int nnz_st = Qst.nonZeros();
		Qx.reserve(nnz_st);

		for (int k=0; k<Qst.outerSize(); ++k)
		  for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
		  {
		    Qx.insert(it.row(),it.col()) = it.value();                 
		  }

        if(nss > 0){
            for (int k=0; k<Qss.outerSize(); ++k)
		        for (SparseMatrix<double>::InnerIterator it(Qss,k); it; ++it)
		        {
		            Qx.insert(it.row()+nst,it.col()+nst) = it.value();                 
		        }

        }

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

/////////////////// Prior /////////////////
double cond_LogPriorLat(SpMat& Q, Vect& beta){
    Vect mu = Vect::Zero(beta.size());

    double f_val = -0.5 * (beta - mu).transpose() * Q * (beta - mu);
    return f_val;
}

/////////////////// Likelihoods /////////////////
// Poisson
double cond_LogPoisLik(Vect& eta, Vect& y){
    double f_val = eta.dot(y) - eta.array().exp().sum();
    return f_val;
}

// TODO: include scaling constant E for each eta -> will also be required in input ...
// 
double cond_negLogPoisLik(Vect& eta, Vect& y, Vect& E){
    // actually link function fixed here but to make input the same ...
    double f_val = eta.dot(y) - (E.array()*(eta.array().exp())).sum();
    return -1*f_val;
}

double cond_LogPois(SpMat& Q, SpMat& A, Vect& E, Vect& beta, Vect& y){

    double f_val_prior = cond_LogPriorLat(Q, beta);

    Vect eta = A * beta;
    double f_val_lik = cond_LogPoisLik(eta, y);

    // times -1: want to minimize
    double f_val = -1 * (f_val_prior + f_val_lik);
    return f_val;
}

// LINK FUNCTIONS
void link_f_sigmoid(Vect& x, Vect& sigmoidX){
    //  1/(1 + e^-x)
    sigmoidX = 1.0 /(1.0 + (-1*x).array().exp());
}

// Binomial
double cond_negLogBinomLik(Vect& eta, Vect& y, Vect& ntrials){
    int m = eta.size();

    Vect linkEta(m);
    // hardcode sigmoid for now
    link_f_sigmoid(eta, linkEta);    
    Vect logLinkEta = linkEta.array().log();
    Vect tmpLinkEta = (Vect::Ones(m) - linkEta).array().log();

    double f_val = y.dot(logLinkEta) + (ntrials - y).dot(tmpLinkEta);
    return -f_val;
}

double cond_logBinom(SpMat& Q, SpMat& A, Vect& ntrials, Vect& beta, Vect& y){
    double f_val_prior = cond_LogPriorLat(Q, beta);

    Vect eta = A * beta;
    double f_val_neg_lik = cond_negLogBinomLik(eta, y, ntrials);

    double f_val = -1 * f_val_prior + f_val_neg_lik;

    return f_val;
}

// general formulation for evaluating log conditional distribution
// pass likelihood & link function as an argument
// later fix these things in class constructor ...
double cond_logDist(SpMat &Q, SpMat& A, Vect& beta, Vect& y, 
                    function<double(Vect&, Vect&, Vect&)> lik_func){

    double f_val = 0;
    return f_val;
}


// naive FIRST ORDER CENTRAL DIFFERENCE (to be improved ...?)
// will simplify once we have this inside class ... only beta will be a variable ...
// expects as input
void FD_gradient(Vect& eta, Vect& y, Vect& E, Vect& grad, function<double(Vect&, Vect&, Vect&)> lik_func){
    int m = eta.size();
    double h = 1e-5;

    // probably better way to do this ...
    SpMat epsId(m,m);
    epsId.setIdentity();
    epsId = h * epsId;

    for(int i=0; i<m; i++){
        Vect eta_forward      = eta + epsId.col(i);
        double f_eta_forward  = lik_func(eta_forward, y, E);

        Vect eta_backward     = eta - epsId.col(i);
        double f_eta_backward = lik_func(eta_backward, y, E);

        grad(i) = (f_eta_forward - f_eta_backward) / (2*h);
    }
}

// naive SECOND ORDER DIFFERENCE: DIAGONAL of Hessian
// expects as input cond_LogPois (generalize ... )
void FD_diag_hessian(Vect& eta, Vect& y, Vect& E, Vect& diag_hess, function<double(Vect&, Vect&, Vect&)> lik_func){
    int m = eta.size();
    double h = 1e-5;

    // probably better way to do this ...
    SpMat epsId(m,m);
    epsId.setIdentity();
    epsId = h * epsId;

    double f_eta = lik_func(eta, y, E);
    for(int i=0; i<m; i++){
        Vect eta_forward    = eta + epsId.col(i);
        double f_eta_forward  = lik_func(eta_forward, y, E);

        Vect eta_backward     = eta - epsId.col(i);
        double f_eta_backward = lik_func(eta_backward, y, E);

        diag_hess(i) = (f_eta_forward - 2*f_eta + f_eta_backward) / (h*h);
    }

}


// within class less inputs required
void NewtonIter(SpMat& Qprior, SpMat& A, Vect& E, Vect& y, Vect& beta, function<double(Vect&, Vect&, Vect&)> lik_func){

    int n = beta.size();
    int m = y.size();

    // prepare for iteration
    Vect beta_new = beta;
    Vect beta_old = Vect::Random(n);

    Vect eta(m);
    Vect gradLik(m);
    Vect diag_hess_eta(m);

    SpMat hess_eta(m,m);
    hess_eta.setIdentity();

    Vect beta_update(n);

    Vect FoD(n);
    SpMat SoD(n,n);

    // Eigen solver for now
    SimplicialLLT<SpMat> solverQ;

    // iteration
    int counter = 0;
    while((beta_new - beta_old).norm() > 1e-5){
        beta_old = beta_new;
        counter += 1;

        if(counter > 20){
            printf("max number of iterations reached in inner Iteration!\n");
            exit(1);
        }

        eta = A * beta_new;

        // compute gradient
        FD_gradient(eta, y, E, gradLik, lik_func);
        // gradient of negative Log conditional  (minimization)
        FoD = Qprior * beta_new + A.transpose() * gradLik;

        // compute hessian
        std::cout << "beta: " << beta_new.transpose() << std::endl;
        FD_diag_hessian(eta, y, E, diag_hess_eta, lik_func);
        hess_eta.diagonal() = diag_hess_eta;
        // hessian of negative log conditional (minimization)
        SpMat hess = Qprior + A.transpose() * hess_eta * A;

        solverQ.compute(hess);

        if(solverQ.info()!=Success) {
            cout << "Oh: Very bad. Hessian not pos. definite." << endl;
            exit(1);
        }

        // Newton step hess(x_k)*(x_k+1 - x_k) = - grad(x_k)
        // beta_update = beta_new - beta_old
        beta_update = solverQ.solve(-FoD);
        beta_new    = beta_update + beta_old;

    }

    beta = beta_new;

}

/* ===================================================================== */

int main(int argc, char* argv[])
{

size_t i; // iteration variable

#if 1
    // generate dummy test case for Poisson distributed data
    /*
    int ns=3;
    int nt=3;
    int nss=1;
    int nb=2;
    int n = ns*nt + nss + nb;
    */

    //SpMat Q = gen_test_mat_base4(ns, nt, nss, nb);
    //SpMat Q = gen_test_mat_base4_prior(ns, nt, nss);
    //std::cout << "Q : \n" << MatrixXd(Q) << std::endl;

    if(argc != 1 + 4){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "[integer: n]                number of latent variables " << std::endl;
        std::cerr << "[integer: m]                number of observations " << std::endl;
        std::cout << "[string:likelihood]         assumed distribution data " << std::endl;
        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        exit(1);
    }

    size_t n              = atoi(argv[1]);
    size_t m              = atoi(argv[2]);
    std::string likelihood = argv[3];

    if(likelihood.compare("Poisson") || likelihood.compare("poisson")){
        likelihood = "Poisson";
    } else if(likelihood.compare("Binomial") || likelihood.compare("binomial")) {
        likelihood = "Binomial";
    } else {
        std::cout << "unknown likelihood: " << likelihood << std::endl;
        exit(1);
    }

    std::string base_path = argv[4];    
    std::cout << "n = " << n << ", m = " << m << ", likelihood: " << likelihood << ", base path: " << base_path << std::endl;

    std::string A_file        =  base_path + "/A_" + to_string(m) + "_" + to_string(n) + ".dat";
    file_exists(A_file); 
    SpMat A = readCSC(A_file);
    //std::cout << "A: \n" << MatrixXd(A) << std::endl;

    std::string y_file        =  base_path + "/y_" + to_string(m) + "_1" + ".dat";
    file_exists(y_file);
    Vect y = read_matrix(y_file, m, 1);
    //std::cout << "y: " << y.transpose() << std::endl;

    //Vect E = Vect::Ones(m);
    std::string extraCoeff_file        =  base_path + "/extraCoeff_" + to_string(m) + "_1" + ".dat";
    file_exists(extraCoeff_file);
    Vect E = read_matrix(extraCoeff_file, m, 1);
    std::cout << "E: " << E.transpose() << std::endl;

    std::string beta_file        =  base_path + "/beta_original_" + to_string(n) + "_1" + ".dat";
    file_exists(beta_file);
    Vect beta_original = read_matrix(beta_file, n, 1);
    std::cout << "beta original: " << beta_original.transpose() << std::endl;



#if 0
    // number of regression coefficients
    int n = 3;
    // number of data samples/observations
    int m = 10;

    // regression coefficients beta
    Vect beta = 2*Vect::Random(n);

    // matrix with covariates A
    MatrixXd A_dense = 0.5*MatrixXd::Random(m,n);
    A_dense.col(0)   = Vect::Ones(m); // set first column to zero for offset
    SpMat A = A_dense.sparseView();

    // linear predictor mu
    Vect muTmp = A*beta;
    Vect mu = muTmp.array().exp();
    std::cout << "mu: " << mu.transpose() << std::endl;

    Vect y(m);

    std::default_random_engine generator;

    for (int i=0; i<m; ++i) {
        std::poisson_distribution<int> distribution(mu[i]);
        y(i) = distribution(generator);
    }

    std::cout << "y: " << y.transpose() << std::endl;
#endif

#if 1
    Vect sigmoidBeta(n);
    link_f_sigmoid(beta_original, sigmoidBeta);
    std::cout << "sigmoid(beta) = " << sigmoidBeta.transpose() << std::endl;

    // define prior Q
    SpMat Qprior(n,n);
    Qprior.setIdentity();
    Qprior = 0.001 * Qprior;

    double f_val_prior = cond_LogPriorLat(Qprior, beta_original);
    printf("val LogPriorLat : %f\n", f_val_prior);

    Vect eta = A * beta_original;

    if(likelihood.compare("Poisson")){
        double f_val_lik = cond_negLogPoisLik(eta, y, E);
        printf("val negLogPoisLik : %f\n", f_val_lik);

        double f_val = cond_LogPois(Qprior, A, E, beta_original, y);
        printf("val LogPois : %f\n", f_val);

        Vect gradLik(m);
        FD_gradient(eta, y, E, gradLik, &cond_negLogPoisLik);
        Vect grad = Qprior * beta_original + A.transpose() * gradLik;
        cout << "gradient: " << grad.transpose() << std::endl;

        //Vect eta = A * beta_original;
        Vect diag_hess_eta(m);
        FD_diag_hessian(eta, y, E, diag_hess_eta, &cond_negLogPoisLik);
        //cout << "diag hessian: " << diag_hess_eta.transpose() << std::endl;

        SpMat hess_eta(m,m);
        hess_eta.setIdentity();
        hess_eta.diagonal() = diag_hess_eta;

        SpMat hess = Qprior + A.transpose() * hess_eta * A;
        cout << "Hessian: \n" << MatrixXd(hess) << std::endl;

        Vect beta = beta_original + 0.5*Vect::Random(n);
        NewtonIter(Qprior, A, E, y, beta, &cond_negLogPoisLik);
        std::cout << "final beta estimate: " << beta.transpose() << std::endl;

    } else if(likelihood.compare("Binomial")){

        double f_val_lik = cond_negLogBinomLik(eta, y, E);
        printf("val negLogBinomLik : %f\n", f_val_lik);

        double f_val = cond_logBinom(Qprior, A, E, beta_original, y);
        printf("val LogBinom : %f\n", f_val);

        Vect gradLik(m);
        FD_gradient(eta, y, E, gradLik, &cond_negLogBinomLik);
        Vect grad = Qprior * beta_original + A.transpose() * gradLik;
        cout << "gradient: " << grad.transpose() << std::endl;

        //Vect eta = A * beta_original;
        Vect diag_hess_eta(m);
        FD_diag_hessian(eta, y, E, diag_hess_eta, &cond_negLogBinomLik);
        //cout << "diag hessian: " << diag_hess_eta.transpose() << std::endl;

        SpMat hess_eta(m,m);
        hess_eta.setIdentity();
        hess_eta.diagonal() = diag_hess_eta;

        SpMat hess = Qprior + A.transpose() * hess_eta * A;
        cout << "Hessian: \n" << MatrixXd(hess) << std::endl;

        Vect beta = beta_original + 0.5*Vect::Random(n);
        NewtonIter(Qprior, A, E, y, beta, &cond_negLogBinomLik);
        std::cout << "final beta estimate: " << beta.transpose() << std::endl;

    }


#endif

    Vect rhs(n);
    rhs.setOnes(n);

    exit(1);

#else

    if(argc != 1 + 7){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points add. spatial field " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        std::cerr << "[string:solver_type]        BTA or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns  = atoi(argv[1]);
    size_t nt  = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb  = atoi(argv[4]);
    std::cout << "ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    size_t no = atoi(argv[5]);
    //std::string no_s = argv[5];
    // to be filled later

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nss + nb);

    std::string base_path = argv[6];    

    std::string solver_type = argv[7];
    // check if solver type is neither PARDISO nor RGF :
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

        if(nss == 0){
            dim_th = 4;
        } else if(nss > 0){
            dim_th = 6;
        } else {
            printf("nss invalid!! nss = %ld\n", nss);
            exit(1);
        }

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
        // prec noise, theta_st_1, theta_st_2, theta_st_3, theta_s_1, theta_s_2
	    //theta << 5, -10, 2.5, 1;
        //theta << 4.000000, -3.344954,  1.039721,  1.386294, 3.0, 2.0; // equals 4,0,0,0 in param scale        
        theta << 1.3539441, -4.4696240,  0.6342557,  1.6739764,  -4.6078180, 2.2436936;
        std::cout << "theta : " << theta.transpose() << std::endl;
	    //theta = {3, -5, 1, 2};
	    //theta.print();
  	}

#endif

printf("# threads: %d\n", omp_get_max_threads());


    
  return 0;

  }
