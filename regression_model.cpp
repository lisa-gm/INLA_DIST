// function evaluation regression model 

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// require armadillo for read dense matrix for now
#include <armadillo>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>

using namespace Eigen;
using namespace std;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::CholmodSimplicialLDLT  <SpMat > Solver;
typedef Eigen::VectorXd Vector;

// attention expects complete matrix (not just lower triangular part)
SpMat readCSC(std::string filename)
{

  int n;
  int nnz;

  fstream fin(filename, ios::in);
  fin >> n;
  fin >> n;
  fin >> nnz;

   // allocate memory
  int outerIndexPtr[n+1];
  int innerIndices[nnz];
  double values[nnz];

  for (int i = 0; i < nnz; i++){
    fin >> innerIndices[i];}

  for (int i = 0; i < n+1; i++){
    fin >> outerIndexPtr[i];}

  for (int i = 0; i < nnz; i++){
    fin >> values[i];}

  fin.close();

  Eigen::Map<Eigen::SparseMatrix<double> > A(n,n,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

  return A;
} 

// expects indices for lower triangular matrix
SpMat read_sym_CSC(std::string filename)
{

  int n;
  int nnz;

  fstream fin(filename, ios::in);
  fin >> n;
  fin >> n;
  fin >> nnz;

   // allocate memory
  int outerIndexPtr[n+1];
  int innerIndices[nnz];
  double values[nnz];

  for (int i = 0; i < nnz; i++){
    fin >> innerIndices[i];}

  for (int i = 0; i < n+1; i++){
    fin >> outerIndexPtr[i];}

  for (int i = 0; i < nnz; i++){
    fin >> values[i];}

  fin.close();

  Eigen::Map<Eigen::SparseMatrix<double> > A_lower(n,n,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

  SpMat A = A_lower.selfadjointView<Lower>();
  std::cout << "input A : " << std::endl;
  std::cout << A << std::endl;

  return A;
} 

// for now use armadillo ... do better once we switch to binary

MatrixXd read_matrix(const string filename,  int n_row, int n_col){

    arma::mat X(n_row, n_col);
    X.load(filename, arma::raw_ascii);
    // X.print();

    return Eigen::Map<MatrixXd>(X.memptr(), X.n_rows, X.n_cols);
}


void file_exists(std::string file_name)
{
    if (std::fstream{file_name}) ;
    else {
      std::cerr << file_name << " couldn\'t be opened (not existing or failed to open)\n"; 
      exit(1);
    }
    
}


void solve_cholmod(SpMat *A, Vector *f, Vector *u, double *log_det)
{

	Solver solver;
	solver.analyzePattern(*A);
	solver.factorize(*A);

	*u = solver.solve(*f);

	*log_det = solver.logDeterminant();

	//std::cout << "solution vector u : " << *u << std::endl;

}


void rnorm_gen(int no, double mean, double sd,  Eigen::VectorXd * x, int seed){
  // unsigned int seed = 2;
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution (mean, sd);
 
  for (int i=0; i< x->size(); ++i){
  		(*x)(i)	 = distribution(generator);
  }

}

void generate_ex_regression( int nb,  int no, double tau, Eigen::MatrixXd *B, Vector *b, Vector *y){

	/* ---------------------- construct random matrix of covariates --------------------- */
	Vector B_ones(no); B_ones.setOnes();

	// require different random seed here than in noise -> otherwise cancels each other out
	// val_l will then equal val_d ... 
	Vector B_random(no*(nb-1));
	rnorm_gen(no, 0.0, 1, &B_random, 2);

	Vector B_tmp(no*nb);
	B_tmp << B_ones, B_random;
	//std::cout << B_tmp << std::endl;

	// TODO: fix this!
	Eigen::Map<Eigen::MatrixXd> tmp(B_tmp.data(), no,nb);
	*B = tmp;
	//*B(B_tmp.data());
	//Eigen::MatrixXd::Map(*B) = B_tmp.data(); 
	//std::cout << *B << std::endl;

	/* -------  construct random solution vector of fixed effects & observations -------- */
	*b = 2*(Vector::Random(nb) + Vector::Ones(nb)); 

	double mean = 0.0;
	double sd = 1/sqrt(exp(tau));
	Vector noise_vec(no);

	rnorm_gen(no, mean, sd, &noise_vec, 4);

	*y = (*B)*(*b) + noise_vec;

	/*std::cout << "noise vec " << std::endl;
	std::cout << noise_vec << std::endl; 

	std::cout << "B*b " << std::endl;
	std::cout << (*B)*(*b) << std::endl;

	std::cout << "y " << std::endl;
	std::cout << *y << std::endl; */

}


void eval_likelihood(int no, double theta, double yTy, double *log_det, double *val){
	

	*log_det = no*theta;
	*val = exp(theta)*yTy;

	/*std::cout << "in eval eval_likelihood " << std::endl;
	std::cout << "theta     : " << theta << std::endl;
	std::cout << "yTy : " << yTy << ", exp(theta) : " << exp(theta) << std::endl;

	std::cout << "log det l : " << *log_det << std::endl;
	std::cout << "val l     : " << *val << std::endl; */

}

void construct_Q_b(int nb, int no, double theta, Vector y, Eigen::MatrixXd B, SpMat *Q, Vector *rhs){
	double exp_theta = exp(theta);

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/
	//Q_b = 1e-5*Q_b.setIdentity();

	// Q.e <- Diagonal(no, exp(theta))
	// Q.xy <- Q.x + crossprod(A.x, Q.e)%*%A.x  # crossprod = t(A)*Q.e (faster)	
	*Q = Q_b + exp_theta*B.transpose()*B;

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

	*rhs = exp_theta*B.transpose()*y;

	// exp(theta)*yTy

	// 1e-5*Id + exp(theta)*B'*B

	// mu*Q*mu != exp(theta)*yTy
	// mu*Q*mu = (Q^-1*(exp(theta)*B'*y*(exp(theta)*B'*y)
	// exp(theta)*Q^-1*exp(theta)*B'*B*y'*y. now ignore Q_b part of Q. then Q = exp(theta)*B'*B, ie.
	// exp(theta)*y'*y remains ... 
}


void eval_denominator(int nb, int no, double theta, Vector y, Eigen::MatrixXd B, double *log_det, double *val, SpMat *Q, Vector *rhs, Vector *mu){

	// construct Q_x|y, b_xey
	construct_Q_b(nb, no, theta, y, B, Q, rhs);

	// solve linear system
	solve_cholmod(Q, rhs, mu, log_det);

	// compute value
	*val = mu->transpose()*(*Q)*(*mu); 

	/*std::cout << "in eval eval_denominator " << std::endl;

	std::cout << "rhs " << std::endl; std::cout <<  *rhs << std::endl;
	std::cout << "mu " << std::endl; std::cout << *mu << std::endl;
	std::cout << "Q " << std::endl; std::cout << Eigen::MatrixXd(*Q) << std::endl;


	std::cout << "log det d : " << *log_det << std::endl;
	std::cout << "val d     : " << *val << std::endl; */
}

void eval_post_theta(int nb, int no, double theta, double yTy, Vector y, Eigen::MatrixXd B, double *val, Vector *mu){

	// numerator :
	// log_det(Q.u) doesnt exist here. 
	// eval_likelihood: log_det, -theta*yTy
	double log_det_l;
	double val_l; 
	eval_likelihood(no, theta, yTy, &log_det_l, &val_l);
	/*std::cout << "log det l : "  << log_det_l << std::endl;
	std::cout << "val l     : " << val_l << std::endl;*/

	// denominator :
	// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
	double log_det_d;
	double val_d;
	SpMat Q(nb, nb);
	Vector rhs(nb);

 	eval_denominator(nb, no, theta, y, B, &log_det_d, &val_d, &Q, &rhs, mu);
	/*std::cout << "log det d : " << log_det_d << std::endl;
	std::cout << "val d     : " <<  val_d << std::endl;*/

  	// add everything together
  	*val = -1 * 0.5 * (log_det_l - val_l - log_det_d + val_d);
  	// std::cout << *val << std::endl;

}

// 1D version, forward difference
void eval_gradient(int nb, int no, double theta, double f_theta, double yTy, Vector y, Eigen::MatrixXd B, Vector *mu, double *grad){


	double eps = 0.05;
	double theta_forw = theta + eps;

	double f_theta_forw;

	// have to be careful that mu doesn't get overwritten?
	// write version such that this doesn't happen?
	eval_post_theta(nb, no, theta_forw, yTy, y, B, &f_theta_forw, mu);
	/*std::cout << "theta_forw   : " << theta_forw << std::endl;
	std::cout << "f_theta_forw : " << f_theta_forw << std::endl;*/

	*grad = (1/eps) * (f_theta_forw - f_theta);
}


void eval_invHess(double s, double y, double *invHess){
	double rho = 1/(y*s);
	double Id = 1;

	*invHess = (Id - rho*s*y)*(*invHess)*(Id - rho*y*s) + rho*s*s;
}

double backtracking(int nb, int no, 
				  double theta, double f_theta, double grad_f, double p,
				  double yTy, Vector y, Eigen::MatrixXd B, 
				  double alpha_0, double c, double rho){

	double norm_p = std::abs(p);
	double alpha_p = 1/norm_p;

	double alpha_min = alpha_0*1e-4;
	double alpha = std::min(alpha_0, alpha_p);

	int max_iter = 50;

	double val;
	Vector mu;

	double theta_test;
	double f_theta_test;
	double f_comp;

	for(int it = 0; it <= max_iter; it ++){

		theta_test = theta + alpha*p;
		eval_post_theta(nb, no, theta_test, yTy, y, B, &f_theta_test, &mu);
		/*std::cout << "theta    : " << theta_test << std::endl;
		std::cout << "f(theta) : " << f_theta_test << std::endl;*/

		f_comp = f_theta + c*alpha*grad_f*p;

		if(f_theta_test <= f_comp){
			std::cout << "chosen alpha : " << alpha << std::endl;
			return(alpha);
		}

		if(alpha < alpha_min){
			std::cout << "mininum stepsize reached. chosen alpha : " << alpha << std::endl;
      		return(alpha);
		}

		alpha = alpha * rho;
	}

	return(-1);
		
}


int main(int argc, char* argv[]) 
{ 
	if (argc != 1 + 5) {
		std::cerr << "simple regression model : nb no theta path_to_B_file path_to_y_file " << std::endl;

		std::cerr << "[integer]: 		number of observations (nb)" << std::endl;     
		std::cerr << "[integer]: 		number of observations (no)" << std::endl;  
		std::cerr << "[double] : 		initial value for theta    " << std::endl;   
		std::cerr << "[string:B_file]:  dense matrix in raw raw_ascii" << std::endl;   
		std::cerr << "[string:y_file]:  vector in raw raw_ascii" << std::endl;      
   
 

		exit(1);
	}

	int nb = atoi(argv[1]);
	int no = atoi(argv[2]);

	// initial value for theta
	double theta = atof(argv[3]);
	std::cout << "theta : " << theta << std::endl;	

	/* ---------------------- read in matrices ------------------- */
  	// write binary. write everything to double. 

	std::string B_file = argv[4];
	file_exists(B_file);
	MatrixXd B = read_matrix(B_file, no, nb);
	//std::cout << "B : " << std::endl; std::cout << B << std::endl;


	std::string y_file = argv[5];
	file_exists(y_file);
	Vector y = read_matrix(y_file, no, 1);
	//std::cout << "y : " << std::endl; std::cout << y << std::endl;


	double yTy = y.dot(y);
	// std::cout << "yTy : " << yTy << std::endl;

	/* ------------------ enable when generating an example in c++ --------------- */
	// actual solution 
	/*double tau = 1.0;

	Eigen::MatrixXd B(no, nb);
	Vector b(nb);
	Vector y(no);

	generate_ex_regression(nb, no, tau, &B, &b, &y);
	std::cout << "original fixed effects : " << b.transpose() << std::endl;*/

	/* ---------------------- initialise BFGS ------------------- */

	double max_iter_BFGS = 20;

	double alpha_0 = 1.0;
	double c = 1e-4;
	double rho = 0.9;
	double alpha;

	// returns value and estimate of fixed effects mu
	double f_theta;
	Vector mu(nb);
	Vector mu_dummy(nb);

	double theta_old;
	double theta_update;
	double f_theta_old;
	double grad_old;
	double grad_update;

	eval_post_theta(nb, no, theta, yTy, y, B, &f_theta, &mu);
	std::cout << "f(theta) : " << f_theta << std::endl;

	double grad;
	eval_gradient(nb, no, theta, f_theta, yTy, y, B, &mu_dummy, &grad);
	// std::cout << "grad f : " << grad << std::endl;

	double invHess = 1;

	for(int i = 0; i < max_iter_BFGS; i++){

		double p = -invHess*grad;

		alpha = backtracking(nb, no, theta, f_theta, grad, p, yTy, y, B, alpha_0, c, rho);

		if(alpha == -1){
			std::cout << "max_iter reached in backtracking. exited function call. " << std::endl;
			std::cout << "final theta : " << theta << std::endl;
			std::cout << "mu 		  : " << mu.transpose() << std::endl;
			std::cout << "gradient f  : " << grad << std::endl;
			exit(1);
		}

		// update x
	    theta_old = theta;
	    theta = theta + alpha*p;
	    std::cout << "theta    : " << theta << std::endl;
	    theta_update = theta - theta_old;

	    // evaluate f
	    f_theta_old = f_theta;
	    eval_post_theta(nb, no, theta, yTy, y, B, &f_theta, &mu);
		std::cout << "f(theta) : " << f_theta << std::endl;
	    
	    // evaluate gradient
	    grad_old = grad;
		eval_gradient(nb, no, theta, f_theta, yTy, y, B, &mu_dummy, &grad);
		//std::cout << "grad f : " << grad << std::endl;
		grad_update = grad - grad_old;  	
    	
    	if(std::abs(grad) < 1e-2){
    		std::cout << "BFGS converged after " << i << " iterations. arg min : " << theta << std::endl;
    		std::cout << "mu : " << mu.transpose() << std::endl;

      	}
       
	    eval_invHess(theta_update, grad_update, &invHess);
		//std::cout << "invHess   : " << invHess << std::endl;

	}

	std::cout << "final theta : " << theta << std::endl;
	std::cout << "mu 		  : " << mu.transpose() << std::endl;
	std::cout << "gradient f  : " << grad << std::endl;

  	return 0;

}
