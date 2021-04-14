// function evaluation regression model 


#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>


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
    X.print();

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
	std::cout << *B << std::endl;

	/* -------  construct random solution vector of fixed effects & observations -------- */
	*b = 2*(Vector::Random(nb) + Vector::Ones(nb)); 
	std::cout << "original fixed effects : " << b->transpose() << std::endl;

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
	std::cout << "yTy : " << yTy << ", exp(theta) : " << exp(theta) << std::endl;

	std::cout << "log det l : " << *log_det << std::endl;
	std::cout << "val l     : " << *val << std::endl;*/

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
	std::cout << "log det l : "  << log_det_l << std::endl;
	std::cout << "val l     : " << val_l << std::endl;

	// denominator :
	// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
	double log_det_d;
	double val_d;
	SpMat Q(nb, nb);
	Vector rhs(nb);

 	eval_denominator(nb, no, theta, y, B, &log_det_d, &val_d, &Q, &rhs, mu);
	std::cout << "log det d : " << log_det_d << std::endl;
	std::cout << "val d     : " <<  val_d << std::endl;

  	// add everything together
  	*val = -1 * 0.5 * (log_det_l - val_l - log_det_d + val_d);
  	// std::cout << *val << std::endl;

}


int main(int argc, char* argv[]) 
{ 
	if (argc != 1 + 3) {
		std::cerr << "simple regression model : nb no theta " << std::endl;

		std::cerr << "[integer]: number of observations (nb)" << std::endl;     
		std::cerr << "[integer]: number of observations (no)" << std::endl;  
		std::cerr << "[double] : initial value for theta    " << std::endl;    

		exit(1);
	}

	int nb = atoi(argv[1]);
	int no = atoi(argv[2]);

	// initial value 
	int theta = atoi(argv[3]);
	// actual solution 
	double tau = 1.0;

	Eigen::MatrixXd B(no, nb);
	Vector b(nb);
	Vector y(no);

	generate_ex_regression(nb, no, tau, &B, &b, &y);

	exit(1);
	double yTy = y.dot(y);

	/* ---------------------- call function evaluation ------------------- */
	// returns value and estimate of fixed effects mu
	double val;
	Vector mu(nb);

	eval_post_theta(nb, no, theta, yTy, y, B, &val, &mu);
	std::cout << val << std::endl;


  return 0;

}
