// function evaluation regression model 

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>


using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::CholmodSimplicialLDLT  <SpMat > Solver;
typedef Eigen::VectorXd Vector;


/*
orgianise class such that function call will return :

- function value for given theta
- update gradient for theta (using forward difference)


*/

class post_theta{

	private:
    int no;
    int nb;
    MatrixXd B;
    VectorXd y;
    double yTy;
    Vector mu;
    Vector* t_grad;

public:
	post_theta(int no_, int nb_, MatrixXd B_, VectorXd y_) : no(no_), nb(nb_), B(B_), y(y_) {
		yTy = y.dot(y);
	}
    double operator()(Vector& theta, Vector& grad){
	//void eval_post_theta(int nb, int no, double theta, double yTy, Vector y, Eigen::MatrixXd B, double *val, Vector *mu){
    	*t_grad = grad;
    	
    	// Vector mu;
    	double f_theta = eval_post_theta(theta, mu);
    	std::cout << "theta   : " << theta << ", f_theta : " << f_theta << std::endl;

    	Vector mu_dummy;
		eval_gradient(theta, f_theta, mu_dummy, grad);
		// std::cout << "grad : " << grad.transpose() << std::endl;


    	return f_theta;
	}

	Vector get_mu(){
		return mu;
	}

	Vector get_grad(){
		return *t_grad;
	}

	double eval_post_theta(Vector& theta, Vector& mu){

		//Vector mu;
		// numerator :
		// log_det(Q.u) doesnt exist here. 
		// eval_likelihood: log_det, -theta*yTy
		double log_det_l;
		double val_l; 
		eval_likelihood(theta, &log_det_l, &val_l);
		/*std::cout << "log det l : "  << log_det_l << std::endl;
		std::cout << "val l     : " << val_l << std::endl;*/

		// denominator :
		// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
		double log_det_d;
		double val_d;
		SpMat Q(nb, nb);
		Vector rhs(nb);

	 	eval_denominator(theta, &log_det_d, &val_d, &Q, &rhs, mu);
		/*std::cout << "log det d : " << log_det_d << std::endl;
		std::cout << "val d     : " <<  val_d << std::endl;*/

	  	// add everything together
	  	double val = -1 * 0.5 * (log_det_l - val_l - log_det_d + val_d);
	  	// std::cout << *val << std::endl;

	  	return val;
	}

	void solve_cholmod(SpMat *A, Vector *f, Vector& u, double *log_det)
	{

		Solver solver;
		solver.analyzePattern(*A);
		solver.factorize(*A);

		u = solver.solve(*f);

		*log_det = solver.logDeterminant();

		//std::cout << "solution vector u : " << *u << std::endl;

	}

	void eval_likelihood(Vector& theta, double *log_det, double *val){
		

		*log_det = no*theta[0];
		*val = exp(theta[0])*yTy;

		/*std::cout << "in eval eval_likelihood " << std::endl;
		std::cout << "theta     : " << theta << std::endl;
		std::cout << "yTy : " << yTy << ", exp(theta) : " << exp(theta) << std::endl;

		std::cout << "log det l : " << *log_det << std::endl;
		std::cout << "val l     : " << *val << std::endl; */

	}

	void construct_Q_b(Vector& theta, SpMat *Q, Vector *rhs){
		double exp_theta = exp(theta[0]);

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


	void eval_denominator(Vector& theta, double *log_det, double *val, SpMat *Q, Vector *rhs, Vector& mu){

		// construct Q_x|y, b_xey
		construct_Q_b(theta, Q, rhs);

		// solve linear system
		solve_cholmod(Q, rhs, mu, log_det);

		// compute value
		*val = mu.transpose()*(*Q)*(mu); 

		/*std::cout << "in eval eval_denominator " << std::endl;

		std::cout << "rhs " << std::endl; std::cout <<  *rhs << std::endl;
		std::cout << "mu " << std::endl; std::cout << *mu << std::endl;
		std::cout << "Q " << std::endl; std::cout << Eigen::MatrixXd(*Q) << std::endl;


		std::cout << "log det d : " << *log_det << std::endl;
		std::cout << "val d     : " << *val << std::endl; */
	}


	// 1D version, forward difference
	void eval_gradient(Vector& theta, double f_theta, Vector& mu, Vector& grad){

		Vector eps(1);
		eps[0] = 0.05;
		Vector theta_forw(1);
		theta_forw = theta + eps;

		double f_theta_forw;

		// have to be careful that mu doesn't get overwritten?
		// write version such that this doesn't happen?
		f_theta_forw = eval_post_theta(theta_forw, mu);
		/*std::cout << "theta_forw   : " << theta_forw << std::endl;
		std::cout << "f_theta_forw : " << f_theta_forw << std::endl;*/

		grad[0] = (1.0/eps[0]) * (f_theta_forw - f_theta);
		// std::cout << "grad : " << grad.transpose() << std::endl;
	}

};