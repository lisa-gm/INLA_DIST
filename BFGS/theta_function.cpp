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

#include "solver.cpp"
// #include "theta_function.hpp"

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

class PostTheta{

	private:
	int ns;
    int nb;
    int no;
    int nu;
    MatrixXd B;
    VectorXd y;
    // next 4 potentially unused (then B unused)
    SpMat Ax;
    SpMat c0;
    SpMat g1;
    SpMat g2;
    double yTy;
    Vector mu;
    Vector t_grad;
    double min_f_theta;

public:
	PostTheta(int nb_, int no_, MatrixXd B_, VectorXd y_) : nb(nb_), no(no_), B(B_), y(y_) {
		ns = 0;
		yTy = y.dot(y);

		// initialise min_f_theta, min_theta
		min_f_theta = 1e10;
	}
	PostTheta(int ns_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_) : ns(ns_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_)  {
		yTy = y.dot(y);
		nu = nb + ns;

		// initialise min_f_theta, min_theta
		min_f_theta = 1e10;
	}

    double operator()(Vector& theta, Vector& grad){

    	t_grad = grad;

    	// initialise min_f_theta, min_theta, store current minimum 

    	// Vector mu;
    	double f_theta = eval_post_theta(theta, mu);


    	if(f_theta < min_f_theta){
    		min_f_theta = f_theta;
    		std::cout << "theta   : " << theta.transpose() << ", f_theta : " << f_theta << std::endl;
    	}

    	Vector mu_dummy;
		eval_gradient(theta, f_theta, mu_dummy, grad);
		// std::cout << "grad : " << grad.transpose() << std::endl;


    	return f_theta;
	}

	Vector get_mu(){
		return mu;
	}

	Vector get_grad(){
		return t_grad;
	}

	MatrixXd get_Covariance_hyperparam(Vector& theta){

		// construct 2nd order cetnral difference
		Vector eps(1);
		eps[0] = 0.005;

		Vector theta_forw(1);
		theta_forw = theta + eps;

		Vector theta_back(1);
		theta_back = theta - eps;

		Vector mu_dummy(1);

		double f_theta = eval_post_theta(theta, mu);
		double f_theta_forw = eval_post_theta(theta_forw, mu_dummy);
		double f_theta_back = eval_post_theta(theta_back, mu_dummy);

		MatrixXd hess(1,1);
		// careful : require negative hessian (swapped signs in eval post theta) 
		// but then precision negative hessian -> no swapping
		hess << (1.0)/(eps[0]*eps[0]) * (f_theta_forw - 2*f_theta + f_theta_back);
		// std::cout << "hess " << hess << std::endl;

		MatrixXd cov(1,1);
		cov << 1.0/hess(0,0);

		return cov;
	}

	MatrixXd get_Covariance(Vector& theta){

		// TODO: multivariate Hessian approximation!

		/*int dim_th = theta.size();
		double eps = 0.005;
		MatrixXd epsId_mat(dim_th, dim_th); 
		epsId_mat = eps*epsId_mat.setIdentity();
		//std::cout << "epsId_mat : " << epsId_mat << std::endl;

		// temp vector
		Vector theta_forw(dim_th);
		Vector theta_backw(dim_th);

		Vector f_forw(dim_th);
		Vector f_backw(dim_th);

		Vector mu_dummy(dim_th);

		// compute final f(theta) 
		double f_theta = eval_post_theta(theta, mu);


		// parallelise loop later
		for(int i=0; i<dim_th; i++){
			theta_forw = theta + epsId_mat.col(i);
			theta_backw = theta - epsId_mat.col(i);

			f_forw[i] = eval_post_theta(theta_forw, mu_dummy);
			f_backw[i] = eval_post_theta(theta_backw, mu_dummy);

		} 


		// compute finite difference in each direction
		grad = 1.0/(2.0*eps)*(f_forw - f_backw);
		// std::cout << "grad  : " << grad << std::endl;*/

		// construct 2nd order cetnral difference
		Vector eps(1);
		eps[0] = 0.005;

		Vector theta_forw(1);
		theta_forw = theta + eps;

		Vector theta_back(1);
		theta_back = theta - eps;

		Vector mu_dummy(1);

		double f_theta = eval_post_theta(theta, mu);
		double f_theta_forw = eval_post_theta(theta_forw, mu_dummy);
		double f_theta_back = eval_post_theta(theta_back, mu_dummy);

		MatrixXd hess(1,1);
		// careful : require negative hessian (swapped signs in eval post theta) 
		// but then precision negative hessian -> no swapping
		hess << (1.0)/(eps[0]*eps[0]) * (f_theta_forw - 2*f_theta + f_theta_back);
		// std::cout << "hess " << hess << std::endl;

		MatrixXd cov(1,1);
		cov << 1.0/hess(0,0);

		return cov;
	}


	Vector get_marginals_f(Vector& theta){
		
		SpMat Q(nb, nb);
		construct_Q(theta, &Q);

		Vector vars(nb);
		extract_inv_diag(Q, vars);

		return(vars);
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
	  	//std::cout << val << std::endl;

	  	return val;
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

	
	// SPDE discretisation -- matrix construction
	void construct_Q_spatial(Vector& theta, SpMat* Qs){
		// Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
		// return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
		*Qs = pow(theta[1],2)*(pow(theta[2], 4) * c0 + 2*pow(theta[2],2) * g1 + g2);
		// extract triplet indices and insert into Qx


	}

	void construct_Q(Vector& theta, SpMat *Q){
		double exp_theta = exp(theta[0]);

		SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
		/*std::cout << "Q_b " << std::endl;
		std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/
		//Q_b = 1e-5*Q_b.setIdentity();

		if(ns > 0){
			SpMat Qs(ns, ns);
			// TODO: find good way to assemble Qx
			construct_Q_spatial(theta, &Qs);
			//Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
			// construct Qx from Qs values, extend by zeros 
			SpMat Qx(nu,nu);         // default is column major			

			int nnz = Qs.nonZeros();
			Qx.reserve(nnz);

			for (int k=0; k<Qs.outerSize(); ++k)
			  for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
			  {
			    Qx.insert(it.row(),it.col()) = it.value();                 
			  }

			//Qs.makeCompressed();
			//SpMat Qx = Map<SparseMatrix<double> >(ns+nb,ns+nb,nnz,Qs.outerIndexPtr(), // read-write
            //                   Qs.innerIndexPtr(),Qs.valuePtr());

			for(int i=ns; i<(ns+nb); i++){
				Qx.coeffRef(i,i) = 1e-5;
			}

			Qx.makeCompressed();

			//std::cout << "Qx  " << Qx << std::endl;

			//std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
			*Q =  Qx + exp_theta * Ax.transpose() * Ax;
		}

		if(ns == 0){
			// Q.e <- Diagonal(no, exp(theta))
			// Q.xy <- Q.x + crossprod(A.x, Q.e)%*%A.x  # crossprod = t(A)*Q.e (faster)	
			*Q = Q_b + exp_theta*B.transpose()*B;
		}

		/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
		std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

	}


	void construct_b(Vector& theta, Vector *rhs){
		double exp_theta = exp(theta[0]);

		if(ns == 0){
			*rhs = exp_theta*B.transpose()*y;
		} else {
			*rhs = exp_theta*Ax.transpose()*y;
		}

		// exp(theta)*yTy

		// 1e-5*Id + exp(theta)*B'*B

		// mu*Q*mu != exp(theta)*yTy
		// mu*Q*mu = (Q^-1*(exp(theta)*B'*y*(exp(theta)*B'*y)
		// exp(theta)*Q^-1*exp(theta)*B'*B*y'*y. now ignore Q_b part of Q. then Q = exp(theta)*B'*B, ie.
		// exp(theta)*y'*y remains ... 
	}


	void eval_denominator(Vector& theta, double *log_det, double *val, SpMat *Q, Vector *rhs, Vector& mu){

		// construct Q_x|y,
		construct_Q(theta, Q);

		//  construct b_xey
		construct_b(theta, rhs);

		// solve linear system
		// returns vector mu, which is of the same size as rhs
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


	void eval_gradient(Vector& theta, double f_theta, Vector& mu, Vector& grad){

		int dim_th = theta.size();

		double eps = 0.005;
		MatrixXd epsId_mat(dim_th, dim_th); 
		epsId_mat = eps*epsId_mat.setIdentity();
		//std::cout << "epsId_mat : " << epsId_mat << std::endl;

		// temp vector
		Vector theta_forw(dim_th);
		Vector theta_backw(dim_th);

		Vector f_forw(dim_th);
		Vector f_backw(dim_th);

		Vector mu_dummy;

		// parallelise loop later
		for(int i=0; i<dim_th; i++){
			theta_forw = theta + epsId_mat.col(i);
			theta_backw = theta - epsId_mat.col(i);

			f_forw[i] = eval_post_theta(theta_forw, mu_dummy);
			f_backw[i] = eval_post_theta(theta_backw, mu_dummy);

		}


		// compute finite difference in each direction
		grad = 1.0/(2.0*eps)*(f_forw - f_backw);
		// std::cout << "grad  : " << grad << std::endl;

	}

};


// for Hessian approximation : 4-point stencil (2nd order)
// -> swap sign, invert, get covariance

// once converged call again : extract -> Q.xy -> selected inverse (diagonal), gives me variance wrt mode theta & data y   