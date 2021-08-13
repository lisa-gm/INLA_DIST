#ifndef __PostTheta
#define __PostTheta

#if 1

#include "PostTheta.h"


PostTheta::PostTheta(int ns, int nt, int nb, int no, MatrixXd B, VectorXd y, Vector theta_prior){
	
	dim_th = 1;  			// only hyperparameter is the precision of the observations
	ns     = 0;
	n      = nb;
	yTy    = y.dot(y);

	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
	#endif

	min_f_theta        = 1e10;			// initialise min_f_theta, min_theta

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	printf("threads level 1 : %d\n", threads_level1);

	dim_grad_loop      = 2*dim_th;

	// one solver per thread, but not more than required
	num_solvers        = std::min(threads_level1, dim_grad_loop);
	printf("num solvers     : %d\n", num_solvers);

	solverQst          = new PardisoSolver[num_solvers];
	solverQ            = new PardisoSolver[num_solvers];

	// set global counter to count function evaluations
	fct_count          = 0;	// initialise min_f_theta, min_theta
}


PostTheta::PostTheta(int ns, int nt, int nb, int no, SpMat Ax, VectorXd y, SpMat c0, SpMat g1, SpMat g2, Vector theta_prior){

	dim_th      = 3;   			// 3 hyperparameters, precision for the observations, 2 for the spatial model
	nu          = ns;
	n           = nb + ns;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta
	yTy         = y.dot(y);

	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
	#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	printf("threads level 1 : %d\n", threads_level1);

	dim_grad_loop      = 2*dim_th;

	// one solver per thread, but not more than required
	num_solvers        = std::min(threads_level1, dim_grad_loop);
	printf("num solvers     : %d\n", num_solvers);

	solverQst          = new PardisoSolver[num_solvers];
	solverQ            = new PardisoSolver[num_solvers];

	// set global counter to count function evaluations
	fct_count          = 0;
}


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vector _theta_prior) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior(_theta_prior)  {

	dim_th      = 4;    	 	// 4 hyperparameters, precision for the observations, 3 for the spatial-temporal model
	nu          = ns*nt;
	n           = nb + ns*nt;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta
	yTy         = y.dot(y);

	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
	#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	printf("threads level 1 : %d\n", threads_level1);

	dim_grad_loop      = 2*dim_th;

	// one solver per thread, but not more than required
	num_solvers        = std::min(threads_level1, dim_grad_loop);
	printf("num solvers     : %d\n", num_solvers);

	solverQst          = new PardisoSolver[num_solvers];
	solverQ            = new PardisoSolver[num_solvers];

	// set global counter to count function evaluations
	fct_count          = 0;
}


double PostTheta::operator()(Vector& theta, Vector& grad){

	t_grad = grad;

	// initialise min_f_theta, min_theta, store current minimum 
	// Vector mu;
	mu.setZero(n);

	double f_theta = eval_post_theta(theta, mu);

	if(f_theta < min_f_theta){
		min_f_theta = f_theta;

		std::cout << "theta : " << std::right << std::fixed << theta.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << std::endl;
		//std::cout << "theta   : " << theta.transpose() << ", f_theta : " << f_theta << std::endl;
	}

	double timespent_grad = -omp_get_wtime();

	Vector mu_dummy(n);
	eval_gradient(theta, f_theta, mu_dummy, grad);
	// std::cout << "grad : " << grad.transpose() << std::endl;

	timespent_grad += omp_get_wtime();

	#ifdef PRINT_MSG
		std::cout << "time spent gradient call : " << timespent_grad << std::endl;
	#endif

	return f_theta;
}

// ============================================================================================ //
// CONVERT MODEL PARAMETRISATION TO INTERPRETABLE PARAMETRISATION & VICE VERSA

void PostTheta::convert_theta2interpret(double lgamE, double lgamS, double lgamT, double& sigU, double& ranS, double& ranT){
	double alpha_t = 1; 
	double alpha_s = 2;
	double alpha_e = 1;

	double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
	double nu_s  = alpha   - 1;
	double nu_t  = alpha_t - 0.5;

	double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*8*pow(M_PI,1.5));
	double gE = exp(lgamE);
	double gS = exp(lgamS);
	double gT = exp(lgamT);

	sigU = log(sqrt(c1)/((gE*sqrt(gT))*pow(gS,alpha-1)));
	ranS = log(sqrt(8*nu_s)/gS);
	ranT = log(gT*sqrt(8*nu_t)/(pow(gS, alpha_s)));
}


void PostTheta::convert_interpret2theta(double sigU, double ranS, double ranT, double& lgamE, double& lgamS, double& lgamT){
	double alpha_t = 1; 
	double alpha_s = 2;
	double alpha_e = 1;

	double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
	double nu_s  = alpha   - 1;
	double nu_t  = alpha_t - 0.5;

	double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*8*pow(M_PI,1.5));
	lgamS = 0.5*log(8*nu_s) - ranS;
	lgamT = ranT - 0.5*log(8*nu_t) + alpha_s * lgamS;
	lgamE = 0.5*log(c1) - 0.5*lgamT - nu_s*lgamS - sigU;
}

// ============================================================================================ //
// FUNCTIONS TO BE CALLED AFTER THE BFGS SOLVER CONVERGED


void PostTheta::get_mu(Vector& theta, Vector& mu){

	#ifdef PRINT_MSG
		std::cout << "get_mu()" << std::endl;
	#endif

	double f_theta = eval_post_theta(theta, mu);

	#ifdef PRINT_MSG
		std::cout << "mu(-10:end) :" << mu.tail(10) << std::endl;
	#endif
}

Vector PostTheta::get_grad(){
	return t_grad;
}


MatrixXd PostTheta::get_Covariance(Vector& theta){

	int dim_th = theta.size();
	MatrixXd hess(dim_th,dim_th);

	// evaluate hessian
	double timespent_hess_eval = -omp_get_wtime();
	hess = hess_eval(theta);

	timespent_hess_eval += omp_get_wtime();
	std::cout << "time spent hessian evaluation: " << timespent_hess_eval << std::endl; 

	//std::cout << "hess : " << hess << std::endl; 

	MatrixXd cov(dim_th,dim_th);

	std::cout << "cholmod excluded for now! " << std::endl;
	//compute_inverse_cholmod(hess, cov);

	//std::cout << "cov_new : " << cov << std::endl; 

	return cov;
}


void PostTheta::get_marginals_f(Vector& theta, Vector& vars){
	
	SpMat Q(n, n);
	construct_Q(theta, Q);

	#ifdef PRINT_MSG
		std::cout << "after construct Q in get get_marginals_f" << std::endl;
	#endif

	double timespent_sel_inv_pardiso = -omp_get_wtime();
	int tid = omp_get_thread_num();
	solverQ[tid].selected_inversion(Q, vars);
	timespent_sel_inv_pardiso += omp_get_wtime();	

	std::cout << "time spent selected inversion pardiso : " << timespent_sel_inv_pardiso << std::endl; 
}


MatrixXd PostTheta::hess_eval(Vector& theta){

	double eps = 0.005;

	int dim_th = theta.size();
	MatrixXd epsId(dim_th, dim_th); 
	epsId = eps*epsId.setIdentity();

	MatrixXd hessUpper = MatrixXd::Zero(dim_th, dim_th);

	// compute upper tridiagonal structure

	#pragma omp parallel for 
	for(int i=0; i < dim_th; i++){
		for(int j=i; j < dim_th; j++){

			// dummy mu
			Vector mu_tmp(n);

			if(i == j){
				Vector theta_forw_i = theta+epsId.col(i);
				Vector theta_back_i = theta-epsId.col(i);

				hessUpper(i,i) = (eval_post_theta(theta_forw_i, mu_tmp) - 2 * eval_post_theta(theta, mu_tmp) + eval_post_theta(theta_back_i, mu_tmp))/(eps*eps);

			} else {
				Vector theta_forw_i_j 		= theta+epsId.col(i)+epsId.col(j);
				Vector theta_forw_i_back_j  = theta+epsId.col(i)-epsId.col(j);
				Vector theta_back_i_forw_j  = theta-epsId.col(i)+epsId.col(j);
				Vector theta_back_i_j 		= theta-epsId.col(i)-epsId.col(j);

    			hessUpper(i,j) = (  eval_post_theta(theta_forw_i_j, mu_tmp) \
                       - eval_post_theta(theta_forw_i_back_j, mu_tmp) - eval_post_theta(theta_back_i_forw_j, mu_tmp) \
                       + eval_post_theta(theta_back_i_j, mu_tmp)) / (4*eps*eps);
       		}
		}
	}

	//	std::cout << "hess upper : " << hessUpper << std::endl;
	MatrixXd hess = hessUpper.selfadjointView<Upper>();
	std::cout << "hessian       : \n" << hess << std::endl;

	// check that matrix positive definite otherwise use only diagonal
	check_pos_def(hess); 

	return hess;
}


 void PostTheta::check_pos_def(MatrixXd &hess){

	// compute Eigenvalues of symmetric matrix
	SelfAdjointEigenSolver<MatrixXd> es(hess);
	/*cout << "The eigenvalues of g1 are:" << endl << es.eigenvalues() << endl;
	cout << "The min eigenvalue of g1 is:" << endl << es.eigenvalues().minCoeff() << endl;*/

	// check if there are eigenvalues smaller or equal to zero
	if((es.eigenvalues().minCoeff()) <= 0){

		std::cout << "Matrix not positive definite only considering diagonal values!! " << std::endl;
		Vector diag_hess = hess.diagonal();
		hess = diag_hess.asDiagonal();
    	std::cout << "new hessian :\n" << hess << std::endl;
	} else {
		#ifdef PRINT_MSG
			std::cout << "Hessian is positive definite.";
		#endif
	}
}

// ============================================================================================ //
// ALL FOLLOWING FUNCTIONS CONTRIBUT TO THE EVALUATION OF F(THETA) & GRADIENT


double PostTheta::eval_post_theta(Vector& theta, Vector& mu){			

	// =============== set up ================= //
	int dim_th = theta.size();

	#ifdef PRINT_MSG
		std::cout << "in eval post theta function. " << std::endl;
		std::cout << "dim_th : " << dim_th << std::endl;
		std::cout << "nt : " << nt << std::endl;			
		std::cout << "theta prior : " << theta_prior.transpose() << std::endl;
	#endif

	// =============== evaluate theta prior based on original solution & variance = 1 ================= //

	//VectorXd zero_vec(dim_th); zero_vec.setZero();
	VectorXd zero_vec(theta_prior.size()); zero_vec.setZero();

	double log_prior_sum = 0;

	// evaluate prior
	VectorXd log_prior_vec(dim_th);
	for( int i=0; i<dim_th; i++ ){
		eval_log_prior(log_prior_vec[i], &theta[i], &theta_prior[i]);
	}
    
	log_prior_sum = log_prior_vec.sum();

	#ifdef PRINT_MSG
		std::cout << "log prior sum : " << log_prior_sum << std::endl;
	#endif

		// =============== evaluate prior of random effects : need log determinant ================= //

	// requires factorisation of Q.u -> can this be done in parallel with the 
	// factorisation of the denominator? 
	// How long does the assembly of Qu take? Should this be passed on to the 
	// denominator to be reused?

	double log_det_Qu = 0;

	if(ns > 0 ){
		eval_log_det_Qu(theta, log_det_Qu);
	}

	#ifdef PRINT_MSG
		std::cout << "log det Qu : "  << log_det_Qu << std::endl;
	#endif

		// =============== evaluate likelihood ================= //

	// eval_likelihood: log_det, -theta*yTy
	double log_det_l;
	double val_l; 
	eval_likelihood(theta, log_det_l, val_l);

	#ifdef PRINT_MSG
		std::cout << "log det l : "  << log_det_l << std::endl;
		std::cout << "val l     : " << val_l << std::endl;
	#endif

		// =============== evaluate denominator ================= //
	// denominator :
	// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
	double log_det_d;
	double val_d;
	SpMat Q(n, n);
	Vector rhs(n);

 	eval_denominator(theta, log_det_d, val_d, Q, rhs, mu);
	#ifdef PRINT_MSG
		std::cout << "log det d : " << log_det_d << std::endl;
		std::cout << "val d     : " <<  val_d << std::endl;
	#endif

		// =============== add everything together ================= //
  	double val = -1 * (log_prior_sum + log_det_Qu + log_det_l + val_l - (log_det_d + val_d));

  	#ifdef PRINT_MSG
  		std::cout << "f theta : " << val << std::endl;
  	#endif

  	return val;
}


void PostTheta::eval_log_prior(double& log_prior, double* thetai, double* thetai_original){

	log_prior = -0.5 * (*thetai - *thetai_original) * (*thetai - *thetai_original);

	#ifdef PRINT_MSG
		std::cout << "log prior for theta_i " << (*thetai) << " : " << (log_prior) << std::endl;
	#endif
}


void PostTheta::eval_log_det_Qu(Vector& theta, double &log_det){

	SpMat Qu(nu, nu);
	if(nt > 1){
		construct_Q_spat_temp(theta, Qu);
	} else {
		construct_Q_spatial(theta, Qu);
	}

	int tid = omp_get_thread_num();
	solverQst[tid].factorize(Qu, log_det);

	#ifdef PRINT_MSG
		std::cout << "log det Qu : " << log_det << std::endl;
	#endif

	log_det = 0.5 * (log_det);
}


void PostTheta::eval_likelihood(Vector& theta, double &log_det, double &val){
	
	// multiply log det by 0.5
	log_det = 0.5 * no*theta[0];
	//log_det = 0.5 * no*3;

	// - 1/2 ...
	val = - 0.5 * exp(theta[0])*yTy;
	//*val = - 0.5 * exp(3)*yTy;

	/*std::cout << "in eval eval_likelihood " << std::endl;
	std::cout << "theta     : " << theta << std::endl;
	std::cout << "yTy : " << yTy << ", exp(theta) : " << exp(theta) << std::endl;
	std::cout << "log det l : " << log_det << std::endl;
	std::cout << "val l     : " << val << std::endl; */
}


void PostTheta::construct_Q_spatial(Vector& theta, SpMat& Qs){

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


void PostTheta::construct_Q_spat_temp(Vector& theta, SpMat& Qst){

	double exp_theta1 = exp(theta[1]);
	double exp_theta2 = exp(theta[2]);
	double exp_theta3 = exp(theta[3]);

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

		//std::cout << "Qst : \n" << Qst->block(0,0,10,10) << std::endl;
}


void PostTheta::construct_Q(Vector& theta, SpMat& Q){

	double exp_theta0 = exp(theta[0]);
	//double exp_theta = exp(3);

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

	if(ns > 0){
		SpMat Qu(nu, nu);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(theta, Qu);
		} else {	
			construct_Q_spatial(theta, Qu);
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

		Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

		#ifdef PRINT_MSG
			std::cout << "exp(theta0) : \n" << exp_theta0 << std::endl;
			std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
		#endif
	}

	if(ns == 0){
		// Q.e <- Diagonal(no, exp(theta))
		// Q.xy <- Q.x + crossprod(A.x, Q.e)%*%A.x  # crossprod = t(A)*Q.e (faster)	
		Q = Q_b + exp_theta0*B.transpose()*B;
	}

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

	#ifdef PRINT_MSG
		std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
		std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
		std::cout << "theta : \n" << theta.transpose() << std::endl;

	#endif 

}


void PostTheta::construct_b(Vector& theta, Vector &rhs){

	double exp_theta = exp(theta[0]);
	//double exp_theta = exp(3);

	if(ns == 0){
		rhs = exp_theta*B.transpose()*y;
	} else {
		rhs = exp_theta*Ax.transpose()*y;
	}
}


void PostTheta::eval_denominator(Vector& theta, double& log_det, double& val, SpMat& Q, Vector& rhs, Vector& mu){

	// construct Q_x|y,
	construct_Q(theta, Q);
	//Q->setIdentity();

	#ifdef PRINT_MSG
		printf("\nin eval denominator after construct_Q call.");
	#endif

	//  construct b_xey
	construct_b(theta, rhs);

	#ifdef PRINT_MSG
		printf("\nin eval denominator after construct_b call.");
	#endif

	// solve linear system
	// returns vector mu, which is of the same size as rhs
	//solve_cholmod(Q, rhs, mu, log_det);

	int tid = omp_get_thread_num();
	solverQ[tid].factorize_solve(Q, rhs, mu, log_det);

	log_det = 0.5 * (log_det);
	
	#ifdef PRINT_MSG
		std::cout << "log det d : " << log_det << std::endl;
	#endif

	// compute value
	val = -0.5 * mu.transpose()*(Q)*(mu); 

	/*std::cout << "in eval eval_denominator " << std::endl;

	std::cout << "rhs " << std::endl; std::cout <<  *rhs << std::endl;
	std::cout << "mu " << std::endl; std::cout << *mu << std::endl;
	std::cout << "Q " << std::endl; std::cout << Eigen::MatrixXd(*Q) << std::endl;

	std::cout << "log det d : " << log_det << std::endl;
	std::cout << "val d     : " << val << std::endl; */
}

// ============================================================================================ //
// FINITE DIFFERENCE GRADIENT EVALUATION


void PostTheta::eval_gradient(Vector& theta, double f_theta, Vector& mu, Vector& grad){

	#ifdef PRINT_MSG
		std::cout << "\nin eval gradient function." << std::endl;
	#endif

	int dim_th = theta.size();

	double eps = 0.005;
	MatrixXd epsId_mat(dim_th, dim_th); 
	epsId_mat = eps*epsId_mat.setIdentity();
	//std::cout << "epsId_mat : " << epsId_mat << std::endl;

	Vector f_forw(dim_th);
	Vector f_backw(dim_th);

	int threads = omp_get_max_threads();

	// naively parallelise using OpenMP, more later
	#pragma omp parallel for
	for(int i=0; i<2*dim_th; i++){

			if(i % 2 == 0){
				int k = i/2;

				#ifdef PRINT_MSG
					std::cout << "forward loop thread rank: " << omp_get_thread_num() << " out of " << threads << std::endl;
					std::cout << "i : " << i << " and k : " << k << std::endl;
				#endif

				// temp vector
				Vector theta_forw(dim_th);
				Vector mu_dummy(n);

				theta_forw = theta + epsId_mat.col(k);
				f_forw[k] = eval_post_theta(theta_forw, mu_dummy);

			} else {
				int k = (i-1)/2;

				#ifdef PRINT_MSG
					std::cout << "backward loop thread rank: " << omp_get_thread_num() << " out of " << threads << std::endl;
					std::cout << "i : " << i << " and k : " << k << std::endl;
				#endif

				// temp vector
				Vector theta_backw(dim_th);
				Vector mu_dummy(n);

				theta_backw = theta - epsId_mat.col(k);
				f_backw[k] = eval_post_theta(theta_backw, mu_dummy);
			}
	}

	// compute finite difference in each direction
	grad = 1.0/(2.0*eps)*(f_forw - f_backw);
	// std::cout << "grad  : " << grad << std::endl;

}


PostTheta::~PostTheta(){

		delete[] solverQst;
		delete[] solverQ;		
}


#endif

// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //


#if 0

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

// std::setwd print out
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/KroneckerProduct>

//#include "solver_cholmod.cpp"
//#include "solver_pardiso.cpp"
#include "PardisoSolver.cpp"

//#define PRINT_MSG

using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::CholmodSimplicialLDLT  <SpMat > Solver;
typedef Eigen::VectorXd Vector;

/**
* @brief Computes the Posterior of the hyperparameters theta. 
* @details Computes the posterior of theta for a given theta and its gradient using a central finite difference approximation. Can additionally compute an approximation to the Hessian. 
*/
class PostTheta{

private:
int ns;				/**<  number of spatial grid points per timestep 	*/
int nt;				/**<  number of temporal time steps 				*/
int nb;				/**<  number of fixed effects 						*/
int no;				/**<  number of observations 						*/
int nu;				/**<  number of random effects, that ns*nu 			*/
int n;				/**<  total number of unknowns, i.e. ns*nt + nb 	*/

int dim_th;			/**<  dimension of hyperparameter vector theta 		*/
int threads_level1; /**<  number of threads on first level 				*/
int dim_grad_loop;  /**<  dimension of gradient loop 					*/
int num_solvers;    /**<  number of pardiso solvers 					*/

PardisoSolver* solverQ;   /**<  list of Pardiso solvers, for denom.		*/
PardisoSolver* solverQst; /**<  list of Pardiso solvers, for Qu         */

int fct_count;      /**< count total number of function evaluations 	*/

VectorXd y; 		/**<  vector of observations y. has length no. 		*/
Vector theta_prior; /**<  vector with prior values. Constructs normal
						      distribution with sd = 1 around these values. */

// either Ax or B used
SpMat Ax;			/**< sparse matrix of size no x (nu+nb). Projects 
						 observation locations onto FEM mesh and 
						 includes covariates at the end.                */
MatrixXd B; 		/**< if space (-time) model included in last 
						 columns of Ax. For regression only B exists.   */

// used in spatial and spatial-temporal case
SpMat c0;			/**< Diagonal mass matrix spatial part. 			*/
SpMat g1;			/**< stiffness matrix space. 						*/
SpMat g2;			/**< defined as : g1 * c0^-1 * g1  					*/

// only used in spatial-temporal case
SpMat g3;			/**< defined as : g1 * (c0^-1 * g1)^2.				*/
SpMat M0;			/**< diagonalised mass matrix time. 				*/
SpMat M1;			/**< diagonal matrix with diag(0.5, 0, ..., 0, 0.5) 
							-> account for boundary						*/
SpMat M2;			/**< stiffness matrix time.							*/

double yTy;			/**< compute t(y)*y once. */
Vector mu;			/**< conditional mean */
Vector t_grad;		/**< gradient of theta */
double min_f_theta; /**< minimum of function*/

public:
 /**
 * @brief constructor for regression model (no random effects). 
 * @param[in] ns_ number of spatial grid points per time step.
 * @param[in] nt_ number of temporal time steps.
 * @param[in] nb_ number of fixed effects.
 * @param[in] no_ number of observations.
 * @param[in] B_  covariate matrix.
 * @param[in] y_  vector with observations.
 * \note B = B_ or is its own copy?
 */	
PostTheta(int ns_, int nt_, int nb_, int no_, MatrixXd B_, VectorXd y_, Vector _theta_prior) : ns(ns_), nt(nt_), nb(nb_), no(no_), B(B_), y(y_), theta_prior(_theta_prior) {
	
	dim_th = 1;  			// only hyperparameter is the precision of the observations
	ns     = 0;
	n      = nb;
	yTy    = y.dot(y);

	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
	#endif

	min_f_theta        = 1e10;			// initialise min_f_theta, min_theta

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	printf("threads level 1 : %d\n", threads_level1);

	dim_grad_loop      = 2*dim_th;

	// one solver per thread, but not more than required
	num_solvers        = std::min(threads_level1, dim_grad_loop);
	printf("num solvers     : %d\n", num_solvers);

	solverQst          = new PardisoSolver[num_solvers];
	solverQ            = new PardisoSolver[num_solvers];

	// set global counter to count function evaluations
	fct_count          = 0;	// initialise min_f_theta, min_theta
}
/**
 * @brief constructor for spatial model (order 2).
 * @param[in] ns_ number of spatial grid points per time step.
 * @param[in] nt_ number of temporal time steps.
 * @param[in] nb_ number of fixed effects.
 * @param[in] no_ number of observations.
 * @param[in] Ax_  covariate matrix.
 * @param[in] y_  vector with observations.
 * @param[in] c0_ diagonalised mass matrix.
 * @param[in] g1_ stiffness matrix.
 * @param[in] g2_ defined as : g1 * c0^-1 * g1
 */	
PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, Vector _theta_prior) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), theta_prior(_theta_prior)  {

	dim_th      = 3;   			// 3 hyperparameters, precision for the observations, 2 for the spatial model
	nu          = ns;
	n           = nb + ns;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta
	yTy         = y.dot(y);

	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
	#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	printf("threads level 1 : %d\n", threads_level1);

	dim_grad_loop      = 2*dim_th;

	// one solver per thread, but not more than required
	num_solvers        = std::min(threads_level1, dim_grad_loop);
	printf("num solvers     : %d\n", num_solvers);

	solverQst          = new PardisoSolver[num_solvers];
	solverQ            = new PardisoSolver[num_solvers];

	// set global counter to count function evaluations
	fct_count          = 0;
}

/**
 * @brief constructor for spatial temporal model.
 * @brief constructor for spatial model (order 2).
 * @param[in] ns_ number of spatial grid points per time step.
 * @param[in] nt_ number of temporal time steps.
 * @param[in] nb_ number of fixed effects.
 * @param[in] no_ number of observations.
 * @param[in] Ax_  covariate matrix.
 * @param[in] y_  vector with observations.
 * @param[in] c0_ diagonalised mass matrix space.
 * @param[in] g1_ stiffness matrix space.
 * @param[in] g2_ defined as : g1 * c0^-1 * g1
 * @param[in] g3_ defined as : g1 * (c0^-1 * g1)^2
 * @param[in] M0_ diagonalised mass matrix time.
 * @param[in] M1_ diagonal matrix with diag(0.5, 0, ..., 0, 0.5) -> account for boundary
 * @param[in] M2_ stiffness matrix time.
 */	
PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vector _theta_prior) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior(_theta_prior)  {

	dim_th      = 4;    	 	// 4 hyperparameters, precision for the observations, 3 for the spatial-temporal model
	nu          = ns*nt;
	n           = nb + ns*nt;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta
	yTy         = y.dot(y);

	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
	#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	printf("threads level 1 : %d\n", threads_level1);

	dim_grad_loop      = 2*dim_th;

	// one solver per thread, but not more than required
	num_solvers        = std::min(threads_level1, dim_grad_loop);
	printf("num solvers     : %d\n", num_solvers);

	solverQst          = new PardisoSolver[num_solvers];
	solverQ            = new PardisoSolver[num_solvers];

	// set global counter to count function evaluations
	fct_count          = 0;
}

/**
 * @brief structure required by BFGS solver, requires : theta, gradient theta
 * \note Gradient call is already parallelised using nested OpenMP. 
 * --> there are l1 threads (usually 8, one for each function evaluation), that themselves
 * then split into another e.g. 8 threads, when calling PARDISO to factorise the system.
 * --> somehow introduce additional parallelism to compute f(theta), possible to do in parallel
 */
double operator()(Vector& theta, Vector& grad){

	t_grad = grad;

	// initialise min_f_theta, min_theta, store current minimum 
	// Vector mu;
	mu.setZero(n);

	double f_theta = eval_post_theta(theta, mu);

	if(f_theta < min_f_theta){
		min_f_theta = f_theta;

		std::cout << "theta : " << std::right << std::fixed << theta.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << std::endl;
		//std::cout << "theta   : " << theta.transpose() << ", f_theta : " << f_theta << std::endl;
	}

	double timespent_grad = -omp_get_wtime();

	Vector mu_dummy(n);
	eval_gradient(theta, f_theta, mu_dummy, grad);
	// std::cout << "grad : " << grad.transpose() << std::endl;

	timespent_grad += omp_get_wtime();

	#ifdef PRINT_MSG
		std::cout << "time spent gradient call : " << timespent_grad << std::endl;
	#endif

	return f_theta;
}

// ============================================================================================ //
// CONVERT MODEL PARAMETRISATION TO INTERPRETABLE PARAMETRISATION & VICE VERSA

/**
 * @brief convert hyperparameters theta from the model parametrisation to the interpretable
 * parametrisation ie. from log(gamma_E, gamma_s, gamma_t) to log(sigma.u, rangeS, rangeT)
 * @param [in]		log(gamma_E)
 * @param [in]		log(gamma_s)
 * @param [in]		log(gamma_t)
 * @param [inout]	log(sigma.u) precision of random effects
 * @param [inout]	log(ranS) 	 spatial range
 * @param [inout]	log(ranT) 	 temporal range
 */
void convert_theta2interpret(double lgamE, double lgamS, double lgamT, double& sigU, double& ranS, double& ranT){
	double alpha_t = 1; 
	double alpha_s = 2;
	double alpha_e = 1;

	double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
	double nu_s  = alpha   - 1;
	double nu_t  = alpha_t - 0.5;

	double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*8*pow(M_PI,1.5));
	double gE = exp(lgamE);
	double gS = exp(lgamS);
	double gT = exp(lgamT);

	sigU = log(sqrt(c1)/((gE*sqrt(gT))*pow(gS,alpha-1)));
	ranS = log(sqrt(8*nu_s)/gS);
	ranT = log(gT*sqrt(8*nu_t)/(pow(gS, alpha_s)));
}

/**
 * @brief convert hyperparameters theta from the interpretable parametrisation to the
 * model parametrisation ie. from log(sigma.u, rangeS, rangeT) to log(gamma_E, gamma_s, gamma_t)
 * @param [in]		log(sigma.u) precision of random effects
 * @param [in]		log(ranS) 	 spatial range
 * @param [in]		log(ranT) 	 temporal range
 * @param [inout]	log(gamma_E) 
 * @param [inout]	log(gamma_s)
 * @param [inout]	log(gamma_t)
 */
void convert_interpret2theta(double sigU, double ranS, double ranT, double& lgamE, double& lgamS, double& lgamT){
	double alpha_t = 1; 
	double alpha_s = 2;
	double alpha_e = 1;

	double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
	double nu_s  = alpha   - 1;
	double nu_t  = alpha_t - 0.5;

	double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*8*pow(M_PI,1.5));
	lgamS = 0.5*log(8*nu_s) - ranS;
	lgamT = ranT - 0.5*log(8*nu_t) + alpha_s * lgamS;
	lgamE = 0.5*log(c1) - 0.5*lgamT - nu_s*lgamS - sigU;
}

// ============================================================================================ //
// FUNCTIONS TO BE CALLED AFTER THE BFGS SOLVER CONVERGED

/**
 * @brief get conditional mean mu for theta.
 * @param [in]    theta hyperparameter vector
 * @param [inout] mu vector of the conditional mean
 */	
void get_mu(Vector& theta, Vector& mu){

	#ifdef PRINT_MSG
		std::cout << "get_mu()" << std::endl;
	#endif

	double f_theta = eval_post_theta(theta, mu);

	#ifdef PRINT_MSG
		std::cout << "mu(-10:end) :" << mu.tail(10) << std::endl;
	#endif
}

Vector get_grad(){
	return t_grad;
}

/**
 * @brief Compute Covariance matrix of hyperparameters theta, at theta.
 * @details computes the hessian of f(theta) using a second order finite
 * difference stencil and then inverts the hessian. Gaussian assumption.
 * @param [in] theta hyperparameter Vector
 * @return cov covariance matrix of the hyperparameters
 */	
MatrixXd get_Covariance(Vector& theta){

	int dim_th = theta.size();
	MatrixXd hess(dim_th,dim_th);

	// evaluate hessian
	double timespent_hess_eval = -omp_get_wtime();
	hess = hess_eval(theta);

	timespent_hess_eval += omp_get_wtime();
	std::cout << "time spent hessian evaluation: " << timespent_hess_eval << std::endl; 

	//std::cout << "hess : " << hess << std::endl; 

	MatrixXd cov(dim_th,dim_th);

	std::cout << "cholmod excluded for now! " << std::endl;
	//compute_inverse_cholmod(hess, cov);

	//std::cout << "cov_new : " << cov << std::endl; 

	return cov;
}

/**
 * @brief Compute the marginal variances of the latent parameters at theta. 
	 * Using selected inversion procedure.
	 * @param[in]  	 Vector theta.
	 * @param[inout] Vector with marginals of f.
 */	
void get_marginals_f(Vector& theta, Vector& vars){
	
	SpMat Q(n, n);
	construct_Q(theta, Q);

	#ifdef PRINT_MSG
		std::cout << "after construct Q in get get_marginals_f" << std::endl;
	#endif

	double timespent_sel_inv_pardiso = -omp_get_wtime();
	int tid = omp_get_thread_num();
	solverQ[tid].selected_inversion(Q, vars);
	timespent_sel_inv_pardiso += omp_get_wtime();	

	std::cout << "time spent selected inversion pardiso : " << timespent_sel_inv_pardiso << std::endl; 
}

/**
 * @brief computes the hessian at theta using second order finite difference.
	 * Is used be get_Covariance.
	 * @param[in] theta hyperparameter vector
	 * @return Dense Matrix with Hessian. 
	 * \todo not yet parallelised .... 
 */	
MatrixXd hess_eval(Vector& theta){

	double eps = 0.005;

	int dim_th = theta.size();
	MatrixXd epsId(dim_th, dim_th); 
	epsId = eps*epsId.setIdentity();

	MatrixXd hessUpper = MatrixXd::Zero(dim_th, dim_th);

	// compute upper tridiagonal structure
	// map 2D structure to 1D to be using omp parallel more efficiently
	int loop_dim = dim_th*dim_th;

	// at the moment 8 threads for k = 16 but only 10 entries need to be filled in matrix. We have 
	// 4 times 3 evaluations and 6 times 4 evaluations. So probably 2 threads do 2*3 = 6 evaluations 
	// and 6 threads each do 4. Also note that f(theta) is being computed 4 times. Should be reduced.

	#pragma omp parallel for
	for(int k = 0; k < loop_dim; k++){			

		// dummy mu
		Vector mu_tmp(n);

		// row index is integer division k / dim_th
		int i = k/dim_th;
		// col index is k mod dim_th
		int j = k % dim_th;

		// diagonal elements
		if(i == j){
			Vector theta_forw_i = theta+epsId.col(i);
			Vector theta_back_i = theta-epsId.col(i);

			double f_theta_forw_i = eval_post_theta(theta_forw_i, mu_tmp);
			double f_theta 		  = eval_post_theta(theta, mu_tmp);
			double f_theta_back_i = eval_post_theta(theta_back_i, mu_tmp);

			hessUpper(i,i) = (f_theta_forw_i - 2 * f_theta + f_theta_back_i)/(eps*eps);
		
		// symmetric only compute upper triangular part
		} else if(j > i) {
				Vector theta_forw_i_j 		= theta+epsId.col(i)+epsId.col(j);
				Vector theta_forw_i_back_j  = theta+epsId.col(i)-epsId.col(j);
				Vector theta_back_i_forw_j  = theta-epsId.col(i)+epsId.col(j);
				Vector theta_back_i_j 		= theta-epsId.col(i)-epsId.col(j);

    			hessUpper(i,j) = (  eval_post_theta(theta_forw_i_j, mu_tmp) \
                       - eval_post_theta(theta_forw_i_back_j, mu_tmp) - eval_post_theta(theta_back_i_forw_j, mu_tmp) \
                       + eval_post_theta(theta_back_i_j, mu_tmp)) / (4*eps*eps);
		}

	}

	//double time_double_loop = -omp_get_wtime();

	//MatrixXd hessUpper_old = MatrixXd::Zero(dim_th, dim_th);
	/*#pragma omp parallel for 
	for(int i=0; i < dim_th; i++){
		for(int j=i; j < dim_th; j++){

			// dummy mu
			Vector mu_tmp(n);

			if(i == j){
				Vector theta_forw_i = theta+epsId.col(i);
				Vector theta_back_i = theta-epsId.col(i);

				hessUpper_old(i,i) = (eval_post_theta(theta_forw_i, mu_tmp) - 2 * eval_post_theta(theta, mu_tmp) + eval_post_theta(theta_back_i, mu_tmp))/(eps*eps);

			} else {
				Vector theta_forw_i_j 		= theta+epsId.col(i)+epsId.col(j);
				Vector theta_forw_i_back_j  = theta+epsId.col(i)-epsId.col(j);
				Vector theta_back_i_forw_j  = theta-epsId.col(i)+epsId.col(j);
				Vector theta_back_i_j 		= theta-epsId.col(i)-epsId.col(j);

    			hessUpper_old(i,j) = (  eval_post_theta(theta_forw_i_j, mu_tmp) \
                       - eval_post_theta(theta_forw_i_back_j, mu_tmp) - eval_post_theta(theta_back_i_forw_j, mu_tmp) \
                       + eval_post_theta(theta_back_i_j, mu_tmp)) / (4*eps*eps);
       		}
		}
	}

	time_double_loop += omp_get_wtime();
	std::cout << "time double loop : " << time_double_loop << std::endl;*/

	//	std::cout << "hess upper : " << hessUpper << std::endl;
	MatrixXd hess = hessUpper.selfadjointView<Upper>();
	//std::cout << "hessian       : \n" << hess << std::endl;

	// check that matrix positive definite otherwise use only diagonal
	check_pos_def(hess); 

	return hess;
}

/**
 * @brief check if Hessian positive definite (matrix assumed to be dense & small since dim(theta) small)
	 * @param[inout] updates hessian to only the diagonal entries if not positive definite.
 */
 void check_pos_def(MatrixXd &hess){

	// compute Eigenvalues of symmetric matrix
	SelfAdjointEigenSolver<MatrixXd> es(hess);
	/*cout << "The eigenvalues of g1 are:" << endl << es.eigenvalues() << endl;
	cout << "The min eigenvalue of g1 is:" << endl << es.eigenvalues().minCoeff() << endl;*/

	// check if there are eigenvalues smaller or equal to zero
	if((es.eigenvalues().minCoeff()) <= 0){

		std::cout << "Matrix not positive definite only considering diagonal values!! " << std::endl;
		Vector diag_hess = hess.diagonal();
		hess = diag_hess.asDiagonal();
    	std::cout << "new hessian :\n" << hess << std::endl;
	} else {
		#ifdef PRINT_MSG
			std::cout << "Hessian is positive definite.";
		#endif
	}
}

// ============================================================================================ //
// ALL FOLLOWING FUNCTIONS CONTRIBUT TO THE EVALUATION OF F(THETA) & GRADIENT

/**
 * @brief Core function. Evaluate posterior of theta. mu are latent parameters.
 * @param[in]    theta hyperparameter vector
	 * @param[inout] mu vector of the conditional mean
	 * @return 		 f(theta) value
 */
double eval_post_theta(Vector& theta, Vector& mu){			

	// =============== set up ================= //
	int dim_th = theta.size();

	#ifdef PRINT_MSG
		std::cout << "in eval post theta function. " << std::endl;
		std::cout << "dim_th : " << dim_th << std::endl;
		std::cout << "nt : " << nt << std::endl;			
		std::cout << "theta prior : " << theta_prior.transpose() << std::endl;
	#endif

	// =============== evaluate theta prior based on original solution & variance = 1 ================= //

	//VectorXd zero_vec(dim_th); zero_vec.setZero();
	VectorXd zero_vec(theta_prior.size()); zero_vec.setZero();

	double log_prior_sum = 0;

	// evaluate prior
	VectorXd log_prior_vec(dim_th);
	for( int i=0; i<dim_th; i++ ){
		eval_log_prior(log_prior_vec[i], &theta[i], &theta_prior[i]);
	}
    
	log_prior_sum = log_prior_vec.sum();

	#ifdef PRINT_MSG
		std::cout << "log prior sum : " << log_prior_sum << std::endl;
	#endif

		// =============== evaluate prior of random effects : need log determinant ================= //

	// requires factorisation of Q.u -> can this be done in parallel with the 
	// factorisation of the denominator? 
	// How long does the assembly of Qu take? Should this be passed on to the 
	// denominator to be reused?

	double log_det_Qu = 0;

	if(ns > 0 ){
		eval_log_det_Qu(theta, log_det_Qu);
	}

	#ifdef PRINT_MSG
		std::cout << "log det Qu : "  << log_det_Qu << std::endl;
	#endif

		// =============== evaluate likelihood ================= //

	// eval_likelihood: log_det, -theta*yTy
	double log_det_l;
	double val_l; 
	eval_likelihood(theta, log_det_l, val_l);

	#ifdef PRINT_MSG
		std::cout << "log det l : "  << log_det_l << std::endl;
		std::cout << "val l     : " << val_l << std::endl;
	#endif

		// =============== evaluate denominator ================= //
	// denominator :
	// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
	double log_det_d;
	double val_d;
	SpMat Q(n, n);
	Vector rhs(n);

 	eval_denominator(theta, log_det_d, val_d, Q, rhs, mu);
	#ifdef PRINT_MSG
		std::cout << "log det d : " << log_det_d << std::endl;
		std::cout << "val d     : " <<  val_d << std::endl;
	#endif

		// =============== add everything together ================= //
  	double val = -1 * (log_prior_sum + log_det_Qu + log_det_l + val_l - (log_det_d + val_d));

  	#ifdef PRINT_MSG
  		std::cout << "f theta : " << val << std::endl;
  	#endif

  	return val;
}

/**
 * @brief evaluate log prior using original theta value
 * @param[in] thetai current theta_i value
 * @param[in] thetai_original original theta_i value
	 * @param[inout] log prior is being updated.
	 * @details variance / precision of 1 : no normalising constant. 
	 * computed through -0.5 * (theta_i* - theta_i)*(theta_i*-theta_i) 
 */	
void eval_log_prior(double& log_prior, double* thetai, double* thetai_original){

	log_prior = -0.5 * (*thetai - *thetai_original) * (*thetai - *thetai_original);

	#ifdef PRINT_MSG
		std::cout << "log prior for theta_i " << (*thetai) << " : " << (log_prior) << std::endl;
	#endif
}

/**
 * @brief evaluate log prior of random effects
 * @param[in] theta current theta vector
	 * @param[inout] log_det inserts log determinant.
	 * \todo construct spatial matrix (at the moment this is happening twice. FIX)
 */	
void eval_log_det_Qu(Vector& theta, double &log_det){

	SpMat Qu(nu, nu);
	if(nt > 1){
		construct_Q_spat_temp(theta, Qu);
	} else {
		construct_Q_spatial(theta, Qu);
	}

	int tid = omp_get_thread_num();
	solverQst[tid].factorize(Qu, log_det);

	#ifdef PRINT_MSG
		std::cout << "log det Qu : " << log_det << std::endl;
	#endif

	log_det = 0.5 * (log_det);
}

/**
 * @brief compute log likelihood : log_det tau*no and value -theta*yTy
 * @param[in] theta current theta vector
	 * @param[inout] log_det inserts log determinant of log likelihood.
	 * @param[inout] val inserts the value of -theta*yTy
 */	
void eval_likelihood(Vector& theta, double &log_det, double &val){
	
	// multiply log det by 0.5
	log_det = 0.5 * no*theta[0];
	//log_det = 0.5 * no*3;

	// - 1/2 ...
	val = - 0.5 * exp(theta[0])*yTy;
	//*val = - 0.5 * exp(3)*yTy;

	/*std::cout << "in eval eval_likelihood " << std::endl;
	std::cout << "theta     : " << theta << std::endl;
	std::cout << "yTy : " << yTy << ", exp(theta) : " << exp(theta) << std::endl;
	std::cout << "log det l : " << log_det << std::endl;
	std::cout << "val l     : " << val << std::endl; */
}

/**
 * @brief spatial model : SPDE discretisation -- matrix construction
 * @param[in] theta current theta vector
	 * @param[inout] Qs fills spatial precision matrix
 */
void construct_Q_spatial(Vector& theta, SpMat& Qs){

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

/**
 * @brief spatial temporal model : SPDE discretisation. DEMF(1,2,1) model.
 * @param[in] theta current theta vector
	 * @param[inout] Qst fills spatial-temporal precision matrix
 */
void construct_Q_spat_temp(Vector& theta, SpMat& Qst){

	double exp_theta1 = exp(theta[1]);
	double exp_theta2 = exp(theta[2]);
	double exp_theta3 = exp(theta[3]);

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

		//std::cout << "Qst : \n" << Qst->block(0,0,10,10) << std::endl;
}

/** @brief construct precision matrix. 
 * Calls spatial, spatial-temporal, etc.
 * @param[in] theta current theta vector
	 * @param[inout] Q fills precision matrix
 */
void construct_Q(Vector& theta, SpMat& Q){

	double exp_theta0 = exp(theta[0]);
	//double exp_theta = exp(3);

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

	if(ns > 0){
		SpMat Qu(nu, nu);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(theta, Qu);
		} else {	
			construct_Q_spatial(theta, Qu);
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

		Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

		#ifdef PRINT_MSG
			std::cout << "exp(theta0) : \n" << exp_theta0 << std::endl;
			std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
		#endif
	}

	if(ns == 0){
		// Q.e <- Diagonal(no, exp(theta))
		// Q.xy <- Q.x + crossprod(A.x, Q.e)%*%A.x  # crossprod = t(A)*Q.e (faster)	
		Q = Q_b + exp_theta0*B.transpose()*B;
	}

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

	#ifdef PRINT_MSG
		std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
		std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
		std::cout << "theta : \n" << theta.transpose() << std::endl;

	#endif 

}

/** @brief Assemble right-handside. 
 * @param[in] theta current theta vector
	 * @param[inout] rhs right-handside
	 * /todo Could compute Ax^T*y once, and only multiply with appropriate exp_theta.
 */	
void construct_b(Vector& theta, Vector &rhs){

	double exp_theta = exp(theta[0]);
	//double exp_theta = exp(3);

	if(ns == 0){
		rhs = exp_theta*B.transpose()*y;
	} else {
		rhs = exp_theta*Ax.transpose()*y;
	}
}

/** @brief Evaluate denominator: conditional probability of Qx|y
 * @param[in] theta current theta vector
 * @param[inout] log_det fill log determinant of conditional distribution of denominator
 * @param[inout] val fill value with mu*Q*mu
 * @param[inout] Q construct precision matrix
	 * @param[inout] rhs construct right-handside
	 * @param[inout] mu insert mean of latent parameters
 */
void eval_denominator(Vector& theta, double& log_det, double& val, SpMat& Q, Vector& rhs, Vector& mu){

	// construct Q_x|y,
	construct_Q(theta, Q);
	//Q->setIdentity();

	#ifdef PRINT_MSG
		printf("\nin eval denominator after construct_Q call.");
	#endif

	//  construct b_xey
	construct_b(theta, rhs);

	#ifdef PRINT_MSG
		printf("\nin eval denominator after construct_b call.");
	#endif

	// solve linear system
	// returns vector mu, which is of the same size as rhs
	//solve_cholmod(Q, rhs, mu, log_det);

	int tid = omp_get_thread_num();
	solverQ[tid].factorize_solve(Q, rhs, mu, log_det);

	log_det = 0.5 * (log_det);
	
	#ifdef PRINT_MSG
		std::cout << "log det d : " << log_det << std::endl;
	#endif

	// compute value
	val = -0.5 * mu.transpose()*(Q)*(mu); 

	/*std::cout << "in eval eval_denominator " << std::endl;

	std::cout << "rhs " << std::endl; std::cout <<  *rhs << std::endl;
	std::cout << "mu " << std::endl; std::cout << *mu << std::endl;
	std::cout << "Q " << std::endl; std::cout << Eigen::MatrixXd(*Q) << std::endl;

	std::cout << "log det d : " << log_det << std::endl;
	std::cout << "val d     : " << val << std::endl; */
}

// ============================================================================================ //
// FINITE DIFFERENCE GRADIENT EVALUATION

/** @brief Compute gradient using central finite difference stencil. Parallelised with OpenMP 
 * @param[in] theta current theta vector
	 * @param[in] f_theta value of f(theta) 
	 * @param[mu] mu 
	 * @param[inout] grad inserts gradient 
	 * \todo don't actually need gradient?
 */
void eval_gradient(Vector& theta, double f_theta, Vector& mu, Vector& grad){

	#ifdef PRINT_MSG
		std::cout << "\nin eval gradient function." << std::endl;
	#endif

	int dim_th = theta.size();

	double eps = 0.005;
	MatrixXd epsId_mat(dim_th, dim_th); 
	epsId_mat = eps*epsId_mat.setIdentity();
	//std::cout << "epsId_mat : " << epsId_mat << std::endl;

	Vector f_forw(dim_th);
	Vector f_backw(dim_th);

	int threads = omp_get_max_threads();

	// naively parallelise using OpenMP, more later
	#pragma omp parallel for
	for(int i=0; i<2*dim_th; i++){

			if(i % 2 == 0){
				int k = i/2;

				#ifdef PRINT_MSG
					std::cout << "forward loop thread rank: " << omp_get_thread_num() << " out of " << threads << std::endl;
					std::cout << "i : " << i << " and k : " << k << std::endl;
				#endif

				// temp vector
				Vector theta_forw(dim_th);
				Vector mu_dummy(n);

				theta_forw = theta + epsId_mat.col(k);
				f_forw[k] = eval_post_theta(theta_forw, mu_dummy);

			} else {
				int k = (i-1)/2;

				#ifdef PRINT_MSG
					std::cout << "backward loop thread rank: " << omp_get_thread_num() << " out of " << threads << std::endl;
					std::cout << "i : " << i << " and k : " << k << std::endl;
				#endif

				// temp vector
				Vector theta_backw(dim_th);
				Vector mu_dummy(n);

				theta_backw = theta - epsId_mat.col(k);
				f_backw[k] = eval_post_theta(theta_backw, mu_dummy);
			}
	}

	// compute finite difference in each direction
	grad = 1.0/(2.0*eps)*(f_forw - f_backw);
	// std::cout << "grad  : " << grad << std::endl;

}

 /**
 * @brief class destructor. Frees memory allocated by PostTheta class.
 */
~PostTheta(){

		delete[] solverQst;
		delete[] solverQ;		
}

};

#endif


// for Hessian approximation : 4-point stencil (2nd order)
// -> swap sign, invert, get covariance

// once converged call again : extract -> Q.xy -> selected inverse (diagonal), gives me variance wrt mode theta & data y

#endif   