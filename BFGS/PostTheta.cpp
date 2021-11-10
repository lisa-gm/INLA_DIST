#include "PostTheta.h"

//#include <likwid-marker.h>


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, MatrixXd B_, VectorXd y_, Vector theta_prior_, string solver_type_) : ns(ns_), nt(nt_), nb(nb_), no(no_), B(B_), y(y_), theta_prior(theta_prior_), solver_type(solver_type_) {
	
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
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	solverQ   = new Solver*[threads_level1];
	solverQst = new Solver*[threads_level1];

	if(solver_type == "PARDISO"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new PardisoSolver();
			solverQst[i] = new PardisoSolver();
		}
	} else if(solver_type == "RGF"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new RGFSolver(ns, nt, nb, no);
			solverQst[i] = new RGFSolver(ns, nt, nb, no);
		}
	} 

	// set global counter to count function evaluations
	fct_count          = 0;	// initialise min_f_theta, min_theta
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls
}


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, Vector theta_prior_, string solver_type_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), theta_prior(theta_prior_), solver_type(solver_type_) {

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
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	solverQ   = new Solver*[threads_level1];
	solverQst = new Solver*[threads_level1];

	if(solver_type == "PARDISO"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new PardisoSolver();
			solverQst[i] = new PardisoSolver();
		}
	} else if(solver_type == "RGF"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new RGFSolver(ns, nt, nb, no);
			solverQst[i] = new RGFSolver(ns, nt, nb, no);
		}
	} 

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

}


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vector theta_prior_, string solver_type_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior(theta_prior_), solver_type(solver_type_)  {

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
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	solverQ   = new Solver*[threads_level1];
	solverQst = new Solver*[threads_level1];

	if(solver_type == "PARDISO"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new PardisoSolver();
			solverQst[i] = new PardisoSolver();
		}
	} else if(solver_type == "RGF"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new RGFSolver(ns, nt, nb, no);
			solverQst[i] = new RGFSolver(ns, nt, nb, no);
		}
	} 

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls
}

/* operator() does exactly two things. 
1) evaluate f(theta) 
2) approx gradient of f(theta) */
/*
Restructure operator() : 

call all eval_post_theta() evaluations from here. This way all 9 can run in parallel. then compute gradient from these values.


*/
double PostTheta::operator()(Vector& theta, Vector& grad){


	#ifdef PRINT_MSG
		std::cout << "\niteration : " << iter_count << std::endl;
	#endif

	#ifdef PRINT_TIMES
		std::cout << "\niteration : " << iter_count << std::endl;
	#endif

	iter_count += 1; 

	double f_theta;

	int dim_th = theta.size();

	double eps = 0.005;
	MatrixXd epsId_mat(dim_th, dim_th); 
	epsId_mat = eps*epsId_mat.setIdentity();
	//std::cout << "epsId_mat : " << epsId_mat << std::endl;

	Vector f_forw(dim_th);
	Vector f_backw(dim_th);

	int threads = omp_get_max_threads();
	double timespent_f_theta_eval;
	double timespent_fct_eval = -omp_get_wtime();

	// do all function evaluations in parallel if enough threads are available
	#pragma omp parallel
	#pragma omp single
	{

	// ===================================== compute f(theta) ===================================== //
	#pragma omp task
	{ 
	mu.setZero(n);
	timespent_f_theta_eval = -omp_get_wtime();


	//LIKWID_MARKER_START("fThetaComputation");
	f_theta = eval_post_theta(theta, mu);
	//LIKWID_MARKER_STOP("fThetaComputation");
	

	timespent_f_theta_eval += omp_get_wtime();
	
	// print all theta's who result in a new minimum value for f(theta)
	if(f_theta < min_f_theta){
		min_f_theta = f_theta;
		std::cout << "theta : " << std::right << std::fixed << theta.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << std::endl;
		//std::cout << "theta   : " << theta.transpose() << ", f_theta : " << f_theta << std::endl;
	}
	} // end pragma omp task

	// ===================================== compute grad f(theta) ============================== //
	for(int i=0; i<2*dim_th; i++){

		if(i % 2 == 0){
			#pragma omp task
			{
			int k = i/2;

			#ifdef PRINT_MSG
				std::cout << "forward loop thread rank: " << omp_get_thread_num() << " out of " << threads << std::endl;
				std::cout << "i : " << i << " and k : " << k << std::endl;
			#endif

			Vector theta_forw(dim_th);
			Vector mu_dummy(n);

			theta_forw = theta + epsId_mat.col(k);
			f_forw[k] = eval_post_theta(theta_forw, mu_dummy);
			} // end pragma omp task

		} else if (i% 2 == 1){
			#pragma omp task
			{				
			int k = (i-1)/2;

			#ifdef PRINT_MSG
				std::cout << "backward loop thread rank: " << omp_get_thread_num() << " out of " << threads << std::endl;
				std::cout << "i : " << i << " and k : " << k << std::endl;
			#endif

			Vector theta_backw(dim_th);
			Vector mu_dummy(n);

			theta_backw = theta - epsId_mat.col(k);
			f_backw[k] = eval_post_theta(theta_backw, mu_dummy);
			}
		}

	} // end for loop

	} // end pragma omp single

	timespent_fct_eval += omp_get_wtime();

	#ifdef PRINT_TIMES
		std::cout << "time spent evaluation f(theta) : " << timespent_f_theta_eval << std::endl;
		std::cout << "time spent for eval f(theta) and grad(f) : " << timespent_fct_eval << std::endl;
	#endif 

	// compute finite difference in each direction
	grad = 1.0/(2.0*eps)*(f_forw - f_backw);
	//std::cout << "grad  : " << grad.transpose() << std::endl;

	t_grad = grad;

	return f_theta;
}

int PostTheta::get_fct_count(){
	return(fct_count);
}

// ============================================================================================ //
// CONVERT MODEL PARAMETRISATION TO INTERPRETABLE PARAMETRISATION & VICE VERSA

void PostTheta::convert_theta2interpret(double lgamE, double lgamS, double lgamT, double& ranT, double& ranS, double& sigU){
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


void PostTheta::convert_interpret2theta(double ranT, double ranS, double sigU, double& lgamE, double& lgamS, double& lgamT){
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

	#pragma omp parallel
	#pragma omp single
	{
	double f_theta = eval_post_theta(theta, mu);
	}

	#ifdef PRINT_MSG
		std::cout << "mu(-10:end) :" << mu.tail(10) << std::endl;
	#endif
}

Vector PostTheta::get_grad(){
	return t_grad;
}


MatrixXd PostTheta::get_Covariance(Vector& theta, double eps){

	int dim_th = theta.size();
	MatrixXd hess(dim_th,dim_th);

	// evaluate hessian
	double timespent_hess_eval = -omp_get_wtime();
	hess = hess_eval(theta, eps);


	timespent_hess_eval += omp_get_wtime();

	#ifdef PRINT_TIMES
		std::cout << "time spent hessian evaluation: " << timespent_hess_eval << std::endl;
	#endif 

	//std::cout << "estimated hessian         : \n" << hess << std::endl; 
	//std::cout << "eps : " << eps << endl;

	MatrixXd cov(dim_th,dim_th);
	// pardiso call with identity as rhs & solve.
	/*PardisoSolver* hessInv;
	hessInv = new PardisoSolver;
	hessInv->compute_inverse_pardiso(hess, cov); */

	// just use eigen solver
	cov = hess.inverse();

	//std::cout << "cov  : \n" << cov << std::endl; 

	return cov;
}


MatrixXd PostTheta::get_Cov_interpret_param(Vector& interpret_theta, double eps){

	int dim_th;

	if(interpret_theta.size() != 4){
		std::cout << "dim(interpret_theta) = " << interpret_theta.size() << ", should be 4!" << std::endl;
		exit(1);
	} else {
		dim_th = 4;
	}

	MatrixXd hess(dim_th,dim_th);

	// evaluate hessian
	double timespent_hess_eval = -omp_get_wtime();
	hess = hess_eval_interpret_theta(interpret_theta, eps);

	timespent_hess_eval += omp_get_wtime();

	#ifdef PRINT_TIMES
		std::cout << "time spent hessian evaluation: " << timespent_hess_eval << std::endl;
	#endif 

	//std::cout << "estimated hessian         : \n" << hess << std::endl; 
	//std::cout << "eps : " << eps << endl;

	MatrixXd cov(dim_th,dim_th);
	// pardiso call with identity as rhs & solve.
	/*PardisoSolver* hessInv;
	hessInv = new PardisoSolver;
	hessInv->compute_inverse_pardiso(hess, cov); */

	// just use eigen solver
	cov = hess.inverse();

	//std::cout << "cov  : \n" << cov << std::endl; 

	return cov;
}


void PostTheta::get_marginals_f(Vector& theta, Vector& vars){
	
	SpMat Q(n, n);
	construct_Q(theta, Q);

	#ifdef PRINT_MSG
		std::cout << "after construct Q in get get_marginals_f" << std::endl;
	#endif

	double timespent_sel_inv_pardiso = -omp_get_wtime();
	
	// call this with one thread
	#pragma omp parallel
	#pragma omp single
	{
	int tid = omp_get_thread_num();
	solverQ[tid]->selected_inversion(Q, vars);
	}

	#ifdef PRINT_TIMES
		timespent_sel_inv_pardiso += omp_get_wtime();
		std::cout << "time spent selected inversion pardiso : " << timespent_sel_inv_pardiso << std::endl; 
	#endif	

}

double PostTheta::f_eval(Vector& theta){
	// x[1]^3*x[2]^2*x[3]

	return(pow(theta[0],3)*pow(theta[1],2)*theta[2] + pow(theta[3],3));
}


MatrixXd PostTheta::hess_eval(Vector& theta, double eps){

	//double eps = 0.005;

	int dim_th = theta.size();
	MatrixXd epsId(dim_th, dim_th); 
	epsId = eps*epsId.setIdentity();

	MatrixXd hessUpper = MatrixXd::Zero(dim_th, dim_th);

	// compute upper tridiagonal structure
	// map 2D structure to 1D to be using omp parallel more efficiently
	int loop_dim = dim_th*dim_th;    

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i = Eigen::MatrixXd::Zero(3,dim_th);
    Eigen::MatrixXd f_i_j = Eigen::MatrixXd::Zero(4,loop_dim);

    double time_omp_task_hess = - omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {

    // compute f(theta) only once.
    #pragma omp task 
    { 
	Vector mu_tmp(n);
	//double f_theta = f_eval(theta);
	double f_theta = eval_post_theta(theta, mu_tmp);
    f_i_i.row(1) = f_theta * Eigen::VectorXd::Ones(dim_th).transpose(); 
    }

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){

        	// compute f(theta+eps_i)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_forw_i = theta+epsId.col(i);
            //f_i_i(0,i) = f_eval(theta_forw_i);
            f_i_i(0,i) = eval_post_theta(theta_forw_i, mu_tmp); 
            }

        	// compute f(theta-eps_i)
            # pragma omp task
            { 
            Vector mu_tmp(n);
            Vector theta_back_i = theta-epsId.col(i);
            //f_i_i(2,i) = f_eval(theta_back_i);
            f_i_i(2,i) = eval_post_theta(theta_back_i, mu_tmp); 
            }

        
        // symmetric only compute upper triangular part
        } else if(j > i) {

        	// compute f(theta+eps_i+eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_forw_i_j 	   = theta+epsId.col(i)+epsId.col(j);
            //f_i_j(0,k) = f_eval(theta_forw_i_j);
            f_i_j(0,k) 				   = eval_post_theta(theta_forw_i_j, mu_tmp); 
            }

        	// compute f(theta+eps_i-eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_forw_i_back_j = theta+epsId.col(i)-epsId.col(j);
            //f_i_j(1,k) = f_eval(theta_forw_i_back_j);
            f_i_j(1,k)                 = eval_post_theta(theta_forw_i_back_j, mu_tmp); 
            }

        	// compute f(theta-eps_i+eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_back_i_forw_j = theta-epsId.col(i)+epsId.col(j);
            //f_i_j(2,k) = f_eval(theta_back_i_forw_j);
            f_i_j(2,k)                 = eval_post_theta(theta_back_i_forw_j, mu_tmp); 
            }

        	// compute f(theta-eps_i-eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_back_i_j 	   = theta-epsId.col(i)-epsId.col(j);
            //f_i_j(3,k) = f_eval(theta_back_i_j);
            f_i_j(3,k)                 = eval_post_theta(theta_back_i_j, mu_tmp); 
            }            
        }

    }

    // potentially use task dependencies
    #pragma omp taskwait

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){
        	/*std::cout << "i = " << i << ",j = " << j << std::endl;
        	std::cout << "f_i_i(0," << i << ") = " << f_i_i(0,i) << std::endl;
        	std::cout << "f_i_i(1," << i << ") = " << f_i_i(1,i) << std::endl;
        	std::cout << "f_i_i(2," << i << ") = " << f_i_i(2,i) << std::endl;*/

            hessUpper(i,i) = (f_i_i(0,i) - 2 * f_i_i(1,i) + f_i_i(2,i))/(eps*eps);

        } else if(j > i){
            hessUpper(i,j) = (f_i_j(0,k) - f_i_j(1,k) - f_i_j(2,k) + f_i_j(3,k)) / (4*eps*eps);
        }
    }

    } // end omp

    time_omp_task_hess += omp_get_wtime();
    #ifdef PRINT_TIMES
    	std::cout << "time hess = " << time_omp_task_hess << std::endl;
    	//std::cout << "hess Upper      \n" << hessUpper << std::endl;
    #endif

    #ifdef PRINT_TIMES
    	std::cout << "time omp task hessian = " << time_omp_task_hess << std::endl;
    #endif

	MatrixXd hess = hessUpper.selfadjointView<Upper>();
	std::cout << "hessian       : \n" << hess << std::endl;

	// check that matrix positive definite otherwise use only diagonal
	//std::cout << "positive definite check disabled." << std::endl;
	check_pos_def(hess); 

	return hess;
}



MatrixXd PostTheta::hess_eval_interpret_theta(Vector& interpret_theta, double eps){

	//double eps = 0.005;

	int dim_th = interpret_theta.size();
	MatrixXd epsId(dim_th, dim_th); 
	epsId = eps*epsId.setIdentity();

	MatrixXd hessUpper = MatrixXd::Zero(dim_th, dim_th);

	// compute upper tridiagonal structure
	// map 2D structure to 1D to be using omp parallel more efficiently
	int loop_dim = dim_th*dim_th;    

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i = Eigen::MatrixXd::Zero(3,dim_th);
    Eigen::MatrixXd f_i_j = Eigen::MatrixXd::Zero(4,loop_dim);

    double time_omp_task_hess = - omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {

    // compute f(theta) only once.
    #pragma omp task 
    { 
	Vector mu_tmp(n);
	//double f_theta = f_eval(theta);
	// convert interpret_theta to theta
	Vector theta(4);
	theta[0] = interpret_theta[0];
	convert_interpret2theta(interpret_theta[1], interpret_theta[2], interpret_theta[3], theta[1], theta[2], theta[3]);
	double f_theta = eval_post_theta(theta, mu_tmp);
    f_i_i.row(1) = f_theta * Eigen::VectorXd::Ones(dim_th).transpose(); 
    }

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){

        	// compute f(theta+eps_i)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector interpret_theta_forw_i = interpret_theta+epsId.col(i);
            Vector theta_forw_i(4);
			theta_forw_i[0] = interpret_theta_forw_i[0];
			convert_interpret2theta(interpret_theta_forw_i[1], interpret_theta_forw_i[2], interpret_theta_forw_i[3], theta_forw_i[1], theta_forw_i[2], theta_forw_i[3]);
            //f_i_i(0,i) = f_eval(theta_forw_i);
            f_i_i(0,i) = eval_post_theta(theta_forw_i, mu_tmp); 
            }

        	// compute f(theta-eps_i)
            # pragma omp task
            { 
            Vector mu_tmp(n);
            Vector interpret_theta_back_i = interpret_theta-epsId.col(i);
            Vector theta_back_i(4);
			theta_back_i[0] = interpret_theta_back_i[0];
			convert_interpret2theta(interpret_theta_back_i[1], interpret_theta_back_i[2], interpret_theta_back_i[3], theta_back_i[1], theta_back_i[2], theta_back_i[3]);
            //f_i_i(2,i) = f_eval(theta_back_i);
            f_i_i(2,i) = eval_post_theta(theta_back_i, mu_tmp); 
            }

        
        // symmetric only compute upper triangular part
        } else if(j > i) {

        	// compute f(theta+eps_i+eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector interpret_theta_forw_i_j 	   = interpret_theta+epsId.col(i)+epsId.col(j);
            Vector theta_forw_i_j(4);
			theta_forw_i_j[0] = interpret_theta_forw_i_j[0];
			convert_interpret2theta(interpret_theta_forw_i_j[1], interpret_theta_forw_i_j[2], interpret_theta_forw_i_j[3], theta_forw_i_j[1], theta_forw_i_j[2], theta_forw_i_j[3]);
            //f_i_j(0,k) = f_eval(theta_forw_i_j);
            f_i_j(0,k) 				   = eval_post_theta(theta_forw_i_j, mu_tmp); 
            }

        	// compute f(theta+eps_i-eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector interpret_theta_forw_i_back_j = interpret_theta+epsId.col(i)-epsId.col(j);
            Vector theta_forw_i_back_j(4);
			theta_forw_i_back_j[0] = interpret_theta_forw_i_back_j[0];
			convert_interpret2theta(interpret_theta_forw_i_back_j[1], interpret_theta_forw_i_back_j[2], interpret_theta_forw_i_back_j[3], theta_forw_i_back_j[1], theta_forw_i_back_j[2], theta_forw_i_back_j[3]);
            //f_i_j(1,k) = f_eval(theta_forw_i_back_j);
            f_i_j(1,k)                 = eval_post_theta(theta_forw_i_back_j, mu_tmp); 
            }

        	// compute f(theta-eps_i+eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector interpret_theta_back_i_forw_j = interpret_theta-epsId.col(i)+epsId.col(j);
            Vector theta_back_i_forw_j(4);
			theta_back_i_forw_j[0] = interpret_theta_back_i_forw_j[0];
			convert_interpret2theta(interpret_theta_back_i_forw_j[1], interpret_theta_back_i_forw_j[2], interpret_theta_back_i_forw_j[3], theta_back_i_forw_j[1], theta_back_i_forw_j[2], theta_back_i_forw_j[3]);
            //f_i_j(2,k) = f_eval(theta_back_i_forw_j);
            f_i_j(2,k)                 = eval_post_theta(theta_back_i_forw_j, mu_tmp); 
            }

        	// compute f(theta-eps_i-eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector interpret_theta_back_i_j 	   = interpret_theta-epsId.col(i)-epsId.col(j);
            Vector theta_back_i_j(4);
			theta_back_i_j[0] = interpret_theta_back_i_j[0];
			convert_interpret2theta(interpret_theta_back_i_j[1], interpret_theta_back_i_j[2], interpret_theta_back_i_j[3], theta_back_i_j[1], theta_back_i_j[2], theta_back_i_j[3]);
            //f_i_j(3,k) = f_eval(theta_back_i_j);
            f_i_j(3,k)                 = eval_post_theta(theta_back_i_j, mu_tmp); 
            }            
        }

    }

    // potentially use task dependencies
    #pragma omp taskwait

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){
        	/*std::cout << "i = " << i << ",j = " << j << std::endl;
        	std::cout << "f_i_i(0," << i << ") = " << f_i_i(0,i) << std::endl;
        	std::cout << "f_i_i(1," << i << ") = " << f_i_i(1,i) << std::endl;
        	std::cout << "f_i_i(2," << i << ") = " << f_i_i(2,i) << std::endl;*/

            hessUpper(i,i) = (f_i_i(0,i) - 2 * f_i_i(1,i) + f_i_i(2,i))/(eps*eps);

        } else if(j > i){
            hessUpper(i,j) = (f_i_j(0,k) - f_i_j(1,k) - f_i_j(2,k) + f_i_j(3,k)) / (4*eps*eps);
        }
    }

    } // end omp

    time_omp_task_hess += omp_get_wtime();
    #ifdef PRINT_TIMES
    	std::cout << "time hess = " << time_omp_task_hess << std::endl;
    	//std::cout << "hess Upper      \n" << hessUpper << std::endl;
    #endif

    #ifdef PRINT_TIMES
    	std::cout << "time omp task hessian = " << time_omp_task_hess << std::endl;
    #endif

	MatrixXd hess = hessUpper.selfadjointView<Upper>();
	std::cout << "hessian       : \n" << hess << std::endl;

	// check that matrix positive definite otherwise use only diagonal
	//std::cout << "positive definite check disabled." << std::endl;
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
// ALL FOLLOWING FUNCTIONS CONTRIBUTE TO THE EVALUATION OF F(THETA) & GRADIENT

double PostTheta::eval_post_theta(Vector& theta, Vector& mu){

	if(omp_get_thread_num() == 0){
		fct_count += 1;
	}			

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
	solverQst[tid]->factorize(Qu, log_det);

	#ifdef PRINT_MSG
		std::cout << "log det Qu : " << log_det << std::endl;
	#endif

	log_det = 0.5 * (log_det);
}


void PostTheta::eval_likelihood(Vector& theta, double &log_det, double &val){
	
	// multiply log det by 0.5
	double theta0 = theta[0];
	log_det = 0.5 * no*theta0;
	//log_det = 0.5 * no*3;

	// - 1/2 ...
	val = - 0.5 * exp(theta0)*yTy;
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

		//std::cout << "Qst : \n" << Qst->block(0,0,10,10) << std::endl;
}


void PostTheta::construct_Q(Vector& theta, SpMat& Q){

	double exp_theta0 = exp(theta[0]);

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

			std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
			std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
			std::cout << "theta : \n" << theta.transpose() << std::endl;

		#endif
	}

	if(ns == 0){
		// Q.e <- Diagonal(no, exp(theta))
		// Q.xy <- Q.x + crossprod(A.x, Q.e)%*%A.x  # crossprod = t(A)*Q.e (faster)	
		Q = Q_b + exp_theta0*B.transpose()*B;	

		#ifdef PRINT_MSG
			std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
			std::cout << "Q : \n" << Q << std::endl;
			std::cout << "theta : \n" << theta.transpose() << std::endl;
		#endif 

	}

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}


void PostTheta::construct_b(Vector& theta, Vector &rhs){

	double exp_theta = exp(theta[0]);

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
	solverQ[tid]->factorize_solve(Q, rhs, mu, log_det);

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


PostTheta::~PostTheta(){

		delete[] solverQst;
		delete[] solverQ;		
}


// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //

// for Hessian approximation : 4-point stencil (2nd order)
// -> swap sign, invert, get covariance

// once converged call again : extract -> Q.xy -> selected inverse (diagonal), gives me variance wrt mode theta & data y

