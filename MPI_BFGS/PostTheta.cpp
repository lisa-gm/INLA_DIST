#include "PostTheta.h"

PostTheta::PostTheta(int dim_th_) : dim_th(dim_th_)  {

	min_f_theta = 1e10;			// initialise min_f_theta, min_theta

	dim_grad_loop      = 2*dim_th;
	no_f_eval		   = 2*dim_th + 1;

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

	t_grad = grad;

	double eps = 0.005;
	MatrixXd epsId_mat(dim_th, dim_th); 
	epsId_mat = eps*epsId_mat.setIdentity();
	//std::cout << "epsId_mat : " << epsId_mat << std::endl;

	Vector f_forw(dim_th);
	Vector f_backw(dim_th);

	/*int threads = omp_get_max_threads();
	double timespent_f_theta_eval;
	double timespent_fct_eval = -omp_get_wtime();*/

	//Vector theta_forw_loc[dim_th];

	// make send Isend -> update wait, careful with theta_loc, theta_loc_array -> make them individual
	// 
	// MPI_send to evaluate f(theta, mu) -> this always goes to rank 1 !!
	theta_array = theta.data();
	MPI_Send(theta_array, dim_th, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

	// MPI_send to evaluate f(theta+eps_i, mu_dummy)
	// send to processes 2 - (dim_th+1)
	for(int i=0; i<dim_th; i++){
		Vector theta_loc = theta + epsId_mat.col(i);
		double* theta_loc_array = theta_loc.data();
		MPI_Send(theta_loc_array, dim_th, MPI_DOUBLE, i+2, 0, MPI_COMM_WORLD);
	}

	// MPI_send to evaluate f(theta-eps_i, mu_dummy)
	// send to processes (dim_th+2) - (2*dim_th+1)
	for(int i=0; i<dim_th; i++){
		Vector theta_loc = theta - epsId_mat.col(i);
		double* theta_loc_array = theta_loc.data();
		MPI_Send(theta_loc_array, dim_th, MPI_DOUBLE, i+dim_th+2, 0, MPI_COMM_WORLD);
	}

	// TODO: Irecv doesn't receive the right result. Why??
	MPI_Status statuses[no_f_eval];
	MPI_Request requests[no_f_eval];
	int num_requests = 0;

	// MPI_Irecv f_theta, deal with mu later. potentially get it from model
	MPI_Irecv(&f_theta, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, requests+num_requests);
	num_requests++;

	// MPI_Irecv f_theta_forw
	for(int i=0; i<dim_th; i++){
		MPI_Irecv(&f_forw[i], 1, MPI_DOUBLE, i+2, 0, MPI_COMM_WORLD, requests+num_requests);
		num_requests++;
	}

	// MPI_Irecv f_theta_back
	for(int i=0; i<dim_th; i++){
		MPI_Irecv(&f_backw[i], 1, MPI_DOUBLE, i+dim_th+2, 0, MPI_COMM_WORLD, requests+num_requests);
		num_requests++;
	}

	MPI_Waitall(num_requests, requests, statuses);

	#ifdef PRINT_MSG
		std::cout << "PostTheta received f_theta = " << f_theta << " from model." << std::endl;
	#endif

	// print all theta's who result in a new minimum value for f(theta)
	if(f_theta < min_f_theta){
		min_f_theta = f_theta;
		std::cout << "theta : " << std::right << std::fixed << theta.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << std::endl;
		//std::cout << "theta   : " << theta.transpose() << ", f_theta : " << f_theta << std::endl;
	}

	// compute finite difference in each direction
	grad = 1.0/(2.0*eps)*(f_forw - f_backw);
	//std::cout << "grad  : " << grad.transpose() << std::endl;

	return f_theta;
}

int PostTheta::get_fct_count(){
	return(fct_count);
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

#if 0
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

	#ifdef PRINT_TIMES
		std::cout << "time spent hessian evaluation: " << timespent_hess_eval << std::endl;
	#endif 

	//std::cout << "hess : " << hess << std::endl; 

	MatrixXd cov(dim_th,dim_th);
	// pardiso call with identity as rhs & solve.
	PardisoSolver* hessInv;
	hessInv = new PardisoSolver;
	hessInv->compute_inverse_pardiso(hess, cov); 

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
	int tid = omp_get_thread_num();
	solverQ[tid]->selected_inversion(Q, vars);

	#ifdef PRINT_TIMES
		timespent_sel_inv_pardiso += omp_get_wtime();
	#endif	

	std::cout << "time spent selected inversion pardiso : " << timespent_sel_inv_pardiso << std::endl; 
}


MatrixXd PostTheta::hess_eval(Vector& theta){

	double eps = 0.005;

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
            f_i_i(0,i) = eval_post_theta(theta_forw_i, mu_tmp); 
            }

        	// compute f(theta-eps_i)
            # pragma omp task
            { 
            Vector mu_tmp(n);
            Vector theta_back_i = theta-epsId.col(i);
            f_i_i(2,i) = eval_post_theta(theta_back_i, mu_tmp); 
            }

        
        // symmetric only compute upper triangular part
        } else if(j > i) {

        	// compute f(theta+eps_i+eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_forw_i_j 	   = theta+epsId.col(i)+epsId.col(j);
            f_i_j(0,k) 				   = eval_post_theta(theta_forw_i_j, mu_tmp); 
            }

        	// compute f(theta+eps_i-eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_forw_i_back_j = theta+epsId.col(i)-epsId.col(j);
            f_i_j(1,k)                 = eval_post_theta(theta_forw_i_back_j, mu_tmp); 
            }

        	// compute f(theta-eps_i+eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_back_i_forw_j = theta-epsId.col(i)+epsId.col(j);
            f_i_j(2,k)                 = eval_post_theta(theta_back_i_forw_j, mu_tmp); 
            }

        	// compute f(theta-eps_i-eps_j)
            #pragma omp task 
            { 
            Vector mu_tmp(n);
            Vector theta_back_i_j 	   = theta-epsId.col(i)-epsId.col(j);
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
	//std::cout << "hessian       : \n" << hess << std::endl;

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

#endif

PostTheta::~PostTheta(){

		//delete[] solverQst;
		//delete[] solverQ;		
}


// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //

// for Hessian approximation : 4-point stencil (2nd order)
// -> swap sign, invert, get covariance

// once converged call again : extract -> Q.xy -> selected inverse (diagonal), gives me variance wrt mode theta & data y

