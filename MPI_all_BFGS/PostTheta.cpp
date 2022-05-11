#include "PostTheta.h"

//#include <likwid-marker.h>


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, MatrixXd B_, Vect y_, Vect theta_prior_param_, string solver_type_, const bool constr_, const MatrixXd Dxy_) : ns(ns_), nt(nt_), nb(nb_), no(no_), B(B_), y(y_), theta_prior_param(theta_prior_param_), solver_type(solver_type_), constr(constr_), Dxy(Dxy_) {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
	
	dim_th = 1;  			// only hyperparameter is the precision of the observations
	ns     = 0;
	n      = nb;
	yTy    = y.dot(y);
	BTy    = B.transpose()*y;


#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
		printf("Eigen -- number of threads used : %d\n", Eigen::nbThreads( ));
#endif

	min_f_theta        = 1e10;			// initialise min_f_theta, min_theta

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();

	if(MPI_rank == 0){
		printf("threads level 1 : %d\n", threads_level1);
	}

	dim_grad_loop      = 2*dim_th;
	no_f_eval 		   = 2*dim_th + 1;

	// one solver per thread, but not more than required
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
#endif

	if(solver_type == "PARDISO"){
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
	} else if(solver_type == "RGF"){
		solverQ   = new RGFSolver(ns, nt, nb, no);
		solverQst = new RGFSolver(ns, nt, nb, no);
	} 

	prior = "gaussian";

	// set global counter to count function evaluations
	fct_count          = 0;	// initialise min_f_theta, min_theta
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

#ifdef SMART_GRAD
	thetaDiff_initialized = false;
	if(MPI_rank == 0){
		std::cout << "Smart gradient enabled." << std::endl;
	}
#else
	if(MPI_rank == 0){
		std::cout << "Regular FD gradient enabled." << std::endl;
	}
#endif

}


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, Vect y_, SpMat c0_, SpMat g1_, SpMat g2_, Vect theta_prior_param_, string solver_type_, const bool constr_, const MatrixXd Dx_, const MatrixXd Dxy_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), theta_prior_param(theta_prior_param_), solver_type(solver_type_), constr(constr_), Dx(Dx_), Dxy(Dxy_) {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);  
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

	dim_th      = 3;   			// 3 hyperparameters, precision for the observations, 2 for the spatial model
	nu          = ns;
	n           = nb + ns;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta
	yTy         = y.dot(y);
	AxTy		= Ax.transpose()*y;
	AxTAx       = Ax.transpose()*Ax;



	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
		printf("Eigen -- number of threads used : %d\n", Eigen::nbThreads( ));

	#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	int threads_level2;

	#pragma omp parallel
    {  
   	threads_level2 = omp_get_max_threads();
    }
	
#ifdef PRINT_MSG
	if(MPI_rank == 0){
		printf("threads level 1 : %d\n", threads_level1);
		printf("threads level 2 : %d\n", threads_level2);
	}
#endif

	dim_grad_loop      = 2*dim_th;
	no_f_eval 		   = 2*dim_th + 1;


	// one solver per thread, but not more than required
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	if(solver_type == "PARDISO"){
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
	} else if(solver_type == "RGF"){
		solverQ   = new RGFSolver(ns, nt, nb, no);
		solverQst = new RGFSolver(ns, nt, nb, no);
	}  

	prior = "gaussian";

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

#ifdef SMART_GRAD
	thetaDiff_initialized = false;
	if(MPI_rank == 0){
		std::cout << "Smart gradient enabled." << std::endl;
	}
#else
	if(MPI_rank == 0){
		std::cout << "Regular FD gradient enabled." << std::endl;
	}
#endif


}


PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, Vect y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vect theta_prior_param_, string solver_type_, const bool constr_, const MatrixXd Dx_, const MatrixXd Dxy_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior_param(theta_prior_param_), solver_type(solver_type_), constr(constr_), Dx(Dx_), Dxy(Dxy_)  {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

	dim_th      = 4;    	 	// 4 hyperparameters, precision for the observations, 3 for the spatial-temporal model
	nu          = ns*nt;
	n           = nb + ns*nt;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta
	yTy         = y.dot(y);
	AxTy		= Ax.transpose()*y;
	AxTAx       = Ax.transpose()*Ax;


	#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
		printf("Eigen -- number of threads used : %d\n", Eigen::nbThreads( ));
	#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	int threads_level1 = omp_get_max_threads();
	int threads_level2;

	#pragma omp parallel
    {  
   	threads_level2 = omp_get_max_threads();
    }
	
#ifdef PRINT_MSG
	if(MPI_rank == 0){
		printf("threads level 1 : %d\n", threads_level1);
		printf("threads level 2 : %d\n", threads_level2);
	}
#endif

	dim_grad_loop      = 2*dim_th;
	no_f_eval 		   = 2*dim_th + 1;

	// one solver per thread, but not more than required
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	if(solver_type == "PARDISO"){
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
	} else if(solver_type == "RGF"){
		solverQ   = new RGFSolver(ns, nt, nb, no);
		solverQst = new RGFSolver(ns, nt, nb, no);
	} 

	// set prior to be gaussian
	//prior = "gaussian";
	prior = "pc";

	if(MPI_rank == 0){
		std::cout << "Prior : " << prior << std::endl;	
	}

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

#ifdef SMART_GRAD
	thetaDiff_initialized = false;
	if(MPI_rank == 0){
		std::cout << "Smart gradient enabled." << std::endl;
	}
#else
	if(MPI_rank == 0){
		std::cout << "Regular FD gradient enabled." << std::endl;
	}
#endif

}

/* operator() does exactly two things. 
1) evaluate f(theta) 
2) approx gradient of f(theta) */
/*
Restructure operator() : 

call all eval_post_theta() evaluations from here. This way all 9 can run in parallel. then compute gradient from these values.


*/
double PostTheta::operator()(Vect& theta, Vect& grad){


	#ifdef PRINT_MSG
		std::cout << "\niteration : " << iter_count << std::endl;
	#endif

	#ifdef PRINT_TIMES
		if(MPI_rank == 0)
			std::cout << "\niteration : " << iter_count << std::endl;
	#endif

	iter_count += 1; 

	int dim_th = theta.size();

	// configure finite difference approximation (along coordinate axes or smart gradient)
	double eps = 0.005;
	// projection matrix G, either Identity or other orthonormal basis (from computeG function)
	//G = MatrixXd::Identity(dim_th, dim_th);

#ifdef SMART_GRAD
	// changing G here, however G needs to be available Hessian later
	computeG(theta);
#else
	G = MatrixXd::Identity(dim_th, dim_th);
#endif


#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "G : \n" << G << std::endl;
	}
#endif



	// initialise local f_value lists
	Vect f_temp_list_loc(no_f_eval); f_temp_list_loc.setZero();

	int threads = omp_get_max_threads();
	double timespent_f_theta_eval;
	double timespent_fct_eval = -omp_get_wtime();

	// ======================================== set up MPI ========================================== //
	// create list that assigns each of the no_f_eval = 2*dim_th+1 function evaluations to a rank
	// if e.g. 9 tasks and mpi_size = 3, then task 1: rank 0, task 2: rank 1, task 3: rank 2, task 4: rank 0, etc.
	ArrayXi task_to_rank_list(no_f_eval);

	for(int i=0; i<no_f_eval; i++){
		task_to_rank_list[i] = i % MPI_size;
	}

#ifdef PRINT_MSG
	if(MPI_rank == 0){  
		std::cout << "task_to_rank_list : " << task_to_rank_list.transpose() << std::endl;
	}
#endif

	// ===================================== compute f(theta) ===================================== //
	if(MPI_rank == task_to_rank_list[0])
	{ 
		mu.setZero(n);
		timespent_f_theta_eval = -omp_get_wtime();

		//LIKWID_MARKER_START("fThetaComputation");
		f_temp_list_loc(0) = eval_post_theta(theta, mu);
		//LIKWID_MARKER_STOP("fThetaComputation");
		
		timespent_f_theta_eval += omp_get_wtime();
	} // end if MPI

	// ===================================== compute grad f(theta) ============================== //
	// fill f_temp_list_loc such that first entry f(theta), next dim_th forward difference, last 
	// dim_th backward difference -> each process has their own copy, rest zero (important), then combine
	int divd = ceil(no_f_eval / double(2));

	for(int i=1; i<no_f_eval; i++){

		// compute all FORWARD DIFFERENCES
		if(i / divd == 0){
			if(MPI_rank == task_to_rank_list[i])
			{
				int k = i-1; 

#ifdef PRINT_MSG
				//std::cout <<"i = " << i << ", i / divd = " << i / divd << ", rank " << MPI_rank << std::endl;
					std::cout << "i : " << i << " and k : " << k << std::endl;
#endif

				Vect theta_forw(dim_th);
				Vect mu_dummy(n);

				theta_forw = theta + eps*G.col(k);
				f_temp_list_loc(i) = eval_post_theta(theta_forw, mu_dummy);
			} // end MPI if
		
		// compute all BACKWARD DIFFERENCES
		} else if (i / divd > 0){
			if(MPI_rank == task_to_rank_list[i])
			{				
				int k = i-1-dim_th; // backward difference in the k-th direction

#ifdef PRINT_MSG
					std::cout <<"i = " << i << ", i / divd = " << i / divd << ", rank " << MPI_rank << std::endl;
					//std::cout << "i : " << i << " and k : " << k << std::endl;
#endif

				Vect theta_backw(dim_th);
				Vect mu_dummy(n);

				theta_backw = theta - eps*G.col(k);
				f_temp_list_loc(i) = eval_post_theta(theta_backw, mu_dummy);
			} // end MPI if
		}
	} // end for loop

	// ================== MPI Waitall & MPI All_Gather ====================== //

#ifdef PRINT_MSG
	std::cout << "rank : " << MPI_rank << ", res : " << f_temp_list_loc.transpose() << std::endl;
#endif

	// wait for all processes to finish
	MPI_Barrier(MPI_COMM_WORLD);

	// distribute the results among all processes
	Vect f_temp_list(no_f_eval);

	MPI_Allreduce(f_temp_list_loc.data(), f_temp_list.data(), no_f_eval, MPI_DOUBLE, MPI_SUM,
              MPI_COMM_WORLD);

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "f temp list : " << f_temp_list.transpose() << std::endl;
	}
#endif

	// now write them into appropriate forward / backward buffer
	double f_theta = f_temp_list(0);

	// print all theta's who result in a new minimum value for f(theta)
	if(f_theta < min_f_theta){
		min_f_theta = f_theta;
		if(MPI_rank == 0){
			//std::cout << "\n>>>>>> theta : " << std::right << std::fixed << theta.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << "<<<<<<" << std::endl;
			std::cout << "theta : " << std::right << std::fixed << theta.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << std::endl;
			/*Vect theta_interpret(4); theta_interpret[0] = theta[0];
			convert_theta2interpret(theta[1], theta[2], theta[3], theta_interpret[1], theta_interpret[2], theta_interpret[3]);
			std::cout << "theta interpret : " << std::right << std::fixed << theta_interpret.transpose() << ",    f_theta : " << std::right << std::fixed << f_theta << std::endl;
			*/
		}
	}

	Vect f_forw  = f_temp_list.segment(1,dim_th);
	Vect f_backw = f_temp_list.tail(dim_th);

	timespent_fct_eval += omp_get_wtime();

#ifdef PRINT_TIMES
		if(MPI_rank == 0){
			std::cout << "time spent evaluation f(theta)         : " << timespent_f_theta_eval << std::endl;
			std::cout << "time spent for all funct. evaluations  : " << timespent_fct_eval << std::endl;
		}
#endif 

#ifdef SMART_GRAD
	grad = 1.0/(2.0*eps)*(f_forw - f_backw);
	// multiply with G^-T
    grad = G.transpose().fullPivLu().solve(grad);
#else
	// compute finite difference in each direction
	grad = 1.0/(2.0*eps)*(f_forw - f_backw);
#endif

	t_grad = grad;
	//std::cout << "grad : " << grad.transpose() << std::endl;

#ifdef PRINT_MSG
	if(MPI_rank == 0){  
        //std::cout << "f_theta : " << std::right << std::fixed << std::setprecision(12) << f_theta << std::endl;
        std::cout << "grad    : " << std::right << std::fixed << std::setprecision(12) << grad.transpose()  << std::endl;
    }
#endif

	return f_theta;

}


#ifdef SMART_GRAD
// compute transformation of derivative directions smart gradient
void PostTheta::computeG(Vect& theta){

	//int n = theta.size();

    // construct/update ThetaDiff matrix
    // go to else in first call otherwise true
    if(thetaDiff_initialized == true){

        VectorXd temp_col(n);
        for(int i=dim_th-1; i>0; i--){
            temp_col = ThetaDiff.col(i-1);
            ThetaDiff.col(i) = temp_col;
        }

        ThetaDiff.col(0) = theta - theta_prev;
        //std::cout << "theta_diff = \n" << ThetaDiff << std::endl;

        // add small noise term to diagonal, in case columns are linearly dependent
        double eps = 10e-6;
        // this additional eps term is being carried around through iterations, 
        // as the columns are just shifted ... 
        ThetaDiff = ThetaDiff + eps*MatrixXd::Identity(dim_th,dim_th);

    } else {
        ThetaDiff = MatrixXd::Identity(dim_th,dim_th);
        //std::cout << "ThetaDiff = \n" << ThetaDiff << std::endl;

        thetaDiff_initialized = true;
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){
    	std::cout << "theta_diff = \n" << ThetaDiff << std::endl;
    }
#endif

    // store current iterate
    theta_prev = theta;

    // do modified GRAM-SCHMIDT-ORTHONORMALIZATION
    G = MatrixXd::Zero(dim_th, dim_th);
    MatrixXd R = Eigen::MatrixXd::Zero(dim_th, dim_th);

    for(int k=0; k<dim_th; k++){
        G.col(k) = ThetaDiff.col(k);
        for(int i = 0; i<k; i++){
            R(i,k) = G.col(i).transpose()*G.col(k);
            G.col(k) = G.col(k) - R(i,k)*G.col(i);
        }
        R(k,k) = G.col(k).norm();
        G.col(k) = G.col(k)/R(k,k);
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){  
            std::cout << "f_theta : " << std::right << std::fixed << std::setprecision(12) << f_theta << std::endl;
            std::cout << "grad    : " << std::right << std::fixed << std::setprecision(12) << grad.transpose()  << std::endl;
    }
#endif


#ifdef PRINT_MSG
    if(MPI_rank == 0){
    	// check if ThetaDiff = G*R
    	//std::cout << "norm(ThetaDiff - G*R) = " << (ThetaDiff - G*R).norm() << std::endl;
    	std::cout << "G = \n" << G << std::endl;
    }
#endif

}

#endif


// need to write this for MPI ... all gather .. sum.
int PostTheta::get_fct_count(){

	// sum over fct counts of all processes 
	int total_fn_count;
	MPI_Reduce(&fct_count, &total_fn_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	// send total number to all processes, not so important
	MPI_Bcast(&total_fn_count, 1, MPI_INT, 0, MPI_COMM_WORLD); 

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "number of fn calls : " << total_fn_count << ", function calls rank " << MPI_rank << " : " << fct_count << std::endl;
	}
#endif

	return(total_fn_count);
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


void PostTheta::get_mu(Vect& theta, Vect& mu){

	#ifdef PRINT_MSG
		std::cout << "get_mu()" << std::endl;
	#endif

	double f_theta = eval_post_theta(theta, mu);

	#ifdef PRINT_MSG
		std::cout << "mu(-10:end) :" << mu.tail(10) << std::endl;
	#endif
}

Vect PostTheta::get_grad(){
	return t_grad;
}


MatrixXd PostTheta::get_Covariance(Vect& theta, double eps){

	int dim_th = theta.size();
	MatrixXd hess(dim_th,dim_th);

	// evaluate hessian
	double timespent_hess_eval = -omp_get_wtime();
	hess = hess_eval(theta, eps);


	timespent_hess_eval += omp_get_wtime();

#ifdef PRINT_TIMES
	if(MPI_rank == 0)
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


MatrixXd PostTheta::get_Cov_interpret_param(Vect& interpret_theta, double eps){

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


void PostTheta::get_marginals_f(Vect& theta, Vect& vars){
	
	SpMat Q(n, n);
	construct_Q(theta, Q);

	#ifdef PRINT_MSG
		std::cout << "after construct Q in get get_marginals_f" << std::endl;
	#endif

	double timespent_sel_inv_pardiso = -omp_get_wtime();

	if(constr == true){
		#pragma omp parallel
		#pragma omp single
		{
			MatrixXd V(n, Dxy.rows());
			solverQ->selected_inversion_w_constr(Q, Dxy, vars, V);
			MatrixXd W = Dxy*V;
			MatrixXd S = W.inverse()*V.transpose();

			Vect update_vars(n);
			for(int i=0; i<n; i++){
				update_vars[i] = V.row(i)*S.col(i);
			}

			//std::cout << "vars        = " << vars.transpose() << std::endl;			
			//std::cout << "update_vars = " << update_vars.transpose() << std::endl;
			vars = vars - update_vars;
			//std::cout << "vars        = " << vars.transpose() << std::endl;			

		}

	} else {
		// nested parallelism, want to call this with 1 thread of omp level 1
		#pragma omp parallel
		#pragma omp single
		{
			solverQ->selected_inversion(Q, vars);
		}
	}
	
	#ifdef PRINT_TIMES
		timespent_sel_inv_pardiso += omp_get_wtime();
		std::cout << "time spent selected inversion pardiso : " << timespent_sel_inv_pardiso << std::endl; 
	#endif	

}

double PostTheta::f_eval(Vect& theta){
	// x[1]^3*x[2]^2*x[3]

	return(pow(theta[0],3)*pow(theta[1],2)*theta[2] + pow(theta[3],3));
}

/* Parallelisation here is a bit of a mess, as I haven't found a "natural" way to combine
the parallel structure of MPI process + nested parallelism with the number of function
evaluations required here. For none Gaussian data this probably needs to be completely 
rewritten.
*/
MatrixXd PostTheta::hess_eval(Vect& theta, double eps){

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "G : \n" << G << std::endl;
	}
#endif

	//double eps = 0.005;
	int dim_th = theta.size();
 	MatrixXd epsG(dim_th, dim_th);
   
#ifdef SMART_GRAD
	if(thetaDiff_initialized == true){
		epsG = eps*G;
	} else {
		if(MPI_rank == 0){
			std::cout << "G not initialised! Using canonic basis!" << std::endl;
		}
		G = MatrixXd::Identity(dim_th, dim_th);
		epsG = eps*G;
	}
#else
	MatrixXd G = MatrixXd::Identity(dim_th, dim_th);
	epsG = eps*G;
#endif

#ifdef PRINT_MSG
	if(MPI_rank == 0)
		std::cout << "epsG = \n" << epsG << std::endl;
#endif

	MatrixXd hessUpper = MatrixXd::Zero(dim_th, dim_th);

	// compute upper tridiagonal structure
	// map 2D structure to 1D to be using omp parallel more efficiently
	int loop_dim = dim_th*dim_th;    

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i_loc = Eigen::MatrixXd::Zero(3,dim_th);
    Eigen::MatrixXd f_i_j_loc = Eigen::MatrixXd::Zero(4,loop_dim);

    // ======================================== set up MPI ========================================== //
	// create list that assigns each of the function evaluations to a rank
	// we have to evaluate f(theta), 2*dim_th for forward-backward diagonal entries &
	// 4 * (no_of_upper_diagonal_entries)
	int no_of_tasks = 1 + 2*dim_th + 4/2*dim_th*(dim_th-1);
	ArrayXi task_to_rank_list(no_of_tasks);
	int divd = ceil(no_of_tasks / double(MPI_size));
	//std::cout << "div : " << div << std::endl;
	double counter = 0;

	for(int i=0; i<task_to_rank_list.size(); i++){
		task_to_rank_list[i] = i / divd;
	}

	#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "task_to_rank_list : " << task_to_rank_list.transpose() << std::endl;
	}
	#endif

    double time_omp_task_hess = - omp_get_wtime();

    // compute f(theta) only once.
	if(MPI_rank == task_to_rank_list[0]){

		
		Vect mu_tmp(n);
		//double f_theta = f_eval(theta);
		double f_theta = eval_post_theta(theta, mu_tmp);
	    f_i_i_loc.row(1) = f_theta * Vect::Ones(dim_th).transpose(); 
    }
    counter++;

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){

        	// compute f(theta+eps_i)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect theta_forw_i = theta+epsG.col(i);
	            //f_i_i(0,i) = f_eval(theta_forw_i);
	            f_i_i_loc(0,i) = eval_post_theta(theta_forw_i, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect theta_back_i = theta-epsG.col(i);
	            //f_i_i(2,i) = f_eval(theta_back_i);
	            f_i_i_loc(2,i) = eval_post_theta(theta_back_i, mu_tmp);
            }
            counter++;

        
        // symmetric only compute upper triangular part
        // diagonal entries from f_temp_list_loc(1:2*dim_th+1)

        } else if(j > i) {

        	// compute f(theta+eps_i+eps_j)
            if(MPI_rank == task_to_rank_list[counter]){             
	            Vect mu_tmp(n);
	            Vect theta_forw_i_j 	   = theta+epsG.col(i)+epsG.col(j);
	            //f_i_j(0,k) = f_eval(theta_forw_i_j);
	            f_i_j_loc(0,k) 				   = eval_post_theta(theta_forw_i_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta+eps_i-eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect theta_forw_i_back_j = theta+epsG.col(i)-epsG.col(j);
	            //f_i_j(1,k) = f_eval(theta_forw_i_back_j);
	            f_i_j_loc(1,k)                 = eval_post_theta(theta_forw_i_back_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i+eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect theta_back_i_forw_j = theta-epsG.col(i)+epsG.col(j);
	            //f_i_j(2,k) = f_eval(theta_back_i_forw_j);
	            f_i_j_loc(2,k)                 = eval_post_theta(theta_back_i_forw_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i-eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect theta_back_i_j 	   = theta-epsG.col(i)-epsG.col(j);
	            //f_i_j(3,k) = f_eval(theta_back_i_j);
	            f_i_j_loc(3,k)                 = eval_post_theta(theta_back_i_j, mu_tmp); 
            }
            counter++;            
        }

    }

    //std::cout << "rank : " << MPI_rank << ", counter : " << counter << ", f_i_j : \n" << f_i_j_loc << std::endl;
    //std::cout << "rank : " << MPI_rank << ", f_i_i : \n" << f_i_i_loc << std::endl;

    // wait for all ranks to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i(3,dim_th);
    Eigen::MatrixXd f_i_j(4,loop_dim);

	MPI_Allreduce(f_i_i_loc.data(), f_i_i.data(), 3*dim_th, MPI_DOUBLE, MPI_SUM,
              MPI_COMM_WORLD);

	MPI_Allreduce(f_i_j_loc.data(), f_i_j.data(), 4*loop_dim, MPI_DOUBLE, MPI_SUM,
          MPI_COMM_WORLD);

	#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "f_i_i : \n" << f_i_i << std::endl;
		std::cout << "f_i_j : \n" << f_i_j << std::endl;
	}
	#endif

	// compute hessian
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

    time_omp_task_hess += omp_get_wtime();
#ifdef PRINT_TIMES
    if(MPI_rank == 0){
    	std::cout << "time hess = " << time_omp_task_hess << std::endl;
    	//std::cout << "hess Upper      \n" << hessUpper << std::endl;
    	}
#endif

	MatrixXd hess = hessUpper.selfadjointView<Upper>();

#ifdef SMART_GRAD
	hess = G.transpose().fullPivLu().solve(hess)*G.transpose();
#endif

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "hessian       : \n" << hess << std::endl;
	}
#endif

	// check that matrix positive definite otherwise use only diagonal
	//std::cout << "positive definite check disabled." << std::endl;
	check_pos_def(hess); 

	return hess;
}

#if 1
MatrixXd PostTheta::hess_eval_interpret_theta(Vect& interpret_theta, double eps){

	//double eps = 0.005;

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "G : \n" << G << std::endl;
	}
#endif

	int dim_th = interpret_theta.size();
	MatrixXd epsG(dim_th, dim_th);
   
#ifdef SMART_GRAD
	if(thetaDiff_initialized == true){
		epsG = eps*G;
	} else {
		if(MPI_rank == 0){
			std::cout << "G not initialised! Using canonic basis!" << std::endl;
		}
		G = MatrixXd::Identity(dim_th, dim_th);
		epsG = eps*G;
	}
#else
	MatrixXd G = MatrixXd::Identity(dim_th, dim_th);
	epsG = eps*G;
#endif

#ifdef PRINT_MSG
	if(MPI_rank == 0)
		std::cout << "epsG = \n" << epsG << std::endl;
#endif

	MatrixXd hessUpper = MatrixXd::Zero(dim_th, dim_th);

	// compute upper tridiagonal structure
	// map 2D structure to 1D to be using omp parallel more efficiently
	int loop_dim = dim_th*dim_th;    

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i_loc = Eigen::MatrixXd::Zero(3,dim_th);
    Eigen::MatrixXd f_i_j_loc = Eigen::MatrixXd::Zero(4,loop_dim);

    // ======================================== set up MPI ========================================== //
	// create list that assigns each of the function evaluations to a rank
	// we have to evaluate f(theta), 2*dim_th for forward-backward diagonal entries &
	// 4 * (no_of_upper_diagonal_entries)
	int no_of_tasks = 1 + 2*dim_th + 4/2*dim_th*(dim_th-1);
	ArrayXi task_to_rank_list(no_of_tasks);
	int divd = ceil(no_of_tasks / double(MPI_size));
	//std::cout << "div : " << div << std::endl;
	double counter = 0;

	for(int i=0; i<task_to_rank_list.size(); i++){
		task_to_rank_list[i] = i / divd;
	}

	#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "task_to_rank_list : " << task_to_rank_list.transpose() << std::endl;
	}
	#endif

    double time_omp_task_hess = - omp_get_wtime();

    // compute f(theta) only once.
	if(MPI_rank == task_to_rank_list[0]){
		Vect mu_tmp(n); 
		// convert interpret_theta to theta
		Vect theta(4);
		theta[0] = interpret_theta[0];
		convert_interpret2theta(interpret_theta[1], interpret_theta[2], interpret_theta[3], theta[1], theta[2], theta[3]);
		double f_theta = eval_post_theta(theta, mu_tmp);
	    f_i_i_loc.row(1) = f_theta * Vect::Ones(dim_th).transpose(); 
    }
    counter++;

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;

        // diagonal elements
        if(i == j){

        	// compute f(theta+eps_i)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
            	Vect interpret_theta_forw_i = interpret_theta+epsG.col(i);
            	Vect theta_forw_i(4);
				theta_forw_i[0] = interpret_theta_forw_i[0];
				convert_interpret2theta(interpret_theta_forw_i[1], interpret_theta_forw_i[2], interpret_theta_forw_i[3], theta_forw_i[1], theta_forw_i[2], theta_forw_i[3]);
	            f_i_i_loc(0,i) = eval_post_theta(theta_forw_i, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
				Vect interpret_theta_back_i = interpret_theta-epsG.col(i);
            	Vect theta_back_i(4);
				theta_back_i[0] = interpret_theta_back_i[0];
				convert_interpret2theta(interpret_theta_back_i[1], interpret_theta_back_i[2], interpret_theta_back_i[3], theta_back_i[1], theta_back_i[2], theta_back_i[3]);
	            f_i_i_loc(2,i) = eval_post_theta(theta_back_i, mu_tmp);
            }
            counter++;

        
        // symmetric only compute upper triangular part
        // diagonal entries from f_temp_list_loc(1:2*dim_th+1)
        } else if(j > i) {

        	// compute f(theta+eps_i+eps_j)
            if(MPI_rank == task_to_rank_list[counter]){             
	            Vect mu_tmp(n);
				Vect interpret_theta_forw_i_j 	   = interpret_theta+epsG.col(i)+epsG.col(j);
            	Vect theta_forw_i_j(4);
				theta_forw_i_j[0] = interpret_theta_forw_i_j[0];
				convert_interpret2theta(interpret_theta_forw_i_j[1], interpret_theta_forw_i_j[2], interpret_theta_forw_i_j[3], theta_forw_i_j[1], theta_forw_i_j[2], theta_forw_i_j[3]);
	            f_i_j_loc(0,k) 				   = eval_post_theta(theta_forw_i_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta+eps_i-eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect interpret_theta_forw_i_back_j = interpret_theta+epsG.col(i)-epsG.col(j);
            	Vect theta_forw_i_back_j(4);
				theta_forw_i_back_j[0] = interpret_theta_forw_i_back_j[0];
				convert_interpret2theta(interpret_theta_forw_i_back_j[1], interpret_theta_forw_i_back_j[2], interpret_theta_forw_i_back_j[3], theta_forw_i_back_j[1], theta_forw_i_back_j[2], theta_forw_i_back_j[3]);
	            f_i_j_loc(1,k)                 = eval_post_theta(theta_forw_i_back_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i+eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect interpret_theta_back_i_forw_j = interpret_theta-epsG.col(i)+epsG.col(j);
            	Vect theta_back_i_forw_j(4);
				theta_back_i_forw_j[0] = interpret_theta_back_i_forw_j[0];
				convert_interpret2theta(interpret_theta_back_i_forw_j[1], interpret_theta_back_i_forw_j[2], interpret_theta_back_i_forw_j[3], theta_back_i_forw_j[1], theta_back_i_forw_j[2], theta_back_i_forw_j[3]);
	            f_i_j_loc(2,k)                 = eval_post_theta(theta_back_i_forw_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i-eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
            	Vect interpret_theta_back_i_j 	   = interpret_theta-epsG.col(i)-epsG.col(j);
            	Vect theta_back_i_j(4);
				theta_back_i_j[0] = interpret_theta_back_i_j[0];
				convert_interpret2theta(interpret_theta_back_i_j[1], interpret_theta_back_i_j[2], interpret_theta_back_i_j[3], theta_back_i_j[1], theta_back_i_j[2], theta_back_i_j[3]);
	            f_i_j_loc(3,k)                 = eval_post_theta(theta_back_i_j, mu_tmp); 
            }
            counter++;            
        }

    }

    //std::cout << "rank : " << MPI_rank << ", counter : " << counter << ", f_i_j : \n" << f_i_j_loc << std::endl;
    //std::cout << "rank : " << MPI_rank << ", f_i_i : \n" << f_i_i_loc << std::endl;

    // wait for all ranks to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // number of rows stems from the required function evaluations of f(theta)
    Eigen::MatrixXd f_i_i(3,dim_th);
    Eigen::MatrixXd f_i_j(4,loop_dim);

	MPI_Allreduce(f_i_i_loc.data(), f_i_i.data(), 3*dim_th, MPI_DOUBLE, MPI_SUM,
              MPI_COMM_WORLD);

	MPI_Allreduce(f_i_j_loc.data(), f_i_j.data(), 4*loop_dim, MPI_DOUBLE, MPI_SUM,
          MPI_COMM_WORLD);

	#ifdef PRINT_MSG
	if(MPI_rank == 0){
		std::cout << "f_i_i : \n" << f_i_i << std::endl;
		std::cout << "f_i_j : \n" << f_i_j << std::endl;
	}
	#endif

	// compute hessian
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

    time_omp_task_hess += omp_get_wtime();
    #ifdef PRINT_TIMES
    	std::cout << "time hess = " << time_omp_task_hess << std::endl;
    	//std::cout << "hess Upper      \n" << hessUpper << std::endl;
    #endif

	MatrixXd hess = hessUpper.selfadjointView<Upper>();

#ifdef SMART_GRAD
	hess = G.transpose().fullPivLu().solve(hess)*G.transpose();
#endif

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		 std::cout << "time hess = " << time_omp_task_hess << std::endl;
		std::cout << "hessian       : \n" << hess << std::endl;
	}
#endif

	// check that matrix positive definite otherwise use only diagonal
	//std::cout << "positive definite check disabled." << std::endl;
	check_pos_def(hess); 

	return hess;
}
#endif

void PostTheta::check_pos_def(MatrixXd &hess){

	// compute Eigenvalues of symmetric matrix
	SelfAdjointEigenSolver<MatrixXd> es(hess);
	/*cout << "The eigenvalues of g1 are:" << endl << es.eigenvalues() << endl;
	cout << "The min eigenvalue of g1 is:" << endl << es.eigenvalues().minCoeff() << endl;*/

	// check if there are eigenvalues smaller or equal to zero
	if((es.eigenvalues().minCoeff()) <= 0){

		if(MPI_rank == 0){
			std::cout << "Matrix not positive definite only considering diagonal values!! " << std::endl;
		}
		Vect diag_hess = hess.diagonal();
		hess = diag_hess.asDiagonal();
		if(MPI_rank == 0){
    		std::cout << "new hessian :\n" << hess << std::endl;
    	}
	} else {
		#ifdef PRINT_MSG
			std::cout << "Hessian is positive definite.";
		#endif
	}
}

// ============================================================================================ //
// ALL FOLLOWING FUNCTIONS CONTRIBUTE TO THE EVALUATION OF F(THETA) & GRADIENT
// INCLUDE: OpenMP division for computation of nominator & denominator : ie. 2 tasks -> 2 threads!
double PostTheta::eval_post_theta(Vect& theta, Vect& mu){

	if(omp_get_thread_num() == 0){
		fct_count += 1;
	}			

	// =============== set up ================= //
	int dim_th = theta.size();

	#ifdef PRINT_MSG
		std::cout << "in eval post theta function. " << std::endl;
		std::cout << "dim_th : " << dim_th << std::endl;
		std::cout << "nt : " << nt << std::endl;			
		std::cout << "theta prior param : " << theta_prior_param.transpose() << std::endl;
	#endif

	// sum log prior, log det spat-temp prior
	double log_prior_sum;
	double log_det_Qu;

	// eval_likelihood: log_det, -theta*yTy
	double log_det_l;
	double val_l; 

	// log det denominator, value
	double log_det_d;
	double val_d;

	#pragma omp parallel
	#pragma omp single
	{
	// =============== evaluate NOMINATOR ================= //
	#pragma omp task
	{ 
	// =============== evaluate theta prior based on original solution & variance = 1 ================= //
	
	#ifdef PRINT_MSG
		std::cout << "prior : " << prior << std::endl;
	#endif

	if(prior == "gaussian" || dim_th != 4){
		// evaluate gaussian prior
		Vect log_prior_vec(dim_th);
		for( int i=0; i<dim_th; i++ ){
			eval_log_gaussian_prior(log_prior_vec[i], &theta[i], &theta_prior_param[i]);
		}
	    
		log_prior_sum = log_prior_vec.sum();

	} else if(prior == "pc"){
		// pc prior
		Vect theta_interpret(dim_th); // order: sigma s, range t, range s, sigma u
		theta_interpret[0] = theta[0];
		convert_theta2interpret(theta[1], theta[2], theta[3], theta_interpret[1], theta_interpret[2], theta_interpret[3]);
		//theta_interpret << 0.5, 10, 1, 4; 
		
		//Vect lambda(4);
		//lambda << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0; // lambda0 & lambda3 equal
		// pc prior(lambda, theta_interpret) -> expects order: sigma s, range t, range s, sigma u (same for lambdas!!)
		//eval_log_pc_prior(log_prior_sum, lambda, theta_interpret);
		eval_log_pc_prior(log_prior_sum, theta_prior_param, theta_interpret);


	} else {
		std::cout << "Prior not appropriately defined." << std::endl;
		exit(1);
	}

	#ifdef PRINT_MSG
		std::cout << "log prior sum : " << log_prior_sum << std::endl;
	#endif

	// =============== evaluate prior of random effects : need log determinant ================= //

	// requires factorisation of Q.u -> can this be done in parallel with the 
	// factorisation of the denominator? 
	// How long does the assembly of Qu take? Should this be passed on to the 
	// denominator to be reused?

	if(ns > 0 ){
		eval_log_det_Qu(theta, log_det_Qu);
	}

	#ifdef PRINT_MSG
		std::cout << "log det Qu : "  << log_det_Qu << std::endl;
	#endif

	// =============== evaluate likelihood ================= //

	eval_likelihood(theta, log_det_l, val_l);

	#ifdef PRINT_MSG
		std::cout << "log det likelihood : "  << log_det_l << std::endl;
		std::cout << "val likelihood     : " << val_l << std::endl;
	#endif

	} // end pragma omp task of computing nominator

	#pragma omp task
	{

	// =============== evaluate denominator ================= //
	// denominator :
	// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
	SpMat Q(n, n);
	Vect rhs(n);

 	eval_denominator(theta, log_det_d, val_d, Q, rhs, mu);

	#ifdef PRINT_MSG
		std::cout << "log det d : " << log_det_d << std::endl;
		std::cout << "val d     : " <<  val_d << std::endl;
	#endif
	}

    #pragma omp taskwait

	} // closing omp parallel region

	// =============== add everything together ================= //
	//std::cout << "log_det_d = "  << log_det_d << ", val d = " << val_d << std::endl;
  	double val = -1 * (log_prior_sum + log_det_Qu + log_det_l + val_l - (log_det_d + val_d));

#ifdef PRINT_MSG
  	std::cout << MPI_rank << " " << std::setprecision(6) << theta.transpose();
  	std::cout << " " << std::fixed << std::setprecision(12);
  	std::cout << log_prior_sum << " ";
  	std::cout << log_det_Qu << " " << log_det_l << " " << val_l << " " << log_det_d << " " << val_d << " " << val << std::endl;

  	//std::cout << "f theta : " << val << std::endl;
#endif

  	return val;
}


void PostTheta::eval_log_gaussian_prior(double& log_prior, double* thetai, double* thetai_original){

	log_prior = -0.5 * (*thetai - *thetai_original) * (*thetai - *thetai_original);

	#ifdef PRINT_MSG
		std::cout << "log prior for theta_i " << (*thetai) << " : " << (log_prior) << std::endl;
	#endif
}

// assume interpret_theta order : sigma.e, range t, range s, sigma.u
void PostTheta::eval_log_pc_prior(double& log_sum, Vect& lambda, Vect& interpret_theta){

  double prior_se = log(lambda[0]) - lambda[0] * exp(interpret_theta[0]) + interpret_theta[0];
  //printf("prior se = %f\n", prior_se);
  double prior_su = log(lambda[3]) - lambda[3] * exp(interpret_theta[3]) + interpret_theta[3];
  //printf("prior su = %f\n", prior_su);

  
  double prior_rt = log(lambda[1]) - lambda[1] * exp(-0.5*interpret_theta[1]) + log(0.5) - 0.5*interpret_theta[1];
  //printf("prior rt = %f\n", prior_rt);
  double prior_rs = log(lambda[2]) - lambda[2] * exp(-interpret_theta[2]) - interpret_theta[2];
  //printf("prior rs = %f\n", prior_rs);

  log_sum = prior_rt + prior_rs + prior_su + prior_se;

	#ifdef PRINT_MSG
		std::cout << "log prior sum " << log_sum << std::endl;
	#endif
}


void PostTheta::update_mean_constr(const MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W, MatrixXd& U, Vect& updated_sol){

    // now that we have V = Q^-1*t(Dxy), compute W = Dxy*V
    W = D*V;
    //std::cout << "W = " << W << std::endl;
    // U = W^-1*V^T, W is spd and small
    // TODO: use proper solver ...
    U = W.inverse()*V.transpose();
    //std::cout << "U = " << U << std::endl;

    Vect c = D*sol - e;
    updated_sol = sol - U.transpose()*c;

    //std::cout << "sum(updated_sol) = " << (D*updated_sol).sum() << std::endl;

}

void PostTheta::eval_log_dens_constr(Vect& x, Vect& mu, SpMat&Q, double& log_det_Q, const MatrixXd& D, MatrixXd& W, double& val_log_dens){

	int rowsQ = Q.rows();

    // log(pi(x)) 
    //std::cout << "log det Q = " << log_det_Q << std::endl;
    //std::cout << "- 0.5*(x_xy-mu_xy).transpose()*Q*(x_xy-mu_xy) = " << - 0.5*(x - mu).transpose()* Q *(x - mu) << std::endl;
    double log_pi_x    = - 0.5*rowsQ*log(2*M_PI) + 0.5*log_det_Q - 0.5*(x - mu).transpose()* Q *(x - mu);
    // log(pi(Ax|x)) 
    MatrixXd DDT = D*D.transpose();
    // .logDeterminant() is in cholmod module, requires inclusion of all of cholmod ...
    // W = D*Q^-1*t(D), want log(sqrt(1/det(W)) = - 0.5 * log(det(W)) 
    double log_pi_Ax_x = - 0.5*log(DDT.determinant());
    //std::cout << "log_pi_Ax_x = " << log_pi_Ax_x << std::endl;
    // log(pi(Ax)), W1 is covariance matrix
    double log_pi_Ax   = - 0.5*D.rows()*log(2*M_PI) - 0.5*log(W.determinant()) - 0.5*(D*x - D*mu).transpose()*W.inverse()*(D*x - D*mu);
    //std::cout << "W = " << W << std::endl;
    //std::cout << "log_pi_Ax = " << log_pi_Ax << ", log(W.determinant()) = " << log(W.determinant()) << "0.5*(D*x - D*mu).transpose()*W.inverse()*(D*x - D*mu) = " << 0.5*(D*x - D*mu).transpose()*W.inverse()*(D*x - D*mu) << std::endl;

    val_log_dens = log_pi_x + log_pi_Ax_x - log_pi_Ax;
    //std::cout << - 0.5*rowsQ*log(2*M_PI) - (- 0.5*D.rows()*log(2*M_PI)) << " " << - 0.5*(rowsQ-D.rows())*log(2*M_PI) << std::endl;
    //std::cout << "log val Bayes cond = " << val_log_dens << std::endl;  

}


void PostTheta::eval_log_det_Qu(Vect& theta, double &log_det){


	double time_construct_Qst = -omp_get_wtime();
	SpMat Qu(nu, nu);
	if(nt > 1){
		construct_Q_spat_temp(theta, Qu);
	} else {
		construct_Q_spatial(theta, Qu);
	}
	time_construct_Qst += omp_get_wtime();


	double time_factorize_Qst = -omp_get_wtime();
	solverQst->factorize(Qu, log_det);
	time_factorize_Qst += omp_get_wtime();

#ifdef PRINT_TIMES
	if(MPI_rank ==0){
		std::cout << "time construct Qst prior = " << time_construct_Qst << std::endl;
		std::cout << "time factorize Qst prior = " << time_factorize_Qst << std::endl;
	}
#endif


#ifdef PRINT_MSG
	std::cout << "log det Qu : " << log_det << std::endl;
#endif

	log_det = 0.5 * (log_det);
}

// 0.5*theta[0]*t(y - B*b - A*u)*(y - B*b - A*u) => normally assume x = u,b = 0
// constraint case -> maybe cannot evaluate in zero i.e. when e != 0, 
// might make more sense to evaluate x = mu_constraint, from Dxy*mu_constraint = e
void PostTheta::eval_likelihood(Vect& theta, double &log_det, double &val){
	
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


void PostTheta::construct_Q_spatial(Vect& theta, SpMat& Qs){

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


void PostTheta::construct_Q_spat_temp(Vect& theta, SpMat& Qst){

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

void PostTheta::update_mean_constr(MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W){

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


void PostTheta::construct_Q(Vect& theta, SpMat& Q){

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

		Q =  Qx + exp_theta0 * AxTAx;

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


void PostTheta::construct_b(Vect& theta, Vect &rhs){

	double exp_theta = exp(theta[0]);

	if(ns == 0){
		rhs = exp_theta*BTy;
	} else {
		rhs = exp_theta*AxTy;
	}
}


void PostTheta::eval_denominator(Vect& theta, double& log_det, double& val, SpMat& Q, Vect& rhs, Vect& mu){

	double time_construct_Q = -omp_get_wtime();
	// construct Q_x|y,
	construct_Q(theta, Q);
	time_construct_Q += omp_get_wtime();

#ifdef PRINT_MSG
	printf("\nin eval denominator after construct_Q call.");
#endif

	//  construct b_xey
	construct_b(theta, rhs);

#ifdef PRINT_MSG
	printf("\nin eval denominator after construct_b call.");
#endif

	double time_solve_Q = -omp_get_wtime();

	if(constr == true){
		//std::cout << "in eval denominator in constr true" << std::endl;
		// Dxy globally known from constructor 
		MatrixXd V(mu.size(), Dxy.rows());
		solverQ->factorize_solve_w_constr(Q, rhs, Dxy, log_det, mu, V);
		//std::cout << "after factorize_solve_w_constr" << std::endl;

		Vect constr_mu(mu.size());
		Vect e = Vect::Zero(1);
		MatrixXd U(Dxy.rows(), mu.size());
		MatrixXd W(Dxy.rows(), Dxy.rows());
		update_mean_constr(Dxy, e, mu, V, W, U, constr_mu);
		Vect unconstr_mu = mu;
		mu = constr_mu;

		Vect x = Vect::Zero(mu.size());
		eval_log_dens_constr(x, unconstr_mu, Q, log_det, Dxy, W, val);

		// set log det to zero because its already in val
		log_det = 0;

	} else {
		// solve linear system
		// returns vector mu, which is of the same size as rhs
		//solve_cholmod(Q, rhs, mu, log_det);
		solverQ->factorize_solve(Q, rhs, mu, log_det);

		log_det = 0.5 * (log_det);
	
#ifdef PRINT_MSG
	std::cout << "log det d : " << log_det << std::endl;
#endif

		// compute value
		val = -0.5 * mu.transpose()*(Q)*(mu);
	}

	time_solve_Q += omp_get_wtime();


#ifdef PRINT_TIMES
	if(MPI_rank == 0){
		std::cout << "time construct Q         = " << time_construct_Q << std::endl;
		std::cout << "time factorize & solve Q = " << time_solve_Q << std::endl;
	}
#endif

	/*std::cout << "in eval eval_denominator " << std::endl;

	std::cout << "rhs " << std::endl; std::cout <<  *rhs << std::endl;
	std::cout << "mu " << std::endl; std::cout << *mu << std::endl;
	std::cout << "Q " << std::endl; std::cout << Eigen::MatrixXd(*Q) << std::endl;

	std::cout << "log det d : " << log_det << std::endl;
	std::cout << "val d     : " << val << std::endl; */

}


PostTheta::~PostTheta(){

		delete solverQst;
		delete solverQ;		
}


// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //

// for Hessian approximation : 4-point stencil (2nd order)
// -> swap sign, invert, get covariance

// once converged call again : extract -> Q.xy -> selected inverse (diagonal), gives me variance wrt mode theta & data y

