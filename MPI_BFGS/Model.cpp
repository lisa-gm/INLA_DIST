#include "Model.h"



Model::Model(int ns, int nt, int nb, int no, MatrixXd B, VectorXd y, Vector theta_prior, string solver_type){
	
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

	printf("num solvers     : %d\n", num_solvers);

	solverQ   = new Solver*[threads_level1];
	solverQst = new Solver*[threads_level1];

	if(solver_type == "PARDISO"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new PardisoSolver();
			solverQst[i] = new PardisoSolver();
		}
	} else if(solver_type == "RGF"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new RGFSolver();
			solverQst[i] = new RGFSolver();
		}
	} 

	// set global counter to count function evaluations
	fct_count          = 0;	// initialise min_f_theta, min_theta
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

}


Model::Model(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, Vector theta_prior_, string solver_type_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), theta_prior(theta_prior_), solver_type(solver_type_) {

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

	printf("num solvers     : %d\n", num_solvers);

	solverQ   = new Solver*[threads_level1];
	solverQst = new Solver*[threads_level1];

	if(solver_type == "PARDISO"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new PardisoSolver();
			solverQst[i] = new PardisoSolver();
		}
	} else if(solver_type == "RGF"){
		for(int i = 0; i < threads_level1; i++){
			solverQ[i]   = new RGFSolver();
			solverQst[i] = new RGFSolver();
		}
	} 

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

}


Model::Model(int ns_, int nt_, int nb_, int no_, SpMat Ax_, VectorXd y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vector theta_prior_, string solver_type_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior(theta_prior_), solver_type(solver_type_)  {

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

	#ifdef PRINT_MSG
		printf("threads level 1 : %d\n", threads_level1);
	#endif

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
			solverQ[i]   = new RGFSolver();
			solverQst[i] = new RGFSolver();
		}
	} 

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count 		   = 0; // have internal iteration count equivalent to operator() calls

	// allocate memory for incoming array from master process
	theta_array = (double*)malloc(dim_th * sizeof(double));

	// TODO: better way to allocate correct size to mu??
	mu = Vector::Ones(n);

	//printf("leaving model constructor now.\n");

}

// ============================================================================================ //
// ALL FOLLOWING FUNCTIONS CONTRIBUTE TO THE EVALUATION OF F(THETA) 

void Model::ready(){

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	

	//cout << "entered ready function. Rank : " << rank << endl;

	MPI_Status status;

	// infinite loop thats just waiting for receives
	for (;;) {
		MPI_Recv(theta_array, dim_th, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		/* Check the tag of the received message */
		if (status.MPI_TAG == DIETAG) {
			//cout << "Process " << rank << " received DIETAG." << endl;
			return;
		}

		//cout << "Process " << rank << " received theta " << theta.transpose() << " from process 0" << endl;;

		// map to eigen vector format
		theta = Eigen::Map<Vector>(theta_array, dim_th);

		f_theta = evaluate(theta, mu);

		//cout << "computed f_theta in model " << f_theta << endl;

   		MPI_Send(&f_theta, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

	}


}

double Model::evaluate(Vector& theta, Vector& mu){

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


void Model::eval_log_prior(double& log_prior, double* thetai, double* thetai_original){

	log_prior = -0.5 * (*thetai - *thetai_original) * (*thetai - *thetai_original);

	#ifdef PRINT_MSG
		std::cout << "log prior for theta_i " << (*thetai) << " : " << (log_prior) << std::endl;
	#endif
}


void Model::eval_log_det_Qu(Vector& theta, double &log_det){

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


void Model::eval_likelihood(Vector& theta, double &log_det, double &val){
	
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


void Model::construct_Q_spatial(Vector& theta, SpMat& Qs){

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


void Model::construct_Q_spat_temp(Vector& theta, SpMat& Qst){

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


void Model::construct_Q(Vector& theta, SpMat& Q){

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


void Model::construct_b(Vector& theta, Vector &rhs){

	double exp_theta = exp(theta[0]);
	//double exp_theta = exp(3);

	if(ns == 0){
		rhs = exp_theta*B.transpose()*y;
	} else {
		rhs = exp_theta*Ax.transpose()*y;
	}

	#ifdef PRINT_MSG
		std::cout << "rhs(5) : " << rhs.head(5).transpose() << endl;
	#endif
}


void Model::eval_denominator(Vector& theta, double& log_det, double& val, SpMat& Q, Vector& rhs, Vector& mu){

	// construct Q_x|y,
	construct_Q(theta, Q);
	//Q->setIdentity();

	#ifdef PRINT_MSG
		printf("in eval denominator after construct_Q call.\n");
	#endif

	//  construct b_xey
	construct_b(theta, rhs);

	#ifdef PRINT_MSG
		printf("in eval denominator after construct_b call.\n");
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


Model::~Model(){

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

