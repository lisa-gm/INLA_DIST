#include "PostTheta.h"

//#include <likwid-marker.h>

PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, MatrixXd B_, Vect y_, Vect theta_prior_param_, string likelihood_, Vect extraCoeffVecLik_, string solver_type_, const bool constr_, const MatrixXd Dxy_, const bool validate_, const Vect w_) : ns(ns_), nt(nt_), nb(nb_), no(no_), B(B_), y(y_), theta_prior_param(theta_prior_param_), likelihood(likelihood_), extraCoeffVecLik(extraCoeffVecLik_), solver_type(solver_type_), constr(constr_), Dxy(Dxy_), validate(validate_), w(w_) {
//PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, MatrixXd B_, Vect y_, Vect theta_prior_param_, string likelihood_, string solver_type_, const bool constr_, const MatrixXd Dxy_, const bool validate_, const Vect w_) : ns(ns_), nt(nt_), nb(nb_), no(no_), B(B_), y(y_), theta_prior_param(theta_prior_param_), likelihood(likelihood_), solver_type(solver_type_), constr(constr_), Dxy(Dxy_), validate(validate_), w(w_) {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
	
	dim_th = 1;  			// only hyperparameter is the precision of the observations
	ns     = 0;
	nss    = 0;
	n      = nb;
	yTy    = y.dot(y);
	BTy    = B.transpose()*y;

	// inefficient but simplifies things later in the code 
	Ax = B.sparseView();

	// careful! Not the same Hyperparameter object ...
	// this apparently works somehow
	dimList.setZero();
	if(likelihood.compare("gaussian") == 0){
		dimList(0) = dim_th;
	} else {
		
		std::cout << "dim(extraCoeffVecLik) = " << extraCoeffVecLik.size() << std::endl;
		//std::cout << "extraCoeffVecLik(1:10) = " << extraCoeffVecLik.head(10).transpose() << std::endl;
	}
	dim_th = dimList.sum();

#if 0 
	//Hyperparameters theta_prior_test = Hyperparameters(0, "", dimList, theta_prior_param, theta_prior_param);
	theta_prior_test = new Hyperparameters(0, "", dimList, 'i', theta_dummy);
	theta_prior_test->update_interpretS(theta_prior_param);
	std::cout << "theta prior test " << theta_prior_test->flatten_interpretS().transpose() << std::endl;
#endif

#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
		printf("Eigen -- number of threads used : %d\n", Eigen::nbThreads( ));
#endif

	min_f_theta        = 1e10;			// initialise min_f_theta, min_theta

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	threads_level1 = omp_get_max_threads();
	threads_level2 = 1; // dummy, can use more ...

	if(MPI_rank == 0){
		printf("threads level 1 : %d\n", threads_level1);
	}

	dim_grad_loop      = 2*dim_th;
	no_f_eval          = 2*dim_th + 1;

	// one solver per thread, but not more than required
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
#endif

	if(solver_type == "PARDISO"){
		#pragma omp parallel
		{
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
		}
	} else if(solver_type == "BTA"){
		solverQ   = new RGFSolver(ns, nt, nb, no);
		solverQst = new RGFSolver(ns, nt, 0, no);
	} else {
		printf("wrong solver type! \n");
		exit(1);
	}

	// doesn't change throughout the algorithm. just set once
	Qb = 1e-3*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 

	prior = "gaussian";
	//std::cout << "Likelihood: " << likelihood << std::endl;


	// set global counter to count function evaluations
	fct_count          = 0;	// initialise min_f_theta, min_theta
	iter_count         = 0; // have internal iteration count equivalent to operator() calls
    iter_acc           = 0;

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

#ifdef RECORD_TIMES
	log_file_name = "log_file_per_iter_" + solver_type + "_ns" + std::to_string(ns) + "_nt" + std::to_string(nt) + "_nb" + std::to_string(nb) + "_" + std::to_string(MPI_size) + "_" + std::to_string(threads_level1) + "_" + std::to_string(threads_level2) + ".txt";
    std::ofstream log_file(log_file_name);
    log_file << "MPI_rank threads_level1 iter_count t_Ftheta_ext t_thread_nom t_priorHyp t_priorLat t_priorLatAMat t_priorLatChol t_likel t_thread_denom t_condLat t_condLatAMat t_condLatChol t_condLatSolve" << std::endl;
    log_file.close(); 
#endif

}

// spatial case
PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, Vect y_, SpMat c0_, SpMat g1_, SpMat g2_, Vect theta_prior_param_, string likelihood_, Vect extraCoeffVecLik_, string solver_type_, int dim_spatial_domain_, string manifold_, const bool constr_, const MatrixXd Dx_, const MatrixXd Dxy_, const bool validate_, const Vect w_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), theta_prior_param(theta_prior_param_), likelihood(likelihood_), extraCoeffVecLik(extraCoeffVecLik_), solver_type(solver_type_), dim_spatial_domain(dim_spatial_domain_), manifold(manifold_), constr(constr_), Dx(Dx_), Dxy(Dxy_), validate(validate_), w(w_) {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);  
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

	nss         = 0;
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

	// careful! Not the same Hyperparameter object ...
	// this apparently works somehow
	dimList.setZero();
	if(likelihood.compare("gaussian") == 0){
		dimList(0) = 1;
		dim_th     = 3;   			// 3 hyperparameters, precision for the observations, 2 for the spatial model
	} else {
		dim_th     = 2;
	}
	dimList(2) = 2;

	if(theta_prior_param.size() != dimList.sum()){
		printf("in spatial constructor. something wrong dimension theta!\n");
		exit(1);
	}
	dim_th = dimList.sum();

#if 0 
	//Hyperparameters theta_prior_test = Hyperparameters(0, "", dimList, theta_prior_param, theta_prior_param);
	theta_prior_test = new Hyperparameters(0, "", dimList, 'i', theta_dummy);
	theta_prior_test->update_interpretS(theta_prior_param);
	std::cout << "theta prior test " << theta_prior_test->flatten_interpretS().transpose() << std::endl;
#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	threads_level1 = omp_get_max_threads();
	threads_level2;

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
		#pragma omp parallel
		{
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
		}
	} else if(solver_type == "BTA"){
		//#pragma omp parallel
		//#pragma omp single
		//{
		//#pragma omp task
		//{
		solverQ   = new RGFSolver(ns, nt, nb, no);
		//}
		//#pragma omp task
		//{
		solverQst = new RGFSolver(ns, nt, 0, no);
		//}
		//}
	} else {
		printf("wrong solver type! \n");
		exit(1);
	}

	// construct Qx, Qxy using arbitrary theta to get sparsity pattern 
	// Qst reconstructed every time, otherwise sparse Kronecker needs to be rewritten.
	// but we need it now for sparsity structure of Qx, Qxy
	// get dimension of theta from theta_prior_param (has same dim. as theta)
	Vect theta_dummy(theta_prior_param.size());
	theta_dummy.setOnes();
	construct_Q_spatial(theta_dummy, Qu);

	int nnz = Qu.nonZeros();
	Qx.resize(n,n);
	Qx.reserve(nnz);

	for (int k=0; k<Qu.outerSize(); ++k){
		for (SparseMatrix<double>::InnerIterator it(Qu,k); it; ++it)
		{
		Qx.insert(it.row(),it.col()) = it.value();                 
		}
	}

	for(int i=nu; i < n; i++){
		// CAREFUL 1e-3 is arbitrary choice!!
		Qx.coeffRef(i,i) = 1e-3;
	}	  

	prior = "gaussian";

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count         = 0; // have internal iteration count equivalent to operator() calls
    iter_acc           = 0;


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

#ifdef RECORD_TIMES
	log_file_name = "log_file_per_iter_NEW_" + solver_type + "_ns" + std::to_string(ns) + "_nt" + std::to_string(nt) + "_nb" + std::to_string(nb) + "_" + std::to_string(MPI_size) + "_" + std::to_string(threads_level1) + "_" + std::to_string(threads_level2) + ".txt";
    std::ofstream log_file(log_file_name);
    log_file << "MPI_rank threads_level1 threads_level2 iter_count t_Ftheta_ext t_thread_nom t_priorHyp t_priorLat t_priorLatAMat t_priorLatChol t_likel t_thread_denom t_condLat t_condLatAMat t_condLatChol t_condLatSolve" << std::endl;
    log_file.close(); 
#endif


}

// constructor for spatial-temporal case
PostTheta::PostTheta(int ns_, int nt_, int nb_, int no_, SpMat Ax_, Vect y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vect theta_prior_param_, string likelihood_, Vect extraCoeffVecLik_, string solver_type_, int dim_spatial_domain_, string manifold_, const bool constr_, const MatrixXd Dx_, const MatrixXd Dxy_, const bool validate_, const Vect w_) : ns(ns_), nt(nt_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior_param(theta_prior_param_), likelihood(likelihood_), extraCoeffVecLik(extraCoeffVecLik_), solver_type(solver_type_), dim_spatial_domain(dim_spatial_domain_), manifold(manifold_), constr(constr_), Dx(Dx_), Dxy(Dxy_), validate(validate_), w(w_)  {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

	dim_th      = 4;    	 	// 4 hyperparameters, precision for the observations, 3 for the spatial-temporal model
	nss         = 0;
	nu          = ns*nt;
	n           = nb + ns*nt;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta

	// slow for large datasets!!
	if(validate){
		// CAREFUL: overwriting y here!
		y          = w.cwiseProduct(y);
		yTy         = y.dot(y);
		//std::cout << yTy << std::endl;

		// CAREFUL: overwriting Ax here!
		for(int i=0; i<Ax.rows(); i++){
			Ax.row(i) *= w(i);
		}

		AxTy        = Ax.transpose()*y;
		AxTAx       = Ax.transpose()*Ax;
		
		w_sum       = w.sum();

	} else {
		yTy         = y.dot(y);
		AxTy		= Ax.transpose()*y;
		AxTAx       = Ax.transpose()*Ax;
	}

#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
		printf("Eigen -- number of threads used : %d\n", Eigen::nbThreads( ));
#endif

	// careful! Not the same Hyperparameter object ...
	// this apparently works somehow
	dimList.setZero();
	if(likelihood.compare("gaussian") == 0){
		dimList(0) = 1;
	}
	dimList(1) = 3;

	if(theta_prior_param.size() != dimList.sum()){
		printf("in spatial-temporal constructor. something wrong dimension theta!\n");
		exit(1);
	}
	dim_th = dimList.sum();

#if 0
	//Hyperparameters theta_prior_test = Hyperparameters(0, "", dimList, theta_prior_param, theta_prior_param);
	theta_prior_test = new Hyperparameters(0, "", dimList, 'i', theta_dummy);
	theta_prior_test->update_interpretS(theta_prior_param);
	std::cout << "theta prior test " << theta_prior_test->flatten_interpretS().transpose() << std::endl;
#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	threads_level1 = omp_get_max_threads();
	//threads_level1 = 2;

	threads_level2;

	#pragma omp parallel
    {  
   		threads_level2 = omp_get_max_threads();
    }
	//threads_level2 = 1;

	likelihood = "gaussian";

	
#ifdef PRINT_MSG
	if(MPI_rank == 0){
		printf("threads level 1 : %d\n", threads_level1);
		printf("threads level 2 : %d\n", threads_level2);
	}
#endif

	dim_grad_loop      = 2*dim_th;
	no_f_eval          = 2*dim_th + 1;

	// one solver per thread, but not more than required
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	if(solver_type == "PARDISO"){
		#pragma omp parallel
		{
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
		}
	} else if(solver_type == "BTA"){
		#pragma omp parallel
		{	
		if(omp_get_thread_num() == 0){	
			solverQst = new RGFSolver(ns, nt, 0, no);
		} 
		if(omp_get_thread_num() == 1 || threads_level1 == 1){
			solverQ = new RGFSolver(ns, nt, nb, no);  // solver for prior random effects. best way to handle this? 
		}
		}
	} else {
		printf("wrong solver type! \n");
		exit(1);
	}

	// construct Qx, Qxy using arbitrary theta to get sparsity pattern 
	// Qst reconstructed every time, otherwise sparse Kronecker needs to be rewritten.
	// but we need it now for sparsity structure of Qx, Qxy
	// get dimension of theta from theta_prior_param (has same dim. as theta)
	Vect theta_dummy(theta_prior_param.size());
	theta_dummy.setOnes();
	construct_Q_spat_temp(theta_dummy, Qst);

	int nnz = Qst.nonZeros();
	Qx.resize(n,n);
	Qx.reserve(nnz);

	for (int k=0; k<Qst.outerSize(); ++k){
		for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
		{
		Qx.insert(it.row(),it.col()) = it.value();                 
		}
	}

	for(int i=nu; i < n; i++){
		// CAREFUL 1e-3 is arbitrary choice!!
		Qx.insert(i,i) = 1e-3;
	}
	
	// set prior to be gaussian
	//prior = "gaussian";
	prior = "pc";

	if(MPI_rank == 0){
		std::cout << "Prior : " << prior << std::endl;	
	}

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count         = 0; // have internal iteration count equivalent to operator() calls
	iter_acc           = 0;

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

#ifdef RECORD_TIMES
    //if((MPI_rank) == 0){
    	log_file_name = "log_file_per_iter_" + solver_type + "_ns" + std::to_string(ns) + "_nt" + std::to_string(nt) + "_nb" + std::to_string(nb) + "_" + std::to_string(MPI_rank) + "_" + std::to_string(MPI_size) + "_" + std::to_string(threads_level1) + "_" + std::to_string(threads_level2) + ".txt";
    	std::ofstream log_file(log_file_name);
    	log_file << "MPI_rank threads_level1 threads_level_2 iter_count t_Ftheta_ext t_thread_nom t_priorHyp t_priorLat t_priorLatAMat t_priorLatChol t_likel t_thread_denom t_condLat t_condLatAMat t_condLatChol t_condLatSolve" << std::endl;
    	log_file.close();
    //}	
#endif

}

// constructor for spatial-temporal field plus spatial field ie. eta = A.st*u.st + A.s*u.s + B*\beta
// assume that Ax_ contains A.st, A.s, B (in that order)
// use c0, g1, g2 to construct spatial field (order 2)
// neglegt constraint case for now -> later add constraints for spatial-temporal and spatial field separately
PostTheta::PostTheta(int ns_, int nt_, int nss_, int nb_, int no_, SpMat Ax_, Vect y_, SpMat c0_, SpMat g1_, SpMat g2_, SpMat g3_, SpMat M0_, SpMat M1_, SpMat M2_, Vect theta_prior_param_, string likelihood_, Vect extraCoeffVecLik_, string solver_type_, int dim_spatial_domain_, string manifold_, const bool constr_, const MatrixXd Dx_, const MatrixXd Dxy_, const bool validate_, const Vect w_) : ns(ns_), nt(nt_), nss(nss_), nb(nb_), no(no_), Ax(Ax_), y(y_), c0(c0_), g1(g1_), g2(g2_), g3(g3_), M0(M0_), M1(M1_), M2(M2_), theta_prior_param(theta_prior_param_), likelihood(likelihood_), extraCoeffVecLik(extraCoeffVecLik_), solver_type(solver_type_), dim_spatial_domain(dim_spatial_domain_), manifold(manifold_), constr(constr_), Dx(Dx_), Dxy(Dxy_), validate(validate_), w(w_)  {

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

	// careful! Not the same Hyperparameter object ...
	// this apparently works somehow
	dimList.setZero();
	if(likelihood.compare("gaussian") == 0){
		dimList(0) = 1;
	}
	dimList(1) = 3;

	if(nss == 0){
		dim_th      = 4;    	 	// 6 hyperparameters, precision for the observations, 3 for the spatial-temporal model, 2 spatial model
	}else if(nss > 0){
		dim_th      = 6;
		dimList(2)  = 2;
		if(MPI_rank == 0){
			printf("With additional spatial field!\n");
		}
	} else{
		printf("invalid value for nss = %d\n!!", nss);
		exit(1);
	}

	if(theta_prior_param.size() != dimList.sum()){
		printf("in spatial-temporal constructor. something wrong dimension theta!\n");
		std::cout << "dim(theta prior param) = " << theta_prior_param.size() << std::endl;
		std::cout << "dimList = " << dimList.transpose() << std::endl;
		exit(1);
	}

	dim_th = dimList.sum();

    if(MPI_rank==0){
		std::cout << "manifold: " << manifold << std::endl;
	}
	
	nst         = ns*nt;
	nu          = nst   + nss;
	n           = ns*nt + nss + nb;
	min_f_theta = 1e10;			// initialise min_f_theta, min_theta

	// slow for large datasets!!
	if(validate){
		printf("validate is TRUE. Not implemented!\n");
		exit(1);
	} else {
		yTy         = y.dot(y);
		AxTy		= Ax.transpose()*y;
		AxTAx       = Ax.transpose()*Ax;
	}

	likelihood = "gaussian";

#ifdef PRINT_MSG
		printf("yTy : %f\n", yTy);
		printf("norm(AxTy) : %f\n", AxTy.norm());
		printf("Eigen -- number of threads used : %d\n", Eigen::nbThreads( ));
#endif

#if 0
	Vect theta_dummy2(theta_prior_param.size());
	theta_dummy2.setOnes();
	if(MPI_rank == 0){
		std::cout << "theta dummy : " << theta_dummy.transpose() << std::endl;
		std::cout << "theta prior param : " << theta_prior_param.transpose() << std::endl;

	}

	// CAREFUL: theta_dummy gets changed throughout -> as input is by reference ...
	//Hyperparameters theta_prior_test = Hyperparameters(0, "", dimList, theta_prior_param, theta_prior_param);
	theta_prior_test = new Hyperparameters(0, "", dimList, 'i', theta_dummy);
	theta_prior_test->update_interpretS(theta_prior_param);
	std::cout << "theta prior test " << theta_prior_test->flat.transpose() << std::endl;

	//Hyperparameters theta_prior_test2 = create_hp(theta_prior_param, 'i');
	//theta_prior_test2.update_interpretS(Vect::Ones(dim_th));
	// TODO: .flat_interpretS doesn't (I guess no access to reference ... )
	//std::cout << "theta prior test 2 : " << theta_prior_test2.flatten_interpretS().transpose() << std::endl;
#endif

	// set up PardisoSolver class in constructor 
	// to be independent of BFGS loop
	threads_level1 = omp_get_max_threads();
	//threads_level1 = 2;
	threads_level2;

	#pragma omp parallel
    {  
   		threads_level2 = omp_get_max_threads();
    }
	//threads_level2 = 1;

#ifdef PRINT_MSG
	if(MPI_rank == 0){
		printf("threads level 1 : %d\n", threads_level1);
		printf("threads level 2 : %d\n", threads_level2);
	}
#endif

#if 1

	dim_grad_loop      = 2*dim_th;
	no_f_eval          = 2*dim_th + 1;

	// one solver per thread, but not more than required
	//num_solvers        = std::min(threads_level1, dim_grad_loop);
	// makes sense to create more solvers than dim_grad_loop for hessian computation later.
	// if num_solver < threads_level1 hess_eval will fail!
	num_solvers        = threads_level1;

	#ifdef PRINT_MSG
		printf("num solvers     : %d\n", num_solvers);
	#endif

	if(solver_type == "PARDISO"){
		#pragma omp parallel
		{
		solverQ   = new PardisoSolver(MPI_rank);
		solverQst = new PardisoSolver(MPI_rank);
		}
	} else if(solver_type == "BTA"){
		#pragma omp parallel
		{	
		if(omp_get_thread_num() == 0){	
			solverQst = new RGFSolver(ns, nt, nss, no);
		} 
		if(omp_get_thread_num() == 1 || threads_level1 == 1){
			solverQ = new RGFSolver(ns, nt, nb+nss, no);  // solver for prior random effects. best way to handle this? 
		}
		}
	} else {
		printf("wrong solver type! \n");
		exit(1);
	}

	// construct Qx, Qxy using arbitrary theta to get sparsity pattern 
	// Qst reconstructed every time, otherwise sparse Kronecker needs to be rewritten.
	// but we need it now for sparsity structure of Qx, Qxy
	// get dimension of theta from theta_prior_param (has same dim. as theta)
	Vect theta_dummy(theta_prior_param.size());
	//theta_dummy.setOnes();
	theta_dummy << 1.386796, -4.434666, 0.6711493, 1.632289, -5.058083, 2.664039;
    //theta_dummy << 1.422895, -4.502342,  0.623269,  1.652469, -4.611303, 2.238631;
	construct_Q_spat_temp(theta_dummy, Qst);

	nnz_Qst = Qst.nonZeros();
	Qx.resize(n,n);
	Qx.reserve(nnz_Qst);

	for (int k=0; k<Qst.outerSize(); ++k){
		for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
		{
		Qx.insert(it.row(),it.col()) = it.value();                 
		}
	}

	if(nss > 0){
		// need to be careful about what theta values are accessed!! now dimension larger
		Vect theta_dummy_spat = theta_dummy(seq(3,5));
		construct_Q_spatial(theta_dummy_spat, Qs);
		nnz_Qs = Qs.nonZeros();

		// insert entries of Qs
		for (int k=0; k<Qs.outerSize(); ++k){
			for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
			{
			Qx.insert(it.row()+nst,it.col()+nst) = it.value();                 
			}
		}
	}

	for(int i=nu; i < n; i++){
		// CAREFUL 1e-3 is arbitrary choice!!
		Qx.insert(i,i) = 1e-3;
	}

	//std::cout << "Qx = \n" << Qx << std::endl;

	//std::string Qprior_fileName = "Q_prior.txt";
	//SpMat A_lower = Qx.triangularView<Lower>();
		
	/*std::string Qprior_fileName = "Qxy_" + to_string(n) + ".txt";
	Qxy = Qx + exp(theta_dummy[0])*AxTAx;
	SpMat A_lower = Qxy.triangularView<Lower>();

	
	int n = A_lower.cols();
	int nnz = A_lower.nonZeros();

	ofstream sol_file(Qprior_fileName);
	sol_file << n << "\n";
	sol_file << n << "\n";
	sol_file << nnz << "\n";

	for (int i = 0; i < nnz; i++){
		sol_file << A_lower.innerIndexPtr()[i] << "\n";
	}   
	for (int i = 0; i < n+1; i++){
			sol_file << A_lower.outerIndexPtr()[i] << "\n";
	}     
	for (int i = 0; i < nnz; i++){
		sol_file << std::setprecision(15) << A_lower.valuePtr()[i] << "\n";
	}

	sol_file.close();
	std::cout << "wrote to file : " << Qprior_fileName << std::endl;
	
	exit(1);*/
	
	// set prior to be gaussian
	//prior = "gaussian";
	prior = "pc";

	if(MPI_rank == 0){
		std::cout << "Prior : " << prior << std::endl;	
	}

	// set global counter to count function evaluations
	fct_count          = 0;
	iter_count         = 0; // have internal iteration count equivalent to operator() calls
	iter_acc           = 0;

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

#ifdef RECORD_TIMES
    //if((MPI_rank) == 0){
    	log_file_name = "log_file_per_iter_" + solver_type + "_ns" + std::to_string(ns) + "_nt" + std::to_string(nt) + "_nb" + std::to_string(nb) + "_" + std::to_string(MPI_rank) + "_" + std::to_string(MPI_size) + "_" + std::to_string(threads_level1) + "_" + std::to_string(threads_level2) + ".txt";
    	std::ofstream log_file(log_file_name);
    	log_file << "MPI_rank threads_level1 threads_level_2 iter_count t_Ftheta_ext t_thread_nom t_priorHyp t_priorLat t_priorLatAMat t_priorLatChol t_likel t_thread_denom t_condLat t_condLatAMat t_condLatChol t_condLatSolve" << std::endl;
    	log_file.close();
    //}	
#endif

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

	if(iter_count == 0){
		t_bfgs_iter = -omp_get_wtime();
	}

	double t_f_grad_f = -omp_get_wtime();

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
	//double eps = 1e-5;
	double eps = 1e-3;
	// projection matrix G, either Identity or other orthonormal basis (from computeG function)
	//G = MatrixXd::Identity(dim_th, dim_th);

	if(MPI_rank == 0 && printed_eps_flag == false){
		std::cout << "Finite Difference    : h = " << std::scientific << eps << std::endl;
		printed_eps_flag = true;
	}


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
#ifdef RECORD_TIMES
		t_Ftheta_ext = -omp_get_wtime();
#endif
		f_temp_list_loc(0) = eval_post_theta(theta, mu);
		//std::cout << "theta   : " << std::right << std::fixed << theta.transpose() << std::endl;
		//std::cout << "before record times section." << std::endl;

#ifdef RECORD_TIMES		
		t_Ftheta_ext += omp_get_wtime();

		// for now write to file. Not sure where the best spot would be.
		// file contains : MPI_rank iter_count l1t l2t t_Ftheta_ext t_priorHyp t_priorLat t_likel t_condLat
		//std::cout << log_file_name << " " << iter_count << " " << t_Ftheta_ext << " " << t_thread_nom << " " << t_priorHyp << " " << t_priorLat << " " << t_priorLatAMat << " ";
		//std::cout << t_priorLatChol << " " << t_likel << " " << t_thread_denom << " " << t_condLat << " " << t_condLatAMat << " " << t_condLatChol << " " << t_condLatSolve << std::endl;
		record_times(log_file_name, iter_count, t_Ftheta_ext, t_thread_nom, t_priorHyp, t_priorLat, t_priorLatAMat, t_priorLatChol, t_likel, t_thread_denom, t_condLat, t_condLatAMat, t_condLatChol, t_condLatSolve);
#endif
		
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
				std::cout <<"i = " << i << ", i / divd = " << i / divd << ", rank " << MPI_rank << std::endl;
					//std::cout << "i : " << i << " and k : " << k << std::endl;
#endif

				Vect theta_forw(dim_th);
				Vect mu_dummy(n);

				theta_forw = theta + eps*G.col(k);
#ifdef RECORD_TIMES
		        t_Ftheta_ext = -omp_get_wtime();
#endif			
	
				f_temp_list_loc(i) = eval_post_theta(theta_forw, mu_dummy);
#ifdef RECORD_TIMES
                t_Ftheta_ext += omp_get_wtime();
           		// for now write to file. Not sure where the best spot would be.
           		// file contains : MPI_rank iter_count l1t t_Ftheta_ext t_priorHyp t_priorLat t_likel t_condLat
                record_times(log_file_name, iter_count, t_Ftheta_ext, t_thread_nom, t_priorHyp, t_priorLat, t_priorLatAMat, t_priorLatChol,
                                                t_likel, t_thread_denom, t_condLat, t_condLatAMat, t_condLatChol, t_condLatSolve);
#endif
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
#ifdef RECORD_TIMES
                                t_Ftheta_ext = -omp_get_wtime();
#endif
				f_temp_list_loc(i) = eval_post_theta(theta_backw, mu_dummy);
#ifdef RECORD_TIMES
                		t_Ftheta_ext += omp_get_wtime();
      			        // for now write to file. Not sure where the best spot would be.
                		// file contains : MPI_rank iter_count l1t t_Ftheta_ext t_priorHyp t_priorLat t_likel t_condLat
                		record_times(log_file_name, iter_count, t_Ftheta_ext, t_thread_nom, t_priorHyp, t_priorLat, t_priorLatAMat, t_priorLatChol,
                                                t_likel, t_thread_denom, t_condLat, t_condLatAMat, t_condLatChol, t_condLatSolve);
#endif

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

	// print all theta's who result in a new minimum value for f(theta)
	if(f_theta < min_f_theta){
		min_f_theta = f_theta;
		iter_acc += 1;
		double t_bfgs_iter_temp = omp_get_wtime() + t_bfgs_iter; 
		if(MPI_rank == 0){
			Vect theta_interpret(dim_th);
			convert_theta2interpret(theta, theta_interpret);
			std::cout << "theta interpret: " << std::right << std::fixed << theta_interpret.transpose() << "    f_theta: " << std::right << std::fixed << f_theta << std::endl;
		}
	}

#ifdef PRINT_MSG
	if(MPI_rank == 0){  
        //std::cout << "f_theta : " << std::right << std::fixed << std::setprecision(12) << f_theta << std::endl;
        //std::cout << "grad    : " << std::right << std::fixed << std::setprecision(4) << grad.transpose()  << std::endl;
    }
#endif

	t_f_grad_f += omp_get_wtime();

	if(MPI_rank == 0){
		std::cout << "time f + grad f eval : " << t_f_grad_f << std::endl;
	}

	return f_theta;

}

#ifdef DATA_SYNTHETIC

double PostTheta::compute_error_bfgs(Vect& theta){
        Vect theta_original(4);
        theta_original << 1.386294, -5.882541,  1.039721,  3.688879;
        double err = (theta - theta_original).norm();

        return err;
}

#endif

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
    	// check if ThetaDiff = G*R
    	//std::cout << "norm(ThetaDiff - G*R) = " << (ThetaDiff - G*R).norm() << std::endl;
    	std::cout << "G = \n" << G << std::endl;
    }
#endif

}

#endif


#if 0 // reactivate if needed
// create hyperparameter object
Hyperparameters PostTheta::create_hp(Vect param, char scale){

	Vect theta_vec     = Vect::Ones(param.size());  
	Hyperparameters theta_param = Hyperparameters(dim_spatial_domain, manifold, dimList, scale, theta_vec);

	// if scale == m -> input parameters in model scale 
	if(scale == 'm'){
		theta_param.update_modelS(param);
	// if scale == i -> input parameter in interpretable scale
	} else if(scale == 'i'){
		theta_param.update_interpretS(param);
	} else {
		printf("in create hp function. invalid scale!\n");
		exit(1);
	}
	std::cout << "theta_param.flatten_interpretS: " << theta_param.flatten_interpretS().transpose() << std::endl;

	return theta_param;
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
void PostTheta::convert_theta2interpret(Vect& theta, Vect& theta_interpret){

	int count = 0;
	if(likelihood.compare("gaussian") == 0){
		theta_interpret[count] = theta[count];
		count++;
	}

	if(ns == 0){
#ifdef PRINT_MSG
		if(MPI_rank == 0){
			printf("No conversion needed!\n");
		}
#endif
	}else if(ns > 0 && nt <= 1){
		convert_theta2interpret_spat(theta[count], theta[count+1], theta_interpret[count], theta_interpret[count+1]);
		count = count + 2;

	} else if(ns > 0 && nt > 1){
		convert_theta2interpret_spatTemp(theta[count], theta[count+1], theta[count+2], theta_interpret[count], theta_interpret[count+1], theta_interpret[count+2]);
		count = count + 3;
		
		if(nss > 0){
			convert_theta2interpret_spat(theta[count], theta[count+1], theta_interpret[count], theta_interpret[count+1]);
			count = count + 2;
		}
	}

	if(count != dim_th){
		printf("count = %d, dim(theta) = %d. Not matching!\n", count, dim_th);
		exit(1);
	}

	//std::cout << "theta           = " << theta.transpose() << std::endl;
	//std::cout << "theta interpret = " << theta_interpret.transpose() << std::endl;

}

// umbrella function that calls respective sub functions 
// by assumption ns > 0 -> otherwise nothing to convert but check anyway
void PostTheta::convert_interpret2theta(Vect& theta_interpret, Vect& theta){

	int count = 0;
	if(likelihood.compare("gaussian") == 0){
		theta[count] = theta_interpret[count];
		count++;
	}

	if(ns == 0){
#ifdef PRINT_MSG
		if(MPI_rank == 0){
			printf("No conversion needed!\n");
		}
#endif
	}else if(ns > 0 && nt <= 1){
		convert_interpret2theta_spat(theta_interpret[count], theta_interpret[count+1], theta[count], theta[count+1]);
		count = count + 2;

	} else if(ns > 0 && nt > 1){
		convert_interpret2theta_spatTemp(theta_interpret[count], theta_interpret[count+1], theta_interpret[count+2], theta[count], theta[count+1], theta[count+2]);
		count = count + 3;
		
		if(nss > 0){
			convert_interpret2theta_spat(theta_interpret[count], theta_interpret[count+1], theta[count], theta[count+1]);
			count = count + 2;
		}
	}

	if(count != dim_th){
		printf("count = %d, dim(theta) = %d. Not matching!\n", count, dim_th);
		exit(1);
	}

	//std::cout << "theta           = " << theta.transpose() << std::endl;
	//std::cout << "theta interpret = " << theta_interpret.transpose() << std::endl;

}



void PostTheta::convert_theta2interpret_spatTemp(double lgamE, double lgamS, double lgamT, double& ranS, double& ranT, double& sigU){
	double alpha_t = 1; 
	double alpha_s = 2;
	double alpha_e = 1;

	double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
	//double nu_s  = alpha   - 1;
	double nu_s  = alpha - 1; 
	double nu_t  = alpha_t - 0.5;

	double gE = exp(lgamE);
	double gS = exp(lgamS);
	double gT = exp(lgamT);

	ranS = log(sqrt(8*nu_s)/gS);
	ranT = log(gT*sqrt(8*nu_t)/(pow(gS, alpha_s)));
	
	if(manifold == "sphere"){
		double cR_t = std::tgamma(alpha_t - 1.0/2.0)/(std::tgamma(alpha_t)*pow(4*M_PI, 1.0/2.0));
		double cS = 0.0;
		for(int k=0; k<50; k++) // compute 1st 100 terms of infinite sum
		{  
			cS += (2.0*k + 1) / (4*M_PI* pow(pow(gS, 2) + k*(k+1), alpha));
		}
		//printf("cS : %f\n", cS);
		sigU = log(sqrt(cR_t*cS)/(gE*sqrt(gT)));

	} else {
		double c1_scaling_const = pow(4*M_PI, dim_spatial_domain/2.0) * pow(4*M_PI, 1.0/2.0); // second for temporal dim
		//double c1_scaling_const = pow(4*M_PI, 1.5);
		//std::cout << "c1_scaling_const theta2interpret : " << c1_scaling_const << std::endl;	
		double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*c1_scaling_const);
		sigU = log(sqrt(c1)/((gE*sqrt(gT))*pow(gS,alpha-dim_spatial_domain/2)));
	}


}

void PostTheta::convert_interpret2theta_spatTemp(double ranS, double ranT, double sigU, double& lgamE, double& lgamS, double& lgamT){
	double alpha_t = 1; 
	double alpha_s = 2;
	double alpha_e = 1;

	double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
	//double nu_s  = alpha   - 1;
	double nu_s  = alpha - 1; 
	double nu_t  = alpha_t - 0.5; // because dim temporal domain always 1

	lgamS = 0.5*log(8*nu_s) - ranS;
	lgamT = ranT - 0.5*log(8*nu_t) + alpha_s * lgamS;

	if(manifold == "sphere"){
		double cR_t = std::tgamma(alpha_t - 1.0/2.0)/(std::tgamma(alpha_t)*pow(4*M_PI, 1.0/2.0));
		double cS = 0.0;
		double t_loop = - omp_get_wtime();
		for(int k=0; k<50; k++) // compute 1st 100 terms of infinite sum
		{  
			cS += (2.0*k + 1) / (4*M_PI* pow(pow(exp(lgamS), 2) + k*(k+1), alpha));
		}
		t_loop += omp_get_wtime();
		// printf("cS: %f\n", cS);
		//printf("sphere. c3 : %f, t loop : %f\n",  0.5*log(cR_t) + 0.5*log(cS), t_loop);
		lgamE = 0.5*log(cR_t) + 0.5*log(cS) - 0.5*lgamT - sigU;

	} else {
		//double c1_scaling_const = pow(4*M_PI, dim_spatial_domain/2.0) * pow(4*M_PI, 1.0/2.0); // second for temporal dim
		double c1_scaling_const = pow(4*M_PI, 1.5);
		//std::cout << "c1_scaling_const interpret2theta : " << c1_scaling_const << std::endl;
		double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*c1_scaling_const);
		//printf("R^d. c3 : %f\n", 0.5*log(c1) - (alpha-dim_spatial_domain/2)*lgamS);
		lgamE = 0.5*log(c1) - 0.5*lgamT - (alpha-dim_spatial_domain/2)*lgamS - sigU;
	}

}

#if 0
void PostTheta::convert_theta2interpret_spatTemp(double lgamE, double lgamS, double lgamT, double& ranT, double& ranS, double& sigU){
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


void PostTheta::convert_interpret2theta_spatTemp(double ranT, double ranS, double sigU, double& lgamE, double& lgamS, double& lgamT){
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
#endif

// just SPATIAL model conversion

/*
### ltheta = c(log_tau, log_kappa)
theta2interpret_spatial <- function(ltheta, d=2){
    alpha = 2
    nu.s = alpha - d/2

    lsigU = 0.5*(lgamma(nu.s) - (lgamma(alpha) + 0.5*d*log(4*pi) + 2*nu.s*ltheta[2] + 2*ltheta[1]))
    lranS = 0.5*log(8*nu.s) - ltheta[2]
    return(c(lsigU, lranS))
}
*/
void PostTheta::convert_theta2interpret_spat( double lgamE, double lgamS, double& lranS, double& lsigU){
	double dim_spat = 2.0;
	double alpha    = 2.0;
	double nu_s     = alpha - dim_spat/2.0;

	lsigU = 0.5*(std::tgamma(nu_s) - (std::tgamma(alpha) + 0.5*dim_spat*log(4*M_PI) + 2*nu_s*lgamS + 2*lgamE));
	lranS = 0.5*log(8*nu_s) - lgamS;
}

/*
interpret2theta_spatial <- function(log_theta.interpret, d=2) {
### note that theta is log(sigma.s, range.s)
            alpha = 2
            nu.s = alpha-d/2;
            log_kappa = 0.5*log(8*nu.s) - log_theta.interpret[2]
            log_tau   = 0.5*(lgamma(nu.s) - (lgamma(alpha) + 0.5*d*log(4*pi) + 2*nu.s*log_kappa + 2*log_theta.interpret[1]))
    return(c(log_tau, log_kappa))
}
*/
void PostTheta::convert_interpret2theta_spat(double lranS, double lsigU, double& lgamE, double& lgamS){
	// assume lranS, lsigU to be in log-scale, lgamS = lkappa, lgamE = ltau
	// assuming spatial dimension d=2 & alpha = 2
	double dim_spat = 2;
	double alpha    = 2;
	double nu_s  = alpha - dim_spat/2.0;

	lgamS = 0.5*log(8*nu_s) - lranS;
	lgamE = 0.5*(std::tgamma(nu_s) - (std::tgamma(alpha) + 0.5*dim_spat*log(4*M_PI) + 2*nu_s*lgamS + 2*lsigU));
}




// ============================================================================================ //
// FUNCTIONS TO BE CALLED AFTER THE BFGS SOLVER CONVERGED

// Gaussian case
void PostTheta::get_mu(Vect& theta, Vect& mu){

#ifdef PRINT_MSG
		std::cout << "get_mu()" << std::endl;
#endif

	double f_theta = eval_post_theta(theta, mu);

#ifdef PRINT_MSG
		std::cout << "mu(-10:end) :" << mu.tail(min(10, n)).transpose() << std::endl;
#endif
}

// non-Gaussian case there are no hyperparameters theta -> no minimization over theta needed 
// just mode of conditional needs to be found -> initial guess x -> gets overwritten
#if 0
void PostTheta::get_mu(Vect& theta, Vect& x){

#ifdef PRINT_MSG
		std::cout << "get_mu()" << std::endl;
#endif

	if()
	// use Qb as prior precision matrix
	SpMat Qprior = Qb;
	NewtonIter(extraCoeffVecLik, Qprior, x);

#ifdef PRINT_MSG
		std::cout << "x(-10:end) :" << x.tail(min(10, n)).transpose() << std::endl;
#endif
}

#endif

Vect PostTheta::get_grad(){
	return t_grad;
}

void PostTheta::get_Qprior(Vect theta, SpMat& Qprior){
	
	// TODO: not very clean ... Qprior <-> Qx
	construct_Qprior(theta, Qx);
	Qprior = Qx;
}


MatrixXd PostTheta::get_Covariance(Vect theta, double eps){

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


MatrixXd PostTheta::get_Cov_interpret_param(Vect interpret_theta, double eps){

	int dim_th = interpret_theta.size();

	/*
	if(interpret_theta.size() != 4){
		std::cout << "dim(interpret_theta) = " << interpret_theta.size() << ", should be 4!" << std::endl;
		exit(1);
	} else {
		dim_th = 4;
	}
	*/

	MatrixXd hess(dim_th,dim_th);

	// evaluate hessian
	double timespent_hess_eval = -omp_get_wtime();
	hess = hess_eval_interpret_theta(interpret_theta, eps);

	timespent_hess_eval += omp_get_wtime();

#ifdef PRINT_TIMES
		std::cout << "time spent hessian evaluation: " << timespent_hess_eval << std::endl;
#endif 

	if(MPI_rank == 0){
		std::cout << "estimated hessian         : \n" << hess << std::endl; 
		std::cout << "eigenvalues hessian : \n" << hess.eigenvalues().real() << std::endl;
		//std::cout << "eps : " << eps << endl;
	}

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

void PostTheta::get_marginals_f(Vect& theta, Vect& mu, Vect& vars){
	
	SpMat Q(n, n);
	construct_Q(theta, mu, Q);

	/*
	MatrixXd Q_d = MatrixXd(Q);
	//std::cout << "Q.bottomRightCorner(10,10) : \n" << Q_d.bottomRightCorner(10,10) << std::endl;
	MatrixXd Q_fe = Q_d.bottomRightCorner(nb,nb);
	std::cout << "dim(Q(FE, FE))    : " << Q_fe.rows() << " " << Q_fe.cols() << std::endl;
	std::cout << "Q_fe : \n" << Q_fe << std::endl;
	Vect eig =  Q_fe.eigenvalues().real();
	std::cout << "\neigenvalues Q(FE, FE) : \n" << eig.transpose() << std::endl;
	Vect norm_eig = eig / eig.minCoeff();
	std::cout << "\nnormalized eigenvalues Q(FE, FE) : \n" << norm_eig.transpose() << std::endl;

	MatrixXd Cov_fe = Q_fe.inverse();
	std::cout << "\nCovariance mat FE = \n" << Cov_fe << std::endl;

	// compute correlation matrix : cor(x_i, x_j) = cov(x_i, x_j) / sqrt(cov(x_i, x_i)*cov(x_j, x_j))
	MatrixXd Cor_fe(Cov_fe.rows(), Cov_fe.cols());
	for(int i = 0; i<Cov_fe.rows(); i++){
		for(int j=0; j<Cov_fe.cols(); j++){
			Cor_fe(i,j) = Cov_fe(i,j) / sqrt(Cov_fe(i,i)*Cov_fe(j,j));
		}
	}

	std::cout << "Correlation mat FE = \n" << Cor_fe << "\n" << std::endl;
	*/

	//std::cout << "exp(theta[0])*AxTAx.bottomRightCorner(10,10) : \n" << exp(theta[0])*MatrixXd(AxTAx).bottomRightCorner(10,10);
	

#ifdef PRINT_MSG
		std::cout << "after construct Q in get get_marginals_f" << std::endl;
#endif

	double timespent_sel_inv_pardiso = -omp_get_wtime();

	if(constr == true){
		#pragma omp parallel
		{
		if(omp_get_thread_num() == 1 || threads_level1 == 1)
		{
			MatrixXd V(n, Dxy.rows());
			solverQ->selected_inversion_w_constr(Q, Dxy, vars, V);
			MatrixXd W = Dxy*V;
			MatrixXd S = W.inverse()*V.transpose();

			Vect update_vars(n);
			for(int i=0; i<n; i++){
				update_vars[i] = V.row(i)*S.col(i);
			}

			//std::cout << "\nvars        = " << vars.transpose() << std::endl;			
			//std::cout << "update_vars = " << update_vars.tail(10).transpose() << std::endl;
			vars = vars - update_vars;
			//std::cout << "vars        = " << vars.tail(10).transpose() << std::endl;			
		}
		}

	} else {
		// nested parallelism, want to call this with 1 thread of omp level 1
		#pragma omp parallel
		{
		if(omp_get_thread_num() == 1 || threads_level1 == 1)
		{
			solverQ->selected_inversion(Q, vars);
		}
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
MatrixXd PostTheta::hess_eval(Vect theta, double eps){

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
		std::cout << "in Hessian. task_to_rank_list : " << task_to_rank_list.transpose() << std::endl;
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
MatrixXd PostTheta::hess_eval_interpret_theta(Vect interpret_theta, double eps){

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
		std::cout << "in Hessian interpret. task_to_rank_list : " << task_to_rank_list.transpose() << std::endl;	
		std::cout << "Loop Dim: " << loop_dim << std::endl;
	}
#endif

    double time_omp_task_hess = - omp_get_wtime();

    // compute f(theta) only once.
	if(MPI_rank == task_to_rank_list[0]){
		Vect mu_tmp(n); 
		// convert interpret_theta to theta
		Vect theta(dim_th);
		convert_interpret2theta(interpret_theta, theta);

		double f_theta = eval_post_theta(theta, mu_tmp);
	    f_i_i_loc.row(1) = f_theta * Vect::Ones(dim_th).transpose(); 
    }
    counter++;

    for(int k = 0; k < loop_dim; k++){          

        // row index is integer division k / dim_th
        int i = k/dim_th;
        // col index is k mod dim_th
        int j = k % dim_th;
		//printf("k = %d, i = %d, j = %d\n", k, i, j);

        // diagonal elements
        if(i == j){

        	// compute f(theta+eps_i)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
            	Vect interpret_theta_forw_i = interpret_theta+epsG.col(i);
            	Vect theta_forw_i(dim_th);
				convert_interpret2theta(interpret_theta_forw_i, theta_forw_i);

				f_i_i_loc(0,i) = eval_post_theta(theta_forw_i, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
				Vect interpret_theta_back_i = interpret_theta-epsG.col(i);
            	Vect theta_back_i(dim_th);
				convert_interpret2theta(interpret_theta_back_i, theta_back_i);

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
            	Vect theta_forw_i_j(dim_th);
				convert_interpret2theta(interpret_theta_forw_i_j, theta_forw_i_j);

				f_i_j_loc(0,k) 				   = eval_post_theta(theta_forw_i_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta+eps_i-eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect interpret_theta_forw_i_back_j = interpret_theta+epsG.col(i)-epsG.col(j);
            	Vect theta_forw_i_back_j(dim_th);
				convert_interpret2theta(interpret_theta_forw_i_back_j, theta_forw_i_back_j);

				/*
				theta_forw_i_back_j[0] = interpret_theta_forw_i_back_j[0];
				if(ns > 0 && nt > 1){
					convert_interpret2theta_spatTemp(interpret_theta_forw_i_back_j[1], interpret_theta_forw_i_back_j[2], interpret_theta_forw_i_back_j[3], theta_forw_i_back_j[1], theta_forw_i_back_j[2], theta_forw_i_back_j[3]);
					if(nss > 0){
						convert_interpret2theta_spat(interpret_theta_forw_i_back_j[4], interpret_theta_forw_i_back_j[5], theta_forw_i_back_j[4], theta_forw_i_back_j[5]);
					}
				} else if(ns > 0 && nt == 1 && dim_th == 3){
					convert_interpret2theta_spat(interpret_theta_forw_i_back_j[1], interpret_theta_forw_i_back_j[2], theta_forw_i_back_j[1], theta_forw_i_back_j[2]);
				} else if(ns > 0 && nt == 1 && dim_th == 2){
					convert_interpret2theta_spat(interpret_theta_forw_i_back_j[0], interpret_theta_forw_i_back_j[1], theta_forw_i_back_j[0], theta_forw_i_back_j[1]);
				}
				*/
				f_i_j_loc(1,k)                 = eval_post_theta(theta_forw_i_back_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i+eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
	            Vect interpret_theta_back_i_forw_j = interpret_theta-epsG.col(i)+epsG.col(j);
            	Vect theta_back_i_forw_j(dim_th);
				convert_interpret2theta(interpret_theta_back_i_forw_j, theta_back_i_forw_j);

				f_i_j_loc(2,k)                 = eval_post_theta(theta_back_i_forw_j, mu_tmp); 
            }
            counter++;

        	// compute f(theta-eps_i-eps_j)
            if(MPI_rank == task_to_rank_list[counter]){ 
	            Vect mu_tmp(n);
            	Vect interpret_theta_back_i_j 	   = interpret_theta-epsG.col(i)-epsG.col(j);
            	Vect theta_back_i_j(dim_th);
				convert_interpret2theta(interpret_theta_back_i_j, theta_back_i_j);

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

#endif // end #if at line 978

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
	double val_prior_lat;

	// eval_likelihood: log_det, -theta*yTy
	double log_det_l;
	double val_l; 

	// value : pi(x | theta, y) or constraint problem pi(x | theta, y, Ax = e)
	double val_d;

	#pragma omp parallel 
	//#pragma omp single
	{

	// =============== evaluate NOMINATOR ================= //
		
#ifdef RECORD_TIMES
	t_thread_nom = -omp_get_wtime();
#endif

	// =============== evaluate theta prior based on original solution & variance = 1 ================= //
	if(omp_get_thread_num() == 0) // instead of #pragma omp task -> want always the same thread to do same task
	{

#ifdef PRINT_MSG
		std::cout << "prior : " << prior << std::endl;
#endif

#ifdef RECORD_TIMES
	t_priorHyp = -omp_get_wtime();
#endif

	if(prior == "gaussian"){   // || dim_th != 4
		// evaluate gaussian prior
		Vect theta_interpret(dim_th);
		convert_theta2interpret(theta, theta_interpret);
		eval_log_gaussian_prior_hp(theta_interpret, theta_prior_param, log_prior_sum);

	} else if(prior == "pc"){
		// pc prior
		Vect theta_interpret(dim_th); 
		convert_theta2interpret(theta, theta_interpret);

		/*
		theta_interpret[0] = theta[0];
		convert_theta2interpret_spatTemp(theta[1], theta[2], theta[3], theta_interpret[1], theta_interpret[2], theta_interpret[3]);
		//theta_interpret << 0.5, 10, 1, 4; 
		if(nss > 0){
			convert_theta2interpret_spat(theta[4], theta[5], theta_interpret[4], theta_interpret[5]);
		}
		*/

		//Vect lambda(4);
		//lambda << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0; // lambda0 & lambda3 equal
		// assumed order theta_prior_param: prec obs, range s for st, range t for st, prec sigma for st, range s for s, prec sigma for s
		eval_log_pc_prior_hp(log_prior_sum, theta_prior_param, theta_interpret);
	} else {
		std::cout << "Prior not appropriately defined." << std::endl;
		exit(1);
	}

#ifdef RECORD_TIMES
	t_priorHyp += omp_get_wtime();
#endif

#ifdef PRINT_MSG
		std::cout << "log prior hyperparm sum : " << log_prior_sum << std::endl;
#endif

	// =============== evaluate prior of random effects : need log determinant ================= //

	// requires factorisation of Q.u -> can this be done in parallel with the 
	// factorisation of the denominator? 
	// How long does the assembly of Qu take? Should this be passed on to the 
	// denominator to be reused?

#ifdef RECORD_TIMES
	t_priorLat = -omp_get_wtime();
#endif

	if(ns > 0){
		eval_log_prior_lat(theta, val_prior_lat);
	}

#ifdef RECORD_TIMES
	t_priorLat += omp_get_wtime();
#endif

#ifdef PRINT_MSG
		std::cout << "val prior lat : "  << val_prior_lat << std::endl;
#endif

	// =============== evaluate likelihood ================= //

#ifdef RECORD_TIMES
	t_likel = -omp_get_wtime();
#endif

	eval_likelihood(theta, log_det_l, val_l);

#ifdef RECORD_TIMES
	t_likel += omp_get_wtime();
#endif

#ifdef PRINT_MSG
		std::cout << "log det likelihood : "  << log_det_l << std::endl;
		std::cout << "val likelihood     : " << val_l << std::endl;
#endif

#ifdef RECORD_TIMES
	t_thread_nom += omp_get_wtime();
#endif

	} // end pragma omp task, evaluating nominator

	if(omp_get_thread_num() == 1 || threads_level1 == 1)//#pragma omp task
	{

#ifdef RECORD_TIMES
	t_thread_denom = -omp_get_wtime();
#endif
	// pin Q here
	


	// =============== evaluate denominator ================= //
	// denominator :
	// log_det(Q.x|y), mu, t(mu)*Q.x|y*mu
	SpMat Q(n, n);
	Vect rhs(n);

#ifdef RECORD_TIMES
	t_condLat = -omp_get_wtime();
#endif

 	eval_denominator(theta, val_d, Q, rhs, mu);

#ifdef RECORD_TIMES
	t_condLat += omp_get_wtime();
#endif

#ifdef PRINT_MSG
		std::cout << "val d     : " <<  val_d << std::endl;
#endif

#ifdef RECORD_TIMES
	t_thread_denom += omp_get_wtime();
#endif

	}

    //#pragma omp taskwait -> implicit barrier at the end of parallel region

	} // closing omp parallel region

	// =============== add everything together ================= //
	//std::cout << " val d = " << val_d << std::endl;
  	double val = -1 * (log_prior_sum + val_prior_lat + log_det_l + val_l - val_d);


#ifdef PRINT_MSG
  	std::cout << MPI_rank << " " << std::setprecision(6) << theta.transpose();
  	std::cout << " " << std::fixed << std::setprecision(12);
  	std::cout << log_prior_sum << " ";
  	std::cout << val_prior_lat << " " << log_det_l << " " << val_l << " " << val_d << " " << val << std::endl;


    std::cout << "sum nominator : " << log_prior_sum + val_prior_lat + log_det_l + val_l  << ", sum denominator : " << val_d << ", f theta : " << val << std::endl;
#endif

  	return val;
}


void PostTheta::eval_log_gaussian_prior_hp(Vect& theta_interpret, Vect& theta_prior_param, double& log_prior){

	log_prior = -0.5 * (theta_interpret - theta_prior_param).transpose() * (theta_interpret - theta_prior_param);

#ifdef PRINT_MSG
		std::cout << "theta param : " << theta_interpret.transpose() << ", log prior sum : " << log_prior << std::endl;
#endif
}

// NEW ORDER sigma.e, range s, range t, sigma.u + range s spat, sigma u spat
void PostTheta::eval_log_pc_prior_hp(double& log_sum, Vect& lambda, Vect& interpret_theta){

  double prior_se = log(lambda[0]) - lambda[0] * exp(interpret_theta[0]) + interpret_theta[0];
  //printf("prior se = %f\n", prior_se);
  double prior_su = log(lambda[3]) - lambda[3] * exp(interpret_theta[3]) + interpret_theta[3];
  //printf("sigma : %f, prior su = %f\n", interpret_theta[3], prior_su);

  double prior_rs = log(lambda[1]) - lambda[1] * exp(-interpret_theta[1]) - interpret_theta[1];
  //printf("range s: %f, prior rs = %f\n", interpret_theta[1], prior_rs);
  
  double prior_rt = log(lambda[2]) - lambda[2] * exp(-0.5*interpret_theta[2]) + log(0.5) - 0.5*interpret_theta[2];
  //printf("range t: %f, prior rt = %f\n", interpret_theta[2], prior_rt);

  log_sum = prior_rt + prior_rs + prior_su + prior_se;

  if(interpret_theta.size() > 4){
	    double dHalf = dim_spatial_domain / 2.0;
		// prior range s for add. spatial field
		log_sum += log(dHalf * lambda[4]) - lambda[4] * exp(- dHalf * interpret_theta[4]) - dHalf * interpret_theta[4];
		//log_sum += log(lambda[4]) - 2*interpret_theta[4] - lambda[4]*exp(-interpret_theta[4]);
		//printf("prior range s for add. s: %f",log(lambda[4]) - 2*interpret_theta[4] - lambda[4]*exp(-interpret_theta[4]));

		// prior sigma u for add. spatial field
 		log_sum += log(lambda[5]) - lambda[5] * exp(interpret_theta[5]) + interpret_theta[5];
		//log_sum += log(lambda[5]) - lambda[5]*exp(interpret_theta[5]);
		//printf(", prior sigma u for add. s: %f", log(lambda[5]) - lambda[5]*exp(interpret_theta[5]));
  }

  		//std::cout << ", total log prior sum hyperparam " << log_sum << std::endl;

#ifdef PRINT_MSG
		std::cout << "log prior sum hyperparam " << log_sum << std::endl;
#endif

  if(isnan(log_sum) || isinf(log_sum)){
	printf("Log Sum = %f!\n", log_sum);
	std::cout << "lambda          = " << lambda.transpose() << std::endl;
	std::cout << "interpret theta = " << interpret_theta.transpose() << std::endl;
	exit(1);
  }
}


void PostTheta::update_mean_constr(const MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W, MatrixXd& U, Vect& updated_sol){

    // now that we have V = Q^-1*t(Dxy), compute W = Dxy*V
    W = D*V;
    //std::cout << "MPI rank : " << MPI_rank << ", norm(V) = " << V.norm() << ", W = " << W << std::endl;
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
    if(isnan(log_pi_Ax)){
    	std::cout << "Log p(Ax) is NaN. log(det(W)) : " << log(W.determinant()) << ", (Dx - Dmu)*inv(W)*(Dx - Dmu) : " << (D*x - D*mu).transpose()*W.inverse()*(D*x - D*mu) << std::endl;
    	std::cout << "norm(mu) = " << mu.norm() << ", norm(x) = " << x.norm() << ", W = " << W << ", inv(W) = " << W.inverse() << std::endl;
    }
    //std::cout << "W = " << W << std::endl;
    //std::cout << "log_pi_Ax = " << log_pi_Ax << ", log(W.determinant()) = " << log(W.determinant()) << "0.5*(D*x - D*mu).transpose()*W.inverse()*(D*x - D*mu) = " << 0.5*(D*x - D*mu).transpose()*W.inverse()*(D*x - D*mu) << std::endl;

    val_log_dens = log_pi_x + log_pi_Ax_x - log_pi_Ax;
    //std::cout << - 0.5*rowsQ*log(2*M_PI) - (- 0.5*D.rows()*log(2*M_PI)) << " " << - 0.5*(rowsQ-D.rows())*log(2*M_PI) << std::endl;
    //std::cout << "log val Bayes cond = " << val_log_dens << std::endl;  

}


void PostTheta::eval_log_prior_lat(Vect& theta, double &val){

	double log_det;

	double time_construct_Qst = -omp_get_wtime();
	SpMat Qu(nu, nu);

#ifdef RECORD_TIMES
	t_priorLatAMat = -omp_get_wtime();
#endif

	if(nt > 1 && nss == 0){
		construct_Q_spat_temp(theta, Qu);
	} else if(nt > 1 && nss > 0){
		SpMat Qst_prior(nst, nst);
		construct_Q_spat_temp(theta, Qst_prior);

		// insert entries of Qst
		for (int k=0; k<Qst_prior.outerSize(); ++k){
			for (SparseMatrix<double>::InnerIterator it(Qst_prior,k); it; ++it)
			{
			Qu.insert(it.row(),it.col()) = it.value();                 
			}
		}

		// TODO: improve. need to be careful about what theta values are accessed!! now dimension larger
		SpMat Qs_prior(nss, nss);
		Vect theta_sub = theta(seq(3,5));
		construct_Q_spatial(theta_sub, Qs_prior);
		nnz_Qs = Qs.nonZeros();

		// insert entries of Qs
		for (int k=0; k<Qs_prior.outerSize(); ++k){
			for (SparseMatrix<double>::InnerIterator it(Qs_prior,k); it; ++it)
			{
			Qu.insert(it.row()+nst,it.col()+nst) = it.value();                 
			}
		}
	} else if(nt == 1 && nss == 0) {
		construct_Q_spatial(theta, Qu);
	} else {
		printf("invalid parameter combination. In eval_log_prior_lat(). ns = %d, nt = %d, nss = %d\n", ns, nt, nss);
		exit(1);
	}

#ifdef RECORD_TIMES
	t_priorLatAMat += omp_get_wtime();
#endif

	time_construct_Qst += omp_get_wtime();
	double time_factorize_Qst = -omp_get_wtime();

	if(constr == true){
		//std::cout << "in eval log det Qu in constr true" << std::endl;
		MatrixXd V(nu, Dx.rows());
		//std::cout << "before factorize w constraint. theta = " << theta.transpose() << std::endl;
		solverQst->factorize_w_constr(Qu, Dx, log_det, V);

		Vect constr_mu(nu);
		Vect e = Vect::Zero(Dx.rows());
		MatrixXd U(Dx.rows(), nu);
		MatrixXd W(Dx.rows(), Dx.rows());
		Vect mu_tmp = Vect::Zero(nu);
		update_mean_constr(Dx, e, mu_tmp, V, W, U, constr_mu);
		//std::cout << "MPI rank : " << MPI_rank << ". In Log Prior Lat. W = " << W << std::endl;

		Vect unconstr_mu = mu_tmp;
		mu_tmp = constr_mu;
		//std::cout << "mu = " << mu_tmp.head(10).transpose() << std::endl;

		Vect x = Vect::Zero(nu);
		// multiplication with 0.5 is already included
		eval_log_dens_constr(x, unconstr_mu, Qu, log_det, Dx, W, val);

	} else{
		solverQst->factorize(Qu, log_det, t_priorLatChol);
		/*if(MPI_rank == 0){
			std::cout << "t_priorLatChol : " << t_priorLatChol << std::endl;
		}*/
		val = 0.5 * (log_det);

	}

	/*
	std::string Qprior_fileName = "Q_prior_InEvalPriorLat.txt";
	SpMat A_lower = Qu.triangularView<Lower>();
	std::cout << "theta in eval log prior lat : " << theta.transpose() << std::endl;

	int n = A_lower.cols();
	int nnz = A_lower.nonZeros();

	ofstream sol_file(Qprior_fileName);
	sol_file << n << "\n";
	sol_file << n << "\n";
	sol_file << nnz << "\n";

	for (int i = 0; i < nnz; i++){
		sol_file << A_lower.innerIndexPtr()[i] << "\n";
	}   
	for (int i = 0; i < n+1; i++){
			sol_file << A_lower.outerIndexPtr()[i] << "\n";
	}     
	for (int i = 0; i < nnz; i++){
		sol_file << std::setprecision(15) << A_lower.valuePtr()[i] << "\n";
	}

	sol_file.close();
	std::cout << "wrote to file : " << Qprior_fileName << std::endl;

	exit(1);
	*/

	time_factorize_Qst += omp_get_wtime();

#ifdef PRINT_MSG
	std::cout << "val log prior lat " << val << std::endl;
#endif


#ifdef PRINT_TIMES
	if(MPI_rank ==0){
		std::cout << "time construct Qst prior = " << time_construct_Qst << std::endl;
		std::cout << "time factorize Qst prior = " << time_factorize_Qst << std::endl;
	}
#endif

}

// ONLY WORKS FOR SUM-TO-ZERO CONSTRAINTS
// 0.5*theta[0]*t(y - B*b - A*u)*(y - B*b - A*u) => normally assume x = u,b = 0
// constraint case -> maybe cannot evaluate in zero i.e. when e != 0, 
// might make more sense to evaluate x = mu_constraint, from Dxy*mu_constraint = e
void PostTheta::eval_likelihood(Vect& theta, double &log_det, double &val){
	
	// multiply log det by 0.5
	double theta0 = theta[0];

	if(validate){
		log_det = 0.5 * w_sum*theta0;  // some are zero ...
	} else {
		log_det = 0.5 * no*theta0;
	}
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

// assumed order parameters: theta = prec_obs, lgamE, lgamS
void PostTheta::construct_Q_spatial(Vect& theta, SpMat& Qs){

	// Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
	// return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
	double exp_theta1; 
	double exp_theta2;
	
	if(dim_th == 2){
		exp_theta1 = exp(theta[0]);
		exp_theta2 = exp(theta[1]);
	} else if(dim_th == 3) {
		exp_theta1 = exp(theta[1]);
		exp_theta2 = exp(theta[2]);	
	}

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

	//std::cout << "in construct_Q_spat_temp(). theta : " << theta.transpose() << std::endl;

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
	if(MPI_rank == 0){
		std::cout << "theta u : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;

		std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
		std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;

		std::cout << "q1s : \n" << q1s.block(0,0,10,10) << std::endl;
	    std::cout << "q2s : \n" << q2s.block(0,0,10,10) << std::endl;
	    std::cout << "q3s : \n" << q3s.block(0,0,10,10) << std::endl;

	    std::cout << "M0  : \n" << M0.block(0,0,10,10) << std::endl;
	    std::cout << "M1  : \n" << M1.block(0,0,10,10) << std::endl;
	    std::cout << "M2  : \n" << M2.block(0,0,10,10) << std::endl;
	   }
#endif

	// assemble overall precision matrix Q.st
	Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));
	/*if(MPI_rank == 0){
		std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
	}*/

	/*
	// check for NaN values in matrix
	SpMat Qst_lower = Qst.triangularView<Lower>(); 
	int nnz = Qst_lower.nonZeros();

	for(int i= 0; i<nnz; i++){
		if(isnan(Qst_lower.valuePtr()[i])){
			std::cout << "In construct_Qst! Found NaN value in Qst. Qst[" << i << "] = " << Qst_lower.valuePtr()[i] << std::endl;
			std::cout << "exp(theta_u) = " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;
		}
	}
	*/

	////////////////////////////////////////////////////////////// 
	// here to stabilize the model ... theoretically shouldn't be here ...
	// is in INLA
	if(constr){
		SpMat epsId(nu,nu);
		epsId.setIdentity();
		epsId = 1e-4*epsId;

		Qst = Qst + epsId;
	}
	////////////////////////////////////////////////////////////// 

#ifdef PRINT_MSG
		//std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
#endif
}

/*
void PostTheta::update_mean_constr(MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W){

    std::cout << "in update mean constr. " << std::endl;
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
*/

void PostTheta::construct_Qprior(Vect& theta, SpMat& Qx){
	
	if(dimList(seq(1,2)).sum() == 0){
		Qx = Qb;
	} else {
		if(dimList(1) == 3){
			construct_Q_spat_temp(theta, Qu);
		} else if(dimList(2) == 2) {	
			construct_Q_spatial(theta, Qu);
		} else {
			std::cout << "in construct_Qprior. nvalid dimList: " << dimList.transpose() << std::endl;
		}	

		// ovewrite value ptr of Qst part -> Q_fe stays the same 
		// ATTENTION: needs to be adapted when additional spatial field is there. 
		//t_Qcomp = - omp_get_wtime();
	    for(int i=0; i<Qu.nonZeros(); i++){
            Qx.valuePtr()[i] = Qu.valuePtr()[i];
            //Qx_new.valuePtr()[i] = 0.0;
        }

		if(nss > 0){
			// TODO: improve. need to be careful about what theta values are accessed!! now dimension larger
			Vect theta_sub = theta(seq(3,5));
			construct_Q_spatial(theta_sub, Qs);
			nnz_Qs = Qs.nonZeros();

			// insert entries of Qs
			for (int k=0; k<Qs.outerSize(); ++k){
				for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
				{
				Qx.coeffRef(it.row()+nst,it.col()+nst) = it.value();  
				}
			}

			//instead go over valuePtr as before
			// Qx.valuePtr()[i]               
		}
	}

}

void PostTheta::construct_Q(Vect& theta, Vect& mu, SpMat& Q){
	
	double exp_theta0;
	if(dimList(0) > 0){
		exp_theta0 = exp(theta[0]);
	}

	if(ns > 0){
		//SpMat Qu(nu, nu);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(theta, Qu);
		} else {	
			construct_Q_spatial(theta, Qu);
		}	

		// ovewrite value ptr of Qst part -> Q_fe stays the same 
		// ATTENTION: needs to be adapted when additional spatial field is there. 
		//t_Qcomp = - omp_get_wtime();
	    for(int i=0; i<Qu.nonZeros(); i++){
            Qx.valuePtr()[i] = Qu.valuePtr()[i];
            //Qx_new.valuePtr()[i] = 0.0;
        }

		if(nss > 0){
			// TODO: improve. need to be careful about what theta values are accessed!! now dimension larger
			Vect theta_sub = theta(seq(3,5));
			construct_Q_spatial(theta_sub, Qs);
			nnz_Qs = Qs.nonZeros();

			// insert entries of Qs
			for (int k=0; k<Qs.outerSize(); ++k){
				for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
				{
				Qx.coeffRef(it.row()+nst,it.col()+nst) = it.value();  
				}
			}

			//instead go over valuePtr as before
			// Qx.valuePtr()[i]               
		}

		//t_Qcomp += omp_get_wtime();

		// old way to assemble matrix, inefficient
		/*
		double t_Qcomp = -omp_get_wtime();	
		// if the sparsity pattern of Qx is not set up yet
		SpMat Qx_comp(n,n);
		int nnz = Qu.nonZeros();
		Qx_comp.reserve(nnz);

		for (int k=0; k<Qu.outerSize(); ++k){
		  for (SparseMatrix<double>::InnerIterator it(Qu,k); it; ++it)
		  {
		    Qx_comp.insert(it.row(),it.col()) = it.value();                 
		  }
		}

		for(int i=nu; i < n; i++){
			// CAREFUL 1e-3 is arbitrary choice!!
			Qx_comp.coeffRef(i,i) = 1e-3;
		}

		t_Qcomp += omp_get_wtime();
		printf("old way: %f\n", t_Qcomp);
		
		printf("new way: %f\n", t_Qcomp);
		printf("norm(Qx - Qx_comp) = %f\n", (Qx - Qx_comp).norm());
		*/


#ifdef PRINT_MSG
		//std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
		//std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;
#endif
		// can also be optimized but not as straight forward
		// would need zero-padding in Qx
		Q =  Qx + exp_theta0 * AxTAx;

#ifdef PRINT_MSG
		/*
		std::cout << "exp(theta0) : \n" << exp_theta0 << std::endl;
		std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;

		std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
		std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
		std::cout << "theta : \n" << theta.transpose() << std::endl;
		*/
#endif
	}

	if(ns == 0){

		if(likelihood.compare("gaussian") == 0){
			//Qb = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
			/*std::cout << "Q_b " << std::endl;
			std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/
			
			// Q.e <- Diagonal(no, exp(theta))
			// Q.xy <- Q.x + crossprod(A.x, Q.e)%*%A.x  # crossprod = t(A)*Q.e (faster)	
			Q = Qb + exp_theta0*B.transpose()*B;	
		
		} else {
			//printf("in non-gaussian regression statement.\n");
			SpRmMat hess_eta(no,no);
			hess_eta.setIdentity();
			// TODO: make x input parameter
			if(n != 4){
				printf("in construct_Q(). Properly implement it for non-Gaussian data!\n");
				exit(1);
			}

			//Vect x(n); 
			//x << -0.644116603734506,  0.473138588559389,  0.653101074984064, -1.813681904468703;
			
			// compute Gaussian at the mode
			//std::cout << "mu: " << mu.transpose() << std::endl;
			Vect eta = B * mu;
			Vect diag_hess_eta(no);
			FD_diag_hessian(eta, diag_hess_eta);
			hess_eta.diagonal() = diag_hess_eta;
			// hessian of negative log conditional (minimization)
			Q = Qb + B.transpose() * hess_eta * B;
			std::cout << "Q:\n" << MatrixXd(Q) << std::endl;
			//std::cout << "inv(Q):\n" << MatrixXd(Q).inverse() << std::endl;
		}

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


void PostTheta::eval_denominator(Vect& theta, double& val, SpMat& Q, Vect& rhs, Vect& mu){

	double log_det;

// ============================================================================================== //
// FOR NOW SEPARATE INTO GAUSSIAN -- NON-GAUSSIAN
// ============================================================================================== //

	if(likelihood.compare("gaussian") == 0){
		double time_construct_Q = -omp_get_wtime();

#ifdef RECORD_TIMES
		t_condLatAMat = -omp_get_wtime();
#endif
		// construct Q_x|y,
		construct_Q(theta, mu, Q);
#ifdef RECORD_TIMES
		t_condLatAMat += omp_get_wtime();
#endif

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
			Vect e = Vect::Zero(Dxy.rows());
			MatrixXd U(Dxy.rows(), mu.size());
			MatrixXd W(Dxy.rows(), Dxy.rows());
			update_mean_constr(Dxy, e, mu, V, W, U, constr_mu);
			Vect unconstr_mu = mu;
			mu = constr_mu;

			Vect x = Vect::Zero(mu.size());
			// log det is already in val
			eval_log_dens_constr(x, unconstr_mu, Q, log_det, Dxy, W, val);

		} else {
			// solve linear system
			// returns vector mu, which is of the same size as rhs
			//solve_cholmod(Q, rhs, mu, log_det);
			solverQ->factorize_solve(Q, rhs, mu, log_det, t_condLatChol, t_condLatSolve);
		
			// compute value
			val = 0.5*log_det - 0.5 * mu.transpose()*(Q)*(mu);
		}

		time_solve_Q += omp_get_wtime();

#ifdef PRINT_MSG
		std::cout << "log det d : " << log_det << std::endl;
#endif

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

	} else if(likelihood.compare("poisson") == 0 || likelihood.compare("binomial") == 0){
		printf("dummy not working!");
		exit(1);

		//NewtonIter(SpMat& Qprior, Vect& x)


	} else {
		printf("invalid likelihhood!");
		exit(1);
	}

}

// ============================================================================================= //
// ==================================== INNER ITERATION ======================================== //
// ============================================================================================= //
// TODO: replace B by Ax -> careful rowmajor -> multiplication problem ... must be make column-major


/////////////////// Prior /////////////////
double PostTheta::cond_LogPriorLat(SpMat& Qprior, Vect& x){
    Vect mean = Vect::Zero(x.size());

	// f_val = n/2*log(2*pi) + 0.5*|Q| - 0.5*t(x - mu) %*% Q (x - mu)
	// Q(theta) -> doesn't change with x -> ignore for now, maybe need later
    double f_val = -0.5 * (x - mean).transpose() * Qprior * (x - mean);
    return f_val;
}

/////////////////// Likelihoods /////////////////
// Poisson
double PostTheta::cond_LogPoisLik(Vect& eta){
    double f_val = eta.dot(y) - (extraCoeffVecLik.array()*(eta.array().exp())).sum();
    return f_val;
}

// TODO: include scaling constant E for each eta -> will also be required in input ...
// 
double PostTheta::cond_negLogPoisLik(Vect& eta){
    // actually link function fixed here but to make input the same ...
	//printf("in cond_negLogPoisLik. dim(y) = %ld, dim(extraCoeffVecLik) = %ld, dim(eta) = %ld\n", y.size(), extraCoeffVecLik.size(), eta.size());
	//std::cout << "eta.dot(y) = " << eta.dot(y) << ", sum(E*exp(eta)) = " << (extraCoeffVecLik.array()*(eta.array().exp())).sum() << std::endl;
    double f_val = eta.dot(y) - (extraCoeffVecLik.array()*(eta.array().exp())).sum();
    return -1*f_val;
}

Vect PostTheta::grad_cond_negLogPoisLik(Vect& eta){
	Vect grad = y.array() - extraCoeffVecLik.array()*(eta.array().exp());
	return -1*grad;
}

Vect PostTheta::diagHess_cond_negLogPoisLik(Vect& eta){
	Vect diagHess = - extraCoeffVecLik.array()*(eta.array().exp());
	return -1*diagHess;
}

double PostTheta::cond_negLogPois(SpMat& Qprior, Vect& x){

    double f_val_prior = cond_LogPriorLat(Qprior, x);

    Vect eta = Ax * x;
    double f_val_lik = cond_LogPoisLik(eta);

    // times -1: want to minimize
    double f_val = -1 * (f_val_prior + f_val_lik);
    return f_val;
}

// LINK FUNCTIONS
void PostTheta::link_f_sigmoid(Vect& x, Vect& sigmoidX){
    //  1/(1 + e^-x)
    sigmoidX = 1.0 /(1.0 + (-1*x).array().exp());
}

// Binomial
double PostTheta::cond_negLogBinomLik(Vect& eta){
    int m = eta.size();

    Vect linkEta(m);
    // hardcode sigmoid for now
    link_f_sigmoid(eta, linkEta);    
    Vect logLinkEta = linkEta.array().log();
    Vect tmpLinkEta = (Vect::Ones(m) - linkEta).array().log();

    double f_val = y.dot(logLinkEta) + (extraCoeffVecLik - y).dot(tmpLinkEta);
    return -f_val;
}

double PostTheta::cond_negLogBinom(SpMat& Qprior, Vect& x){
    double f_val_prior = cond_LogPriorLat(Qprior, x);

    Vect eta = B * x;
    double f_val_neg_lik = cond_negLogBinomLik(eta);

    double f_val = -1 * f_val_prior + f_val_neg_lik;

    return f_val;
}

// general formulation for evaluating log conditional distribution
// pass likelihood & link function as an argument
// later fix these things in class constructor ...
double PostTheta::cond_negLogDist(SpMat &Qprior, Vect& x, function<double(Vect&, Vect&)> lik_func){
	printf("DUMMY VERSION. NOT WORKING YET!\n");
    double f_val = 0;
    return f_val;
}


// naive FIRST ORDER CENTRAL DIFFERENCE (to be improved ...?)
// will simplify once we have this inside class ... only beta will be a variable ...
// expects as input
void PostTheta::FD_gradient(Vect& eta, Vect& grad){
    int m = eta.size();
    double h = 1e-5;

	if(eta.size() != grad.size()){
		printf("dim(eta): %ld & dim(grad): %ld don't match!\n", eta.size(), grad.size()); 
	}

    // probably better way to do this ...
    SpMat epsId(m,m);
    epsId.setIdentity();
    epsId = h * epsId;

    for(int i=0; i<m; i++){
        Vect eta_forward      = eta + epsId.col(i);
		Vect eta_backward     = eta - epsId.col(i);

		double f_eta_forward;
		double f_eta_backward;

		if(likelihood.compare("poisson") == 0){
        	f_eta_forward  = cond_negLogPoisLik(eta_forward);
			f_eta_backward = cond_negLogPoisLik(eta_backward);

		} else if(likelihood.compare("binomial") == 0){
        	f_eta_forward  = cond_negLogBinomLik(eta_forward);
			f_eta_backward = cond_negLogBinomLik(eta_backward);
		} else {
			printf("invalid likelihood!\n");
			exit(1);
		}

		if(i == 0){
			std::cout << "f(eta_forward)  = " << std::fixed << std::setprecision(15) << f_eta_forward << std::endl;
			std::cout << "f(eta_backward) = " << std::fixed << std::setprecision(15) << f_eta_backward << std::endl;
		}

        grad(i) = (f_eta_forward - f_eta_backward) / (2*h);
    }
}

// naive SECOND ORDER DIFFERENCE: DIAGONAL of Hessian
// expects as input cond_LogPois (generalize ... )
void PostTheta::FD_diag_hessian(Vect& eta, Vect& diag_hess){
    int m = eta.size();
    double h = 1e-5;

    // probably better way to do this ...
    SpMat epsId(m,m);
    epsId.setIdentity();
    epsId = h * epsId;

    double f_eta;
	if(likelihood.compare("poisson") == 0){
		f_eta  = cond_negLogPoisLik(eta);
	} else if(likelihood.compare("binomial") == 0){
		f_eta = cond_negLogBinomLik(eta);
	} else {
		printf("invalid likelihood!\n");
		exit(1);
	}
	
	//printf("f(eta) = %f\n", f_eta);
	//std::cout << "f(eta) = " << std::fixed << std::setprecision(15) << f_eta << std::endl;

    for(int i=0; i<m; i++){
        Vect eta_forward    = eta + epsId.col(i);
        Vect eta_backward   = eta - epsId.col(i);

		double f_eta_forward;
		double f_eta_backward;

		if(likelihood.compare("poisson") == 0){
        	f_eta_forward  = cond_negLogPoisLik(eta_forward);
			f_eta_backward = cond_negLogPoisLik(eta_backward);

		} else if(likelihood.compare("binomial") == 0){
        	f_eta_forward  = cond_negLogBinomLik(eta_forward);
			f_eta_backward = cond_negLogBinomLik(eta_backward);
		}
		/*if(i == 0){
			std::cout << "f(eta_forward)  = " << std::fixed << std::setprecision(15) << f_eta_forward << std::endl;
			std::cout << "f(eta_backward) = " << std::fixed << std::setprecision(15) << f_eta_backward << std::endl;
		}*/
        diag_hess(i) = (f_eta_forward - 2*f_eta + f_eta_backward) / (h*h);
    }

}


// inner Iteration: form Gaussian approximation to conditional
void PostTheta::NewtonIter(SpMat& Qprior, Vect& x){

    int n = x.size();
    int m = y.size();

    // prepare for iteration
    Vect x_new = x;
    Vect x_old = Vect::Random(n);

    Vect eta(m);
    Vect gradLik(m);
    Vect diag_hess_eta(m);

    SpRmMat hess_eta(m,m);
    hess_eta.setIdentity();

    Vect x_update(n);

	Vect negFoD(n);
	SpMat hess(n,n);

    //Vect FoD(n);
    SpMat SoD(n,n);

    // Eigen solver for now
    //SimplicialLLT<SpMat> solverNewton;

    // iteration
    int counter = 0;
    while((x_new - x_old).norm() > 1e-3){
        x_old = x_new;
        counter += 1;
		printf("\ncounter = %d\n", counter);

        if(counter > 50){
            printf("max number of iterations reached in inner Iteration!\n");
            exit(1);
        }

		if(dimList(seq(1,2)).sum() == 0){
       		eta = B * x_new;
		} else {
			eta = Ax * x_new;
		}

        // compute gradient
        //FD_gradient(eta, gradLik);
		gradLik = grad_cond_negLogPoisLik(eta);

		// compute hessian
        //std::cout << "x: " << x_new.head(min(10, (int) n)).transpose() << std::endl;
        //FD_diag_hessian(eta, diag_hess_eta);
		diag_hess_eta = diagHess_cond_negLogPoisLik(eta);
        hess_eta.diagonal() = diag_hess_eta;
		//std::cout << "diagHessEta = " << diag_hess_eta.head(min(10, (int) n)).transpose() << std::endl;

		if(dimList(seq(1,2)).sum() == 0){
        	//FoD = Qprior * x_new + B.transpose() * gradLik;
        	negFoD = -1* (Qprior * x_new + B.transpose() * gradLik);

			// hessian of negative log conditional (minimization)
       		hess = Qprior + B.transpose() * hess_eta * B;
		} else {
        	//FoD = Qprior * x_new + Ax.transpose() * gradLik;
        	negFoD = -1* (Qprior * x_new + Ax.transpose() * gradLik);	

			Qxy = Qprior + Ax.transpose() * hess_eta * Ax;
			//std::cout << "hess(1:10,1:10) = \n" << hess.block(0,0,10,10) << std::endl;
			//std::cout << "hess(-10:end, -10:end) = \n" << hess.block(n-10, n-10, 10,10) << std::endl;
		}

        /*
		solverNewton.compute(hess);

        if(solverNewton.info()!=Success) {
            cout << "Oh: Very bad. Hessian not pos. definite." << endl;
            exit(1);
        }
        // Newton step hess(x_k)*(x_k+1 - x_k) = - grad(x_k)
        // x_update = x_new - x_old
       	x_update = solverNewton.solve(-FoD);
		*/

		double t_condLatChol, t_condLatSolver;
		double log_det;
		//Vect x_update_new(x_update.size());
		solverQ->factorize_solve(hess, negFoD, x_update, log_det, t_condLatChol, t_condLatSolve);
		//std::cout << "norm(x_update - x_update_new) = " << (x_update - x_update_new).norm() << std::endl;

        x_new    = x_update + x_old;

    }

    x = x_new;
	std::cout << "converged after " << counter << " iterations." << std::endl;


}

// record times within one iteration (over multiple iterations)
void PostTheta::record_times(std::string file_name, int iter_count, double t_Ftheta_ext, double t_thread_nom, double t_priorHyp, 
								double t_priorLat, double t_priorLatAMat, double t_priorLatChol, double t_likel, 
								double t_thread_denom, double t_condLat, double t_condLatAMat, double t_condLatChol, double t_condLatSolve){

	std::cout << "in record times function" << std::endl;
    std::ofstream log_file(file_name, std::ios_base::app | std::ios_base::out);
    log_file << MPI_rank       << " " << threads_level1 << " " << threads_level2 << " " << iter_count     << " ";
    log_file << t_Ftheta_ext   << " " << t_thread_nom   << " " << t_priorHyp     << " " << t_priorLat     << " " << t_priorLatAMat << " " << t_priorLatChol << " " << t_likel << " ";
    log_file << t_thread_denom << " " << t_condLat    << " " << t_condLatAMat  << " " << t_condLatChol  << " " << t_condLatSolve << " " << std::endl;

	log_file.close(); 
}



PostTheta::~PostTheta(){

		delete solverQst;
		delete solverQ;		

		//delete theta_prior_test;
		//delete theta_test;
}


// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //

// for Hessian approximation : 4-point stencil (2nd order)
// -> swap sign, invert, get covariance

// once converged call again : extract -> Q.xy -> selected inverse (diagonal), gives me variance wrt mode theta & data y


