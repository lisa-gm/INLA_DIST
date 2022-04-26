#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// choose one of the two
#define DATA_SYNTHETIC
//#define DATA_TEMPERATURE

// enable RGF solver or not
#define RGF

#ifdef RGF
#include "cuda_runtime_api.h" // to use cudaGetDeviceCount()
#endif

//#define WRITE_LOG

#include "mpi.h"

//#include <likwid.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include <armadillo>
#include <LBFGS.h>


#include "PostTheta.h"
#include "../read_write_functions.cpp"

using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;

using namespace LBFGSpp;

/* ===================================================================== */

int main(int argc, char* argv[])
{
    // start timer for overall runtime
    double t_total = -omp_get_wtime();

    /* ======================= SET UP MPI ============================= */
    // Unique rank is assigned to each process in a communicator
    int MPI_rank;

    // Total number of ranks
    int MPI_size;

    // Initializes the MPI execution environment
    MPI_Init(&argc, &argv);

    // Get this process' rank (process within a communicator)
    // MPI_COMM_WORLD is the default communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

    // Get the total number ranks in this communicator
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size); 

    int threads_level1 = omp_get_max_threads();
    int threads_level2;

    #pragma omp parallel
    {  
    threads_level2 = omp_get_max_threads();
    }

    // overwrite in case RGF is used
    int noGPUs;
   
    if(MPI_rank == 0){
        printf("total no MPI ranks  : %d\n", MPI_size);
        printf("OMP threads level 1 : %d\n", threads_level1);
        printf("OMP threads level 2 : %d\n", threads_level2);
#ifdef RGF
	cudaGetDeviceCount(&noGPUs);
	printf("available GPUs      : %d\n\n", noGPUs);
#else
	printf("RGF dummy version\n");
    noGPUs = 0;
#endif
    }  
    
    #if 0
    // #include "generate_regression_data.cpp"

    if(argc != 1 + 2){
        std::cout << "wrong number of input parameters. " << std::endl;
        exit(1);
    }
    std::cout << "generates random sample" << std::endl;

    int nb = atoi(argv[1]);
    int no = atoi(argv[2]);

    Eigen::MatrixXd B(no, nb);
    Vect b(nb);
    Vect y(no);

    double tau = 0.5;
    generate_ex_regression(nb, no, tau, &B, &b, &y); 
    
    // Initial guess
    Vect theta(1);
    theta[0] = 3;
    #endif

    if(argc != 1 + 6 && MPI_rank == 0){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        exit(1);
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){
        std::cout << "reading in example. " << std::endl;
    }
#endif

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    //size_t no = atoi(argv[4]);
    std::string no_s = argv[4];
    // to be filled later
    size_t no;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    //std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nb);

    std::string base_path = argv[5];    

    std::string solver_type = argv[6];

    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("RGF") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << std::endl;
        exit(1);
    }

    if(MPI_rank == 0){
        std::cout << "Solver : " << solver_type << std::endl;
    }

    if(MPI_rank == 0 && solver_type.compare("RGF") == 0){
        // required memory on CPU to store Cholesky factor
        double mem_gb = (2*(nt-1)*ns*ns + ns*ns + (ns*nt+nb)*nb) * sizeof(T) / pow(10.0,9.0);
        printf("Memory Usage of each Cholesky factor on CPU = %f GB\n\n", mem_gb);
    }

    /* ---------------- read in matrices ---------------- */

    // dimension hyperparamter vector
    int dim_th;

    // spatial component
    SpMat c0; 
    SpMat g1; 
    SpMat g2;

    // spatial-temporal parts
    SpMat g3;
    SpMat M0;
    SpMat M1;
    SpMat M2;

    // data component / fixed effects
    MatrixXd B;
    SpMat Ax; 
    Vect y;

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }

        B = read_matrix(B_file, no, nb);

        // std::cout << "y : \n"  << y << std::endl;    
        // std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

        if(MPI_rank == 0){
            std::cout << "spatial model." << std::endl;
        }

        dim_th = 3;

        // check spatial FEM matrices
        std::string c0_file       =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file       =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file       =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);

        // doesnt require no to be read, can read no from Ax
        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();

        if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        if(MPI_rank == 0){
            std::cout << "spatial-temporal model." << std::endl;
        }

        dim_th = 4;

        // files to construct Q.u depending on HYPERPARAMETERS theta
        std::string c0_file      =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file      =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file      =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);
        std::string g3_file      =  base_path + "/g3_" + ns_s + ".dat";
        file_exists(g3_file);

        std::string M0_file      =  base_path + "/M0_" + nt_s + ".dat";
        file_exists(M0_file);
        std::string M1_file      =  base_path + "/M1_" + nt_s + ".dat";
        file_exists(M1_file);
        std::string M2_file      =  base_path + "/M2_" + nt_s + ".dat";
        file_exists(M2_file);  

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);
        g3 = read_sym_CSC(g3_file);

        M0 = read_sym_CSC(M0_file);
        //arma::mat(M0).submat(0,0,nt-1,nt-1).print();
        M1 = read_sym_CSC(M1_file);
        //arma::mat(M1).submat(0,0,nt-1,nt-1).print();
        M2 = read_sym_CSC(M2_file);
        //arma::mat(M2).submat(0,0,nt-1,nt-1).print();

        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();

        if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }

    } else {
        if(MPI_rank == 0){
            std::cout << "invalid parameters : ns nt !!" << std::endl;
            exit(1);
        }    
    }

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);


    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);             // define initial guess in model parametrization
    Vect theta_param(dim_th);       // or in interpretable parametrization
    Vect theta_prior_param(dim_th);
    Vect theta_original(dim_th); theta_original.setZero();

    int n;
    std::string data_type;

    // initialise theta
    if(ns == 0 && nt == 0){
        n = nb;
        // Initial guess
        theta[0] = 3;

        if(MPI_rank == 0){
            std::cout << "initial theta : "  << theta.transpose() << std::endl; 
        }   

    } else if(ns > 0 && nt == 1){
        n = ns + nb;
        //theta << 1, -1, 1;
        //theta << 1, -2, 2;
        //theta_prior << 0, 0, 0;

        std::cout << "using Elias TOY DATASET" << std::endl;
        // from INLA : log prec Gauss obs, log(Range) for i, log(Stdev) for i     
        //theta_prior << 1.0087220,  -1.0536157, 0.6320466;
        theta_prior_param << 1, -2.3, 2.1;
        theta << theta_prior_param;

        std::cout << "initial theta : "  << theta.transpose() << std::endl;   

    } else {
        n = ns*nt + nb;

#ifdef DATA_SYNTHETIC
        data_type = "synthetic";

        // =========== synthetic data set =============== //
        if(MPI_rank == 0){ 
            std::cout << "using SYNTHETIC DATASET" << std::endl; 
        }     
        // sigma.e (noise observations), sigma.u, range s, range t
        //theta_original_param << 0.5, 4, 1, 10;
        // sigma.e (noise observations), gamma_E, gamma_s, gamma_t
        theta_original << 1.4, -5.9,  1,  3.7;  // here exact solution, here sigma.u = 4
        //theta_prior << 1.386294, -5.594859,  1.039721,  3.688879; // here sigma.u = 3
        //theta_prior << 1.386294, -5.594859, 1.039721,  3.688879; // here sigma.u = 3
        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        if(MPI_rank == 0){
            std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        }
        //theta << 1.4, -5.9,  1,  3.7; 
        //theta << 1, -3, 1, 3;   // -> the one used so far !! maybe a bit too close ... 
        //theta_param << 4, 0, 0, 0;
        theta_param << 4,4,4,4;
        /*theta << 2, -3, 1.5, 5;
        if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/

#elif defined(DATA_TEMPERATURE)

        // =========== temperature data set =============== //
        data_type = "temperature";

        if(MPI_rank == 0){
            std::cout << "using TEMPERATURE DATASET" << std::endl; 
        }
        //theta << 4, 4, 4, 4;    // -> converges to wrong solution
        //theta_param << 4, 0, 0, 0;
        //theta_param << -5.967, 0.234, 0.547, 0.547;
        theta_param << -1.545, 2.358, 4.960, 4.940;
        //theta_param << -1.045, 8.917, 8.868, 3.541;
        // theta solution : -0.962555  6.309191 -8.195620 -7.203450
        //theta << 1, 8, -5, -5;   // -> works!
        //theta << 2, 8, -4, -4; // -> works!
        //theta << 2, 8, -2, -2; // doesn't work!

        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        //theta << -0.2, -2, -2, 3;
        /*if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/

#else 
        std::cerr << "\nUnknown datatype! Choose synthetic or temperature dataset!" << std::endl;
        exit(1);

#endif

    }

    Vect b(nb);

    // ============================ set up BFGS solver ======================== //

    // Set up parameters
    LBFGSParam<double> param;    
    // set convergence criteria
    // stop if norm of gradient smaller than :
    // computed as ||𝑔|| < 𝜖 ⋅ max(1,||𝑥||)
    param.epsilon = 1e-1;
    // or if objective function has not decreased by more than  
    // cant find epsilon_rel in documentation ...
    // stops if grad.norm() < eps_rel*x.norm() 
    param.epsilon_rel = 1e-3;
    // in the past ... steps
    param.past = 2;
    // TODO: stepsize too small? seems like it almost always accepts step first step.    // changed BFGS convergence criterion, now stopping when abs(f(x_k) - f(x_k-1)) < delta
    // is this sufficiently bullet proof?!
    //param.delta = 1e-2;
    param.delta = 1e-3;
    // maximum line search iterations
    param.max_iterations = 200;


    // Create solver and function object
    LBFGSSolver<double> solver(param);

    /*std::cout << "\nspatial grid size  : " << std::right << std::fixed << g1.rows() << " " << g1.cols() << std::endl;
    std::cout << "temporal grid size : " << M1.rows() << " " << M1.cols() << std::endl;
    std::cout << "Ax size            : " << Ax.rows() << " " << Ax.cols() << std::endl;*/

    // ============================ set up Posterior of theta ======================== //

    //std::optional<PostTheta> fun;
    PostTheta* fun;

    if(ns == 0){
        // fun.emplace(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, B, y, theta_prior_param, solver_type);
    } else if(ns > 0 && nt == 1) {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial constructor." << std::endl;
        }
        // PostTheta fun(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior_param, solver_type);
    } else {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial-temporal constructor." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type);
    }

//#ifdef DATA_TEMPERATURE
    theta[0] = theta_param[0];
    fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
    if(MPI_rank == 0){
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
    }
//#endif

    if(MPI_rank == 0){
        Vect theta_interpret_initial(dim_th);
        theta_interpret_initial[0] = theta[0];
        fun->convert_theta2interpret(theta[1], theta[2], theta[3], theta_interpret_initial[1], theta_interpret_initial[2], theta_interpret_initial[3]);
        std::cout << "initial theta interpret. param. : " << theta_interpret_initial.transpose() << std::endl;
    }


    #if 1
    double fx;

    //Vect grad_test(dim_th);
    //fx = fun(theta, grad_test);
    //std::cout <<  "f(x) = " << fx << std::endl;

    if(MPI_rank == 0){
        std::cout << "\nCall BFGS solver now. " << std::endl;
    }

    //LIKWID_MARKER_INIT;
    //LIKWID_MARKER_THREADINIT;

    double time_bfgs = -omp_get_wtime();
    int niter = solver.minimize(*fun, theta, fx, MPI_rank);

    //LIKWID_MARKER_CLOSE;

    time_bfgs += omp_get_wtime();

    // get number of function evaluations.
    int fn_calls = fun->get_fct_count();

    if(MPI_rank == 0){
        std::cout << niter << " iterations and " << fn_calls << " fn calls." << std::endl;
        //std::cout << "time BFGS solver             : " << time_bfgs << " sec" << std::endl;

        std::cout << "\nf(x)                         : " << fx << std::endl;
    }

    /*int fct_count = fun->get_fct_count();
    std::cout << "function counts thread zero  : " << fct_count << std::endl;*/

    Vect grad = fun->get_grad();
    if(MPI_rank == 0){
        std::cout << "grad                         :" << grad.transpose() << std::endl;
    }

    /*std::cout << "\nestimated mean theta         : " << theta.transpose() << std::endl;
    std::cout << "original theta               : " << theta_prior.transpose() << "\n" << std::endl;*/

    /*double eps = 0.005;
    Vect temp(4);
    temp << -5,2,3,-2;
    double f_temp = fun->f_eval(temp);
    std::cout << "f eval test : " << f_temp << endl;
    MatrixXd cov = fun->get_Covariance(temp, eps);
    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;*/

    if(MPI_rank == 0){
        std::cout << "\norig. mean parameters        : " << theta_original.transpose() << std::endl;
        std::cout << "est.  mean parameters        : " << theta.transpose() << std::endl;
    }

    // convert between different theta parametrisations
    if(dim_th == 4 && MPI_rank == 0){
        double prior_sigU; double prior_ranS; double prior_ranT;
        fun->convert_theta2interpret(theta_original[1], theta_original[2], theta_original[3], prior_ranT, prior_ranS, prior_sigU);
        std::cout << "\norig. mean interpret. param. : " << theta_original[0] << " " << prior_ranT << " " << prior_ranS << " " << prior_sigU << std::endl;

        double lgamE = theta[1]; double lgamS = theta[2]; double lgamT = theta[3];
        double sigU; double ranS; double ranT;
        fun->convert_theta2interpret(lgamE, lgamS, lgamT, ranT, ranS, sigU);
        std::cout << "est.  mean interpret. param. : " << theta[0] << " " << ranT << " " << ranS << " " << sigU << std::endl;
    }

    #endif

    #if 1

    Vect theta_max(dim_th);
    //theta_max << 2.675054, -2.970111, 1.537331;    // theta
    //theta_max = theta_prior;
    theta_max = theta;
    //theta_max << 1.391313, -5.913299,  1.076161,  3.642337;
    //theta_max << 1.331607, -5.893736,  1.001546,  3.743028;


    /*std::cout << "Estimated Covariance Matrix INLA : " << std::endl;
    MatrixXd Cov_INLA(4,4);

    Cov_INLA << 0.023833160, 0.01486733, 0.004853688, 0.005288554,
                0.014867325, 0.12749968, 0.056587582, 0.048833876,
                0.004853688, 0.05658758, 0.025517230, 0.022059932,
                0.005288554, 0.04883388, 0.022059932, 0.019274723;

    std::cout << Cov_INLA << std::endl;*/

    // in what parametrisation are INLA's results ... ?? 
    double eps = 0.005;
    MatrixXd cov(dim_th,dim_th);

    #if 0
    double t_get_covariance = -omp_get_wtime();

    eps = 0.005;
    //cov = fun->get_Covariance(theta_max, sqrt(eps));
    cov = fun->get_Covariance(theta_max, eps);

    t_get_covariance += omp_get_wtime();

    if(MPI_rank ==0){
        std::cout << "covariance                   : \n" << cov << std::endl;
        std::cout << "time get covariance          : " << t_get_covariance << " sec" << std::endl;
    }
    #endif


    #if 1
    //convert to interpretable parameters
    // order of variables : gaussian obs, range t, range s, sigma u
    Vect interpret_theta(4);
    interpret_theta[0] = theta_max[0];
    fun->convert_theta2interpret(theta_max[1], theta_max[2], theta_max[3], interpret_theta[1], interpret_theta[2], interpret_theta[3]);
   
#ifdef PRINT_MSG 
    if(MPI_rank == 0){
        std::cout << "est.  mean interpret. param. : " << interpret_theta[0] << " " << interpret_theta[1] << " " << interpret_theta[2] << " " << interpret_theta[3] << std::endl;
    }
#endif

    double t_get_covariance = -omp_get_wtime();
    cov = fun->get_Cov_interpret_param(interpret_theta, eps);
    t_get_covariance += omp_get_wtime();


    if(MPI_rank == 0){
        std::cout << "\ncovariance interpr. param.  : \n" << cov << std::endl;
        //std::cout << "time get covariance         : " << t_get_covariance << " sec" << std::endl;
    }
    #endif

    double t_get_fixed_eff;
    Vect mu(n);

    if(MPI_rank == 0){
        t_get_fixed_eff = - omp_get_wtime();
        
        fun->get_mu(theta, mu);

        t_get_fixed_eff += omp_get_wtime();
        std::cout << "\nestimated mean fixed effects : " << mu.tail(nb).transpose() << std::endl;
        //std::cout << "time get fixed effects       : " << t_get_fixed_eff << " sec\n" << std::endl;
    }

    #endif

  
    #if 1

    double t_get_marginals;
    Vect marg(n);

    //theta << 1.391313, -5913299,  1.076161,  3.642337;
    // when the range of u is large the variance of b0 is large.
    if(MPI_rank == 0){

        t_get_marginals = -omp_get_wtime();

        fun->get_marginals_f(theta, marg);

        t_get_marginals += omp_get_wtime();

        std::cout << "\nest. variances fixed eff.    :  " << marg.tail(nb).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  :  " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;
    }
    #endif
	
    #if 1
    t_total +=omp_get_wtime();
    if(MPI_rank == 0){
        // total number of post_theta_eval() calls 
        //std::cout << "\ntotal number fn calls        : " << fun->get_fct_count() << std::endl;
        std::cout << "\ntime BFGS solver             : " << time_bfgs << " sec" << std::endl;
        std::cout << "time get covariance          : " << t_get_covariance << " sec" << std::endl;
        std::cout << "time get marginals FE        : " << t_get_marginals << " sec" << std::endl;
        std::cout << "total time                   : " << t_total << std::endl;
    }
    #endif


    // ======================== write LOG file ===================== //
    // mpi_size, no_threads level 1, level 2, solver_type
    // ns, nt, nb, no, synthetic/real, theta_prior, theta_original (set all zero in real case), in both 
    // parametrisations?
    // theta_max, interpret_param_theta_max
    // mean fixed effects
    // var/std fixed effects
    // no of iterations BFGS
    // runtimes: BFGS, time per iteration, covariance (hessian), partial inversion, total

#ifdef WRITE_LOG
    if(MPI_rank == 0){

        // add time stamp to not overwrite?
        std::string log_file_name = base_path + "/log_"+ solver_type + "_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
        std::ofstream log_file(log_file_name);
        log_file << "mpi_size\t" << MPI_size << std::endl;
        log_file << "threads_l1\t" << threads_level1 << std::endl;
        log_file << "threads_l2\t" << threads_level2 << std::endl;
        log_file << "noGPUs\t" << noGPUs << std::endl;
        log_file << "solver_type\t" << solver_type << std::endl;
        log_file << "datatype\t" << data_type << std::endl;
        log_file << "ns\t" << ns << std::endl;
        log_file << "nt\t" << nt << std::endl;
        log_file << "num_fixed\t" << nb << std::endl;
        log_file << "num_obs\t" << no << std::endl;
        log_file << "theta_prior_param\t" << theta_prior_param.transpose() << std::endl;
        log_file << "theta_original\t" << theta_original.transpose() << std::endl;
        log_file << "theta_max\t" << theta_max.transpose() << std::endl;
        log_file << "interpret_param_theta_max\t" << interpret_theta.transpose() << std::endl;
	log_file << "mean_fixed_eff\t" << mu.tail(nb).transpose() << std::endl;
        log_file << "std_dev_fixed_eff\t" << marg.tail(nb).cwiseSqrt().transpose() << std::endl;          log_file << "time_bfgs\t" << time_bfgs << std::endl;
        log_file << "noIter\t" << niter << std::endl; 
        log_file << "time_Iter\t" << time_bfgs/niter << std::endl;
        log_file << "time_Cov\t" << t_get_covariance << std::endl;               
        log_file << "time_MargFE\t" << t_get_marginals <<  std::endl;
        log_file << "t_total\t" << t_total << std::endl;

  log_file.close(); 

    }

#endif


    delete fun;

    MPI_Finalize();
    return 0;
    
}
