#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// choose one of the two
//#define DATA_SYNTHETIC
#define DATA_TEMPERATURE

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
#include <Eigen/SparseCholesky>


#include <armadillo>
#include <LBFGS.h>

#include "PostTheta.h"
#include "../read_write_functions.cpp"

// comment out when not needed
#include "generate_regression_data.cpp"

using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;

using namespace LBFGSpp;


void construct_Q_spat_temp(Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3, SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Qst){

    double exp_theta1 = exp(theta[1]);
    double exp_theta2 = exp(theta[2]);
    double exp_theta3 = exp(theta[3]);

    // g^2 * fem$c0 + fem$g1
    SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

    // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
    SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

    // g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
    SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

    // assemble overall precision matrix Q.st
    Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + 2*exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

}

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
        printf("\n============== PARALLELISM & NUMERICAL SOLVERS ==============\n");
        printf("total no MPI ranks  : %d\n", MPI_size);
        printf("OMP threads level 1 : %d\n", threads_level1);
        printf("OMP threads level 2 : %d\n", threads_level2);
#ifdef RGF
	cudaGetDeviceCount(&noGPUs);
	printf("available GPUs      : %d\n", noGPUs);
#else
	printf("RGF dummy version\n");
    noGPUs = 0;
#endif
    }  
    
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
    size_t no = atoi(argv[4]);

    // to be filled later

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    size_t n = ns*nt + nb;


    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
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

    // we have two cholesky factors ...
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

    int num_constr = 1;
    //bool constr = false;
    bool constr = true;
    Vect e;
    MatrixXd Dx;
    MatrixXd Dxy;

    MatrixXd Prec;
    Vect b;
    Vect u;
    double tau;

    if(MPI_rank == 0)
        printf("\n==================== MODEL SPECIFICATIONS ===================\n");

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // #include "generate_regression_data.cpp"
        tau = 0.5;   // is log(precision)

        Dxy.resize(num_constr, nb);
        Dxy << MatrixXd::Ones(num_constr, nb);

        // FOR NOW only SUM-TO-ZERO constraints feasible
        e.resize(num_constr);
        e = Vect::Zero(num_constr);

        if(MPI_rank == 0)
            std::cout << "generate constraint regression data." << std::endl;
        generate_ex_regression_constr(nb, no, tau, Dxy, e, Prec, B, b, y);
        if(MPI_rank == 0)
            std::cout << "b = " << b.transpose() << std::endl;

        // compute true UNCONSTRAINED marginal variances as
        // Q_xy = Q_x + t(B)*Q_e*B, where Q_e = exp(tau)*I  
        MatrixXd Q_xy_true = Prec + exp(tau)*B.transpose()*B;
        MatrixXd Cov_xy_true = Q_xy_true.inverse(); // need to call pardiso here to be efficient ...       
        //std::cout << "Cov_xy_true = \n" << Cov_xy_true << std::endl;
        if(MPI_rank == 0)
            std::cout << "unconstrained variances fixed eff : " << Cov_xy_true.diagonal().transpose() << std::endl;

        // update for constraints  -> would need a solver if Cov_xy_true not available ...
        MatrixXd constraint_Cov_xy_true = Cov_xy_true - Cov_xy_true*Dxy.transpose()*(Dxy*Cov_xy_true*Dxy.transpose()).inverse()*Dxy*Cov_xy_true;
        if(MPI_rank == 0)
            std::cout << "constrained variances fixed eff   : " << constraint_Cov_xy_true.diagonal().transpose() << std::endl;
    
        // read in design matrix 
        // files containing B
        /*std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }

        B = read_matrix(B_file, no, nb);*/

        //std::cout << "y : \n"  << y << std::endl;    
        //std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

        if(MPI_rank == 0)
            std::cout << "spatial model." << std::endl;

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

        // TODO: fix.
        if(constr == true){
            // choose true parameters for theta = log precision observations, 
            Vect theta_original(dim_th);
            theta_original << 0.5, log(4), log(1);  // -> not actually the true parameters!!
            //std::cout << "theta_original = " << theta_original.transpose() << std::endl; 

            // assemble Qs ... 
            MatrixXd Qs = pow(exp(theta_original[1]),2)*(pow(exp(theta_original[2]), 4) * c0 + 2*pow(exp(theta_original[2]),2) * g1 + g2);

            Dx.resize(num_constr, ns);
            Dx << MatrixXd::Ones(num_constr, ns);

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);

            // FOR NOW only SUM-TO-ZERO constraints feasible
            e.resize(num_constr);
            e = Vect::Zero(num_constr);            

            if(MPI_rank == 0)
                std::cout << "generate constrained spatial data." << std::endl;
            generate_ex_spatial_constr(ns, nb, no, theta_original, Qs, Ax, Dx, e, Prec, B, b, u, y);


            exit(1);

        }

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        if(MPI_rank == 0)
            std::cout << "spatial-temporal model." << std::endl;
        
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

        /*if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }*/

    } else {
        if(MPI_rank == 0){
            std::cout << "invalid parameters : ns nt !!" << std::endl;
            exit(1);
        }    
    }


//#ifdef DATA_TEMPERATURE
    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);   
//#endif


#ifdef DATA_SYNTHETIC
    if(constr == false){
        // data y
        std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
        file_exists(y_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y = read_matrix(y_file, no, 1);
    }
#endif


    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);             // define initial guess in model parametrization
    Vect theta_param(dim_th);       // or in interpretable parametrization
    Vect theta_prior_param(dim_th);
    Vect theta_original(dim_th); theta_original.setZero();

    std::string data_type;

    // initialise theta
    if(ns == 0 && nt == 0){

        theta_original[0] = tau;

        // Initial guess
        theta[0] = 3;
        theta_prior_param[0] = theta[0];


        if(MPI_rank == 0){
            std::cout << "initial theta : "  << theta.transpose() << std::endl; 
        }   

    } else if(ns > 0 && nt == 1){
        //theta << 1, -1, 1;
        //theta << 1, -2, 2;
        //theta_prior << 0, 0, 0;
        //std::cout << "using Elias TOY DATASET" << std::endl;
        // from INLA : log prec Gauss obs, log(Range) for i, log(Stdev) for i     
        //theta_prior << 1.0087220,  -1.0536157, 0.6320466;
        theta_prior_param << 1, -2.3, 2.1;
        theta << theta_prior_param;

        std::cout << "initial theta : "  << theta.transpose() << std::endl;   

    } else {

#ifdef DATA_SYNTHETIC
        data_type = "synthetic";

        // =========== synthetic data set =============== //
        if(MPI_rank == 0){ 
            std::cout << "using SYNTHETIC DATASET" << std::endl; 
        }     
        // sigma.e (noise observations), sigma.u, range s, range t
        //theta_original_param << 0.5, 4, 1, 10;
        // sigma.e (noise observations), gamma_E, gamma_s, gamma_t
        theta_original << 1.386294, -5.882541,  1.039721,  3.688879;  // here exact solution, here sigma.u = 4
        //theta_prior << 1.386294, -5.594859,  1.039721,  3.688879; // here sigma.u = 3
        //theta_prior << 1.386294, -5.594859, 1.039721,  3.688879; // here sigma.u = 3
        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //theta_param << 1.373900, 2.401475, 0.046548, 1.423546; 
        //theta << 1, -3, 1, 3;   // -> the one used so far !! maybe a bit too close ... 
        theta_param << 4, 0, 0, 0;
        //theta_param << 4,4,4,4;
        //theta_param << 1.366087, 2.350673, 0.030923, 1.405511;
        /*theta << 2, -3, 1.5, 5;
        if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/


        if(constr == true){

            if(MPI_rank == 0)
                std::cout << "assuming sum-to-zero constraints on spatial-temporal field." << std::endl;
            // sum to zero constraint for latent parameters
            // construct vector (in GMRF book called A, I will call it D) as D=diag(kron(M0, c0)) 
            // in equidistant mesh this would be a vector of all ones, we want sum D_i x_i = 0
            Dx.resize(num_constr, ns*nt);
            SpMat D = KroneckerProductSparse<SpMat, SpMat>(M0, c0);
            //Vect D_diag = D.diagonal();
            Dx = D.diagonal().transpose();
            //Dx << MatrixXd::Ones(num_constr, ns*nt);

            // rescale Dx such that each row sums to one
            for(int i=0; i<num_constr; i++){
                double sum_row = Dx.row(i).sum();
                Dx = 1/sum_row*Dx;
            }

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);

            // FOR NOW only SUM-TO-ZERO constraints feasible
            e.resize(num_constr);
            e = Vect::Zero(num_constr);            

            /*if(MPI_rank == 0)
                std::cout << "constrain spatial-temporal data." << std::endl;
            
            // assemble Qst ... 
            SpMat Qst(ns*nt, ns*nt);
            construct_Q_spat_temp(theta_original, c0, g1, g2, g3, M0, M1, M2, Qst);
            MatrixXd Qst_d = MatrixXd(Qst);

            generate_ex_spatial_temporal_constr(ns, nt, nb, no, theta_original, Qst_d, Ax, Dx, e, Prec, b, u, y);

            Vect x(n);
            x << u, b;
            //std::cout << "true x = " << x.transpose() << std::endl;

            // compute true UNCONSTRAINED marginal variances as
            MatrixXd Q_x(n,n);
            Q_x << Qst_d, MatrixXd::Zero(ns*nt, nb), MatrixXd::Zero(nb, ns*nt), Prec;
            //std::cout << "Q_x : \n" << Q_x.block(0,0,20,20) << std::endl;
            //std::cout << "dim(Q_x) = " << Q_x.rows() << " " << Q_x.cols() << std::endl;
            //std::cout << "Q_x : \n" << Q_x(127,127) << std::endl; //Q_x.block(110,110,127,127)

            //Q_xy = Q_x + t(Ax)*Q_e*Ax, where Q_e = exp(tau)*I  -> how to incorporate constraints?
            MatrixXd Q_xy_true = Q_x + exp(theta_original[0])*Ax.transpose()*Ax;
            MatrixXd Cov_xy_true = Q_xy_true.inverse(); // need to call pardiso here to be efficient ...       
            //std::cout << "Cov_xy_true = \n" << Cov_xy_true << std::endl;
            if(MPI_rank == 0)
                std::cout << "unconstr. var. fixed eff : " << Cov_xy_true.diagonal().tail(nb).transpose() << std::endl;

            // update for constraints
            MatrixXd constraint_Cov_xy_true = Cov_xy_true - Cov_xy_true*Dxy.transpose()*(Dxy*Cov_xy_true*Dxy.transpose()).inverse()*Dxy*Cov_xy_true;
            if(MPI_rank == 0)
                std::cout << "constr. var. fixed eff   : " << constraint_Cov_xy_true.diagonal().tail(nb).transpose() << std::endl;

            //exit(1);*/

        }

#elif defined(DATA_TEMPERATURE)

        // =========== temperature data set =============== //
        data_type = "temperature";

        if(MPI_rank == 0){
            std::cout << "using TEMPERATURE DATASET" << std::endl; 
            if(constr)
                std::cout << "assuming sum-to-zero constraints on spatial-temporal field." << std::endl;
        }
        //theta << 4, 4, 4, 4;    // -> converges to wrong solution
        theta_param << 4, 0, 0, 0;
        //theta_param << -1.308664,  0.498426,  4.776162,  1.451209;
        //theta_param << -1.269992, 12.132359, 9.772552, 4.710185;
        //theta_param << -1.25, 13.6, 10.4, 5.4;
        theta_original << -1.269613,  5.424197, -8.734293, -6.026165;  // estimated from INLA / same for my code varies a bit according to problem size


        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        //theta << -0.2, -2, -2, 3;
        /*if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/

        if(constr){

            // set up constraints Dx = e
            Dx.resize(num_constr, ns*nt);
            /*SpMat D = KroneckerProductSparse<SpMat, SpMat>(M0, c0);
            Dx = D.diagonal().transpose();
            if(MPI_rank == 0){
                std::cout << "sum(Dx)  = " << Dx.row(0).sum() << std::endl;
                //std::cout << "Dx(1:50) = " << Dx.block(0,0,1,50) << std::endl;
            }*/
            Dx << MatrixXd::Ones(num_constr, ns*nt);

            // rescale Dx such that each row sums to one
            for(int i=0; i<num_constr; i++){
                double sum_row = Dx.row(i).sum();
                Dx = 1/sum_row*Dx;
            }

            if(MPI_rank == 0){
                std::cout << "sum(Dx)  = " << Dx.row(0).sum() << std::endl;
                //std::cout << "Dx(1:50) = " << Dx.block(0,0,1,50) << std::endl;
            }

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);


            // FOR NOW only SUM-TO-ZERO constraints possible
            e.resize(num_constr);
            e = Vect::Zero(num_constr); 

            //exit(1);
        }

#else 
        std::cerr << "\nUnknown datatype! Choose synthetic or temperature dataset!" << std::endl;
        exit(1);
#endif

    }

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
    // TODO: stepsize too small? seems like it almost always accepts step first step.    
    // changed BFGS convergence criterion, now stopping when abs(f(x_k) - f(x_k-1)) < delta
    // is this sufficiently bullet proof?!
    //param.delta = 1e-2;
    param.delta = 1e-3;
    // maximum line search iterations
    param.max_iterations = 100;

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
        if(MPI_rank == 0){
            std::cout << "Call constructor for regression model." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, B, y, theta_prior_param, solver_type, constr, Dxy);
    } else if(ns > 0 && nt == 1) {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial constructor." << std::endl;
        }
        // PostTheta fun(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior_param, solver_type, constr, Dx, Dxy);
    } else {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial-temporal constructor." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type, constr, Dx, Dxy);
    }

    if(MPI_rank == 0)
        printf("\n======================= HYPERPARAMETERS =====================\n");

    if(MPI_rank == 0){
        std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        std::cout << "theta prior param  : " << theta_prior_param.transpose() << std::endl;
    }

    // convert from interpretable parametrisation to internal one
    if(dim_th == 4){
        theta[0] = theta_param[0];
        fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);

        if(MPI_rank == 0){
            Vect theta_interpret_initial(dim_th);
            theta_interpret_initial[0] = theta[0];
            fun->convert_theta2interpret(theta[1], theta[2], theta[3], theta_interpret_initial[1], theta_interpret_initial[2], theta_interpret_initial[3]);
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
            std::cout << "initial theta interpret. param. : " << theta_interpret_initial.transpose() << std::endl;
        }
    }


#if 0 // for testing constraints

    //int m = 5;
    //int num_constr = 1;

    //MatrixXd D(num_constr, nb);
    //Vect e(num_constr);
    MatrixXd Cov = Prec.inverse();
    //Vect mu_normal(nb);
    //mu_normal << 0,0,0,0; //1,2,3,4;
    Vect mu_normal = b;
    Vect rhs_normal = Vect::Zero(nb);

    //fun->generate_test_constraints(m, num_constr, D, e, Cov, mu_normal, rhs_normal);
    //std::cout << "Cov = \n" << Cov << std::endl;
    //std::cout << "Dx = " << Dx << ", e = " << e.transpose() << std::endl;
    std::cout << "mu = " << mu_normal.transpose() << "\nrhs = " << rhs_normal.transpose() << std::endl;

    MatrixXd V = Cov*Dx.transpose();
    MatrixXd W(num_constr, num_constr);
    MatrixXd U(num_constr, nb);
    Vect constr_mu_normal(nb);

    fun->update_mean_constr(Dx, e, mu_normal, V, W, U, constr_mu_normal);
    std::cout << "constrained mu = " << constr_mu_normal.transpose() << std::endl;

    // constr_mu_st will by definition satisfy Ax = e, hence choose x = constr_mu_xy
    //Vect x_normal = constr_mu_normal;
    Vect x_normal = Vect::Zero(nb);
    SpMat Q = Cov.inverse().sparseView();
    double log_det_Q = - log(Cov.determinant());

    double val;

    fun->eval_log_dens_constr(x_normal, mu_normal, Q, log_det_Q, Dx, W, val);

    // ================================================================================================== //
    // compute control, can get very inaccurate very quickly as dimension increases !!! pseudo-inverse not numerically stable ...    
    // compute constrained mean and covariance
    MatrixXd invW = (Dx*Cov*Dx.transpose()).inverse();
    //std::cout << "W = " << W << ", W = " << Dx*Cov*Dx.transpose() << std::endl;
    //std::cout << "inv(W) = " << W.inverse() << ", invW = " << invW << std::endl;
    Vect constr_mu_normal2 = mu_normal - Cov*Dx.transpose()*invW*(Dx*mu_normal - e);
    //std::cout << "norm(constr_mu_xy - constr_mu_xy2) = " << (constr_mu_normal - constr_mu_normal2).norm() << std::endl;
    //std::cout << "constr_mu = " << constr_mu.transpose() << std::endl;
    MatrixXd constr_Cov = Cov - Cov*Dx.transpose()*invW*Dx*Cov;
    //std::cout << "constr_Cov = \n" << constr_Cov << std::endl;

    EigenSolver<MatrixXd> es(constr_Cov);
    MatrixXd EV = es.eigenvectors().real();
    //cout << "Eigenvectors = " << endl << V << endl;

    Vect eivals = es.eigenvalues().real();
    //std::cout << "eigenvalues(constr_Cov) = " <<  eivals.transpose() << std::endl;
    //std::cout << "V*D*V^T = " << endl << V*eivals.asDiagonal()*V.transpose() << std::endl;
          
    // identify non-zero eigenvalues
    double log_sum = 0;
    double prod = 1;
    Vect invD = Vect::Zero(nb);
    for(int i=0; i<nb; i++){
        if(eivals[i] > 1e-7){
            log_sum += log(eivals[i]);
            prod *= eivals[i];
            invD[i] = 1/eivals[i];
        }
    }

    MatrixXd invD_mat = invD.asDiagonal();
    //std::cout << "invD = " << invD_mat << std::endl;

    //printf("sum(eivals)     = %f\n", log(prod));
    //printf("log_sum(eivals) = %f\n", log_sum);

    // compute log pi(x | Ax = e), evaluated at x, make sure x satisfies condition ...
    //Vect x = Vect::Zero(m);
    //Vect x = constr_mu;
    MatrixXd pInv = EV*invD.asDiagonal()*EV.transpose();
    //std::cout << "pInv = \n" << pInv << std::endl;
    //std::cout << "Cov = \n" << V*eivals.asDiagonal()*V.transpose() << std::endl;
    std::cout << "(V*eivals.asDiagonal()*V.transpose() - constr_Cov).norm() = " << (EV*eivals.asDiagonal()*EV.transpose() - constr_Cov).norm() << std::endl;
    double temp = (x_normal - constr_mu_normal2).transpose()*pInv*(x_normal - constr_mu_normal2);
    std::cout << "temp = " << temp << std::endl;
    double log_val = - 0.5*(Cov.rows()-Dx.rows())*log(2*M_PI) - 0.5*log_sum - 0.5*temp;
    std::cout << - 0.5*Cov.rows()*log(2*M_PI) - (- 0.5*Dx.rows()*log(2*M_PI)) << " " << - 0.5*(Cov.rows()-Dx.rows())*log(2*M_PI) << std::endl;
    std::cout << "log val direct = " << log_val << std::endl;
    
#endif    



#if 0   // for measuring noise

    PostTheta* fun2;
    fun2 = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type);

    Vect theta2 = theta;

    Vect mu(n);
    Vect grad(dim_th);

    /*if(MPI_rank == 0){
        std::cout << "\ntheta : " << std::right << std::fixed << std::setprecision(4) << theta.transpose() << std::endl;
    }*/

    for(int i = 0; i<1; i++){
        if(MPI_rank == 0){
                double f_theta = fun->eval_post_theta(theta, mu); // gets called by each rank individually
        }
        //double f_theta = fun->operator()(theta, grad);  // same for all ranks
        /*if(MPI_rank == 0){  
            std::cout << "f_theta : " << std::right << std::fixed << std::setprecision(12) << f_theta << std::endl;
            std::cout << "grad    : " << std::right << std::fixed << std::setprecision(12) << grad.transpose()  << std::endl;
        }*/
    }

    if(MPI_rank == 0){
        std::cout << "\n============================================================================\n" << std::endl;
        std::cout << "theta2 : " << theta2.transpose() << std::endl;
    }

    for(int i = 0; i<1; i++){
        if(MPI_rank == 0){
            double f_theta = fun2->eval_post_theta(theta, mu); // gets called by each rank individually
        }
        //double f_theta = fun2->operator()(theta, grad);  // same for all ranks
        /*if(MPI_rank == 0){  
            std::cout << "f_theta : " << std::right << std::fixed << std::setprecision(12) << f_theta << std::endl;
            std::cout << "grad    : " << std::right << std::fixed << std::setprecision(12) << grad.transpose()  << std::endl;
        }*/
    }

#endif


#if 1

    double fx;

    //Vect grad_test(dim_th);
    //fx = fun(theta, grad_test);
    //std::cout <<  "f(x) = " << fx << std::endl;

    if(MPI_rank == 0)
        printf("\n====================== CALL BFGS SOLVER =====================\n");

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
        std::cout << "grad                         : " << grad.transpose() << std::endl;
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

    #if 0
    Vect theta_max(dim_th);
    //theta_max << 2.675054, -2.970111, 1.537331;    // theta
    //theta_max = theta_prior;
    theta_max = theta;

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
        std::cout << "estimated mean random effects: " << mu.head(50).transpose() << std::endl;
        //std::cout << "time get fixed effects       : " << t_get_fixed_eff << " sec\n" << std::endl;
//#if PRINT_MSG
        if(constr == true)
            std::cout << "Dxy*mu : " << Dxy*mu << ", should be : " << e << std::endl;
//#endif
    }

    #endif

  
    // =================================== compute marginal variances =================================== //
#if 1

    double t_get_marginals;
    Vect marg(n);

    // when the range of u is large the variance of b0 is large.
    if(MPI_rank == 0){

        std::cout << "==================== compute marginal variances ================" << std::endl;
        //theta << -1.269613,  5.424197, -8.734293, -6.026165; // most common solution for temperature dataset
        std::cout << "\nUSING ESTIMATED THETA : " << theta.transpose() << std::endl;

        t_get_marginals = -omp_get_wtime();
        fun->get_marginals_f(theta, marg);
        t_get_marginals += omp_get_wtime();

        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;

#ifdef DATA_SYNTHETIC
        // ============================================ //
        Vect marg_original(n);
        fun->get_marginals_f(theta_original, marg_original);

        std::cout << "\nUSING ORIGINAL THETA : " << theta_original.transpose() << std::endl;
        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg_original.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg_original.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;
#endif

        // ============================================ //
        Vect theta_INLA(dim_th);
#ifdef DATA_SYNTHETIC
        if(no == 128400)
            theta_INLA << 1.384902, -5.880753, 1.015273, 3.709717;  // 642, 20, 128400
        if(no == 25680)
            theta_INLA << 1.393959, -5.915735, 1.262288, 3.475778;    // 642, 20,  25680
        if(no == 2520)
            theta_INLA << 1.306129, -5.840268, 0.879396, 3.718468;
        if(no == 630)
             theta_INLA <<  2.005811, -6.074556, 1.112258,  3.828027;
#elif defined(DATA_TEMPERATURE)
        Vect theta_INLA_param(4);
        theta_INLA_param << -1.270, 12.132, 9.773, 4.710;
        //theta_INLA_param << -1.352, 1.912, 3.301, 3.051;

        theta_INLA[0] = theta_INLA_param[0];
        fun->convert_interpret2theta(theta_INLA_param[1], theta_INLA_param[2], theta_INLA_param[3], theta_INLA[1], theta_INLA[2], theta_INLA[3]);

#endif

        Vect marg_INLA(n);
        fun->get_marginals_f(theta_INLA, marg_INLA);

        std::cout << "\nUSING THETA FROM INLA : " << theta_INLA.transpose() << std::endl;
        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg_INLA.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg_INLA.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;
    }
#endif  // #if 0/1 for marginals

#if 0
    // =================== expected marginal variances ============================ //
    // first way
    // construct Q = Qx(theta_mode) + theta_mode*AxTAx => assume Q_b = 1e-5*Id
    // compute inverse -> using pardiso & using .inverse() => extract diagonal => compare

    SpMat Qst(n,n);
    construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, Qst);

    SpMat Qx(n,n);        
    int nnz = Qst.nonZeros();
    Qx.reserve(nnz);

    for (int k=0; k<Qst.outerSize(); ++k){
      for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
      {
        Qx.insert(it.row(),it.col()) = it.value();                 
      }
    }

    for(int i=ns*nt; i < n; i++){
        Qx.coeffRef(i,i) = 1e-3;
    }

    Qx.makeCompressed();

    SpMat Q =  Qx + exp(theta[0]) * Ax.transpose()*Ax;


    if(MPI_rank == 0){
        //std::cout << "Qx(0,0,10,10) : \n" << Qx.block(0,0,10,10) << std::endl;
        //std::cout << "Ax(0,0,30,30) : \n" << Ax.block(0,0,30,30) << std::endl;
        
        //MatrixXd AxTAx = exp(theta[0]) * Ax.transpose()*Ax;
        //std::cout << "exp(theta[0]) = " << exp(theta[0]) << std::endl;
        //std::cout << "AxTAx.topLeft(30,30) : \n" << AxTAx.topLeftCorner(30,30) << std::endl;
        //std::cout << "AxTAx.bottomRight(10,10) : \n" << AxTAx.bottomRightCorner(10,10) << std::endl;


        MatrixXd Q_d = MatrixXd(Q);
        std::cout << "Q_topLeft(0,0,10,10) : \n" << Q_d.topLeftCorner(10,10) << std::endl;
        std::cout << "Q_bottomRight(0,0,10,10) : \n" << Q_d.bottomRightCorner(10,10) << std::endl;

        /*MatrixXd invQ_d = Q_d.inverse();
        std::cout << "\nmarginals fixed effects = " << invQ_d.diagonal().tail(10).transpose() << std::endl;
        std::cout << "marginals random effects= " << invQ_d.diagonal().head(10).transpose() << std::endl;*/

        Solver* solverQ;
        solverQ   = new PardisoSolver(MPI_rank);

        Vect inv_diag(n);
        solverQ->selected_inversion(Q, inv_diag);
        std::cout << "\nmarg std dev FE 'true'       : " << inv_diag.tail(10).cwiseSqrt().transpose() << std::endl;
        std::cout << "marg std dev RE 'true'       : " << inv_diag.head(10).cwiseSqrt().transpose() << std::endl;

        // invert using eigen
        /*SimplicialLDLT<SparseMatrix<double>> solverEigen;
        solverEigen.compute(Q);

        SpMat I(n,n); I.setIdentity();
        MatrixXd fullInv(n,n);

        std::cout << "identity matrix : \n" << I.block(0,0,10,10) << std::endl;

        fullInv = solverEigen.solve(I);
        std::cout << "norm(I - Q*fullInv) = " << (I - Q*fullInv).norm() << std::endl;

        std::cout << "marg std dev FE fullInv      : " << fullInv.diagonal().tail(10).cwiseSqrt().transpose() << std::endl;
        std::cout << "marg std dev RE fullInv      : " << fullInv.diagonal().head(10).cwiseSqrt().transpose() << std::endl;
        */

    }    
#endif

    // =================================== print times =================================== //
    #if 0
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
