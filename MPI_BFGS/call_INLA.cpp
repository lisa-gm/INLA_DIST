#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>

#include "mpi.h"

//#include <optional>

#include <armadillo>

#include "PostTheta.h"
#include "Model.h"
#include "../read_write_functions.cpp"

//#include "PardisoSolver.h"
//#include "RGFSolver.h"
#include <unsupported/Eigen/KroneckerProduct>

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;


using namespace LBFGSpp;

/* ===================================================================== */

int main(int argc, char* argv[])
{
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
    Vector b(nb);
    Vector y(no);

    double tau = 0.5;
    generate_ex_regression(nb, no, tau, &B, &b, &y); 
    
    // Initial guess
    Vector theta(1);
    theta[0] = 3;
    
    
    #endif

    if(argc != 1 + 6){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        exit(1);
    }

    if(MPI_rank == 1){
        std::cout << "reading in example. " << std::endl;
    }

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    size_t no = atoi(argv[4]);

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

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
    Vector y;

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        B = read_matrix(B_file, no, nb);

        // std::cout << "y : \n"  << y << std::endl;    
        // std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

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

        Ax = readCSC(Ax_file);

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        if(MPI_rank == 1){
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

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
        exit(1);
    }


    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    y = read_matrix(y_file, no, 1);


    /* ----------------------- initialise random theta -------------------------------- */

    Vector theta(dim_th);
    Vector theta_prior(dim_th);

    // initialise theta
    if(ns == 0 && nt == 0){
        // Initial guess
        theta[0] = 3;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else if(ns > 0 && nt == 1){
        //theta << 1, -1, 1;
        theta << 1, -2, 2;
        theta_prior << 0, 0, 0;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else {
        // =========== synthetic data set =============== //
        theta_prior << 1.4, -5.9,  1,  3.7;  // here exact solution
        //theta << 1.4, -5.9,  1,  3.7; 
        theta << 1, -3, 1, 3;
        //theta << 0.5, -1, 2, 2;

        // =========== temperature data set =============== //
        /*theta_prior << -0.294769, -5.670050, -3.452297,  5.627084;       // EU only (solution from INLA)
        //theta_original << 5, -10, 2.5, 1;
        theta << -0.2, -2, -2, 3;*/

        if(MPI_rank == 1){
            std::cout << "using SYNTHETIC DATASET" << std::endl;        
            //std::cout << "using TEMPERATURE DATASET" << std::endl; 
            std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }
    }

    Vector b(nb);

    /*std::cout << "\nspatial grid size  : " << std::right << std::fixed << g1.rows() << " " << g1.cols() << std::endl;
    std::cout << "temporal grid size : " << M1.rows() << " " << M1.cols() << std::endl;
    std::cout << "Ax size            : " << Ax.rows() << " " << Ax.cols() << std::endl;*/

    // ============================ set up Posterior of theta ======================== //

    if(MPI_rank == 0){
        PostTheta* fun;
        fun = new PostTheta(dim_th);

        // ============================ set up BFGS solver ======================== //

        // Set up parameters
        LBFGSParam<double> param;    
        // set convergence criteria
        // stop if norm of gradient smaller than :
        // computed as ||𝑔|| < 𝜖 ⋅ max(1,||𝑥||)
        param.epsilon = 1e-1;
        // or if objective function has not decreased by more than  
        // cant find epsilon_rel in documentation ...
        param.epsilon_rel = 1e-1;
        // in the past ... steps
        param.past = 1;
        // maximum line search iterations
        param.max_iterations = 1;
        // TODO: stepsize too small? seems like it almost always accepts step first step.

        // Create solver and function object
        LBFGSSolver<double> solver(param);

        double fx;

        double time_bfgs = -omp_get_wtime();
        int niter = solver.minimize(*fun, theta, fx);
        time_bfgs += omp_get_wtime();

        std::cout << niter << " iterations" << std::endl;
        std::cout << "BFGS solver time             : " << time_bfgs << " sec" << std::endl;

        Vector grad = fun->get_grad();
        std::cout << "grad                         :" << grad.transpose() << std::endl;

        #if 0 
        Vector theta_max(dim_th);
        //theta_max = theta;
        theta_max << 1.299205, -5.590766,  0.943657,  3.746657;

        Vector mu(n);
        fun->get_mu(theta_max, mu);
        std::cout << "\nestimated mean fixed effects : " << mu.tail(nb).transpose() << std::endl;

        Vector marg(n);
        fun->get_marginals_f(theta_max, marg);
        std::cout << "est. std. dev. fixed eff.    : " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;

        // check if MPI_size is large enough, for dim(theta) = 4, require :
        // 1 + 2 * 4 + 4 * 6 = 33 + 1 (master) = 34
        int required_MPI_size = 34;
        if(MPI_size < required_MPI_size){
            std::cout << "Not enough MPI processes to compute covariance of theta!" << std::endl;

        } else {

            MatrixXd cov = fun->get_Covariance(theta_max);
            //std::cout << "estimated covariance theta   :  \n" << cov << std::endl;

            std::cout << "estimated standard dev theta :  " << cov.cwiseSqrt().diagonal().transpose() << std::endl;
            std::cout << "estimated variances theta    :  " << cov.diagonal().transpose() << std::endl;
        } 
        #endif

        // put all workers to sleep using 0 tag
        for(int i=1; i<MPI_size; i++){
            MPI_Send(NULL, 0, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        delete fun;

    } else {

        Model* model;

        if(ns == 0){
            // fun.emplace(nb, no, B, y);
            model = new Model(ns, nt, nb, no, B, y, theta_prior, solver_type);
        } else if(ns > 0 && nt == 1) {
            std::cout << "\ncall spatial constructor." << std::endl;
            // PostTheta fun(nb, no, B, y);
            model = new Model(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior, solver_type);
        } else {
            if(MPI_rank == 1){
                std::cout << "\ncall spatial-temporal constructor." << std::endl;
            }
            model = new Model(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior, solver_type);
        }

        model->ready();

        delete model;
    }

    MPI_Finalize();

    return 0;
    
}