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

#include <optional>

#include <armadillo>

#include "PostTheta.h"
#include "../read_write_functions.cpp"


using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;


using namespace LBFGSpp;


int main(int argc, char* argv[])
{

    
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

    if(argc != 1 + 5){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

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

    // dimension hyperparamter initialised to 1
    int dim_th = 1;

    /* ---------------- read in matrices ---------------- */

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

    int n;

    // initialise theta
    if(ns == 0 && nt == 0){
        n = nb;
        // Initial guess
        theta[0] = 3;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else if(ns > 0 && nt == 1){
        n = ns + nb;
        //theta << 1, -1, 1;
        theta << 1, -2, 2;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else {
        n = ns*nt + nb;

        // =========== synthetic data set =============== //
        /*std::cout << "using SYNTHETIC DATASET" << std::endl;        
        theta_prior << 1.4, -5.9,  1,  3.7;  // here exact solution
        std::cout << "theta original     : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        //theta << 1.4, -5.9,  1,  3.7; 
        theta << 1, -3, 1, 3;
        //theta << 0.5, -1, 2, 2;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;*/

        // =========== temperature data set =============== //
        std::cout << "using TEMPERATURE DATASET" << std::endl; 
        theta_prior << -0.294769, -5.670050, -3.452297,  5.627084;       // EU only (solution from INLA)
        //theta_original << 5, -10, 2.5, 1;
        std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        theta << -0.2, -2, -2, 3;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;


    }


    //exit(1);

    Vector b(nb);

    // ============================ set up BFGS solver ======================== //

    // Set up parameters
    LBFGSParam<double> param;    
    // set convergence criteria
    // stop if norm of gradient smaller than :
    param.epsilon = 1e-3;
    // or if objective function has not decreased by more than  
    param.epsilon_rel = 1e-1;
    // in the past ... steps
    param.past = 1;
    // maximum line search iterations
    param.max_iterations = 30;

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
        fun = new PostTheta(ns, nt, nb, no, B, y, theta_prior);
    } else if(ns > 0 && nt == 1) {
        std::cout << "\ncall spatial constructor." << std::endl;
        // PostTheta fun(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior);
    } else {
        std::cout << "\ncall spatial-temporal constructor." << std::endl;
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior);
    }

    double fx;

    //Vector grad_test(dim_th);
    //fx = fun(theta, grad_test);
    //std::cout <<  "f(x) = " << fx << std::endl;

    std::cout << "\nCall BFGS solver now. " << std::endl;

    double time_bfgs = -omp_get_wtime();
    int niter = solver.minimize(*fun, theta, fx);

    time_bfgs += omp_get_wtime();

    std::cout << niter << " iterations" << std::endl;
    std::cout << "BFGS solver time             : " << time_bfgs << " sec" << std::endl;

    std::cout << "\nf(x)                         : " << fx << std::endl;

    int fct_count = fun->get_fct_count();
    std::cout << "function counts thread zero  : " << fct_count << std::endl;


    Vector grad = fun->get_grad();
    std::cout << "grad                         : " << grad.transpose() << std::endl;

    std::cout << "\nestimated mean theta         : " << theta.transpose() << std::endl;
    std::cout << "original theta               : " << theta_prior.transpose() << "\n" << std::endl;

    // convert between different theta parametrisations
    double lgamE = theta[1]; double lgamS = theta[2]; double lgamT = theta[3];
    double sigU; double ranS; double ranT;
    fun->convert_theta2interpret(lgamE, lgamS, lgamT, sigU, ranS, ranT);
    std::cout << "est. mean interpret. param.  : " << theta[0] << " " << sigU << " " << ranS << " " << ranT << std::endl;
    
    double prior_sigU; double prior_ranS; double prior_ranT;
    fun->convert_theta2interpret(theta_prior[1], theta_prior[2], theta_prior[3], prior_sigU, prior_ranS, prior_ranT);
    std::cout << "org. mean interpret. param.  : " << theta_prior[0] << " " << prior_sigU << " " << prior_ranS << " " << prior_ranT << std::endl;

    Vector theta_max(dim_th);
    //theta_max << 2.675054, -2.970111, 1.537331;    // theta
    theta_max = theta_prior;
    //theta_max = theta;

    // in what parametrisation are INLA's results ... ?? 
    MatrixXd cov = fun->get_Covariance(theta_max);
    std::cout << "estimated standard dev theta :  " << cov.cwiseSqrt().diagonal().transpose() << std::endl;

    /*std::cout << "estimated covariance theta   :  \n" << cov << std::endl;
    std::cout << "estimated variances theta    :  " << cov.diagonal().transpose() << std::endl;*/

    #if 0


    Vector mu(n);
    fun->get_mu(theta_max, mu);
    std::cout << "\nestimated mean fixed effects : " << mu.tail(nb).transpose() << std::endl;
    
    // when the range of u is large the variance of b0 is large.
    Vector marg(n);
    fun->get_marginals_f(theta_max, marg);
    std::cout << "est. variances fixed eff.    :  " << marg.tail(nb).transpose() << std::endl;

    #endif

    delete fun;

    return 0;
    
}
