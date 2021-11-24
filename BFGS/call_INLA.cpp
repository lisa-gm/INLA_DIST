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
#include <unsupported/Eigen/KroneckerProduct>

#include <armadillo>

#include <LBFGS.h>

//#include <optional>

#include "PostTheta.h"
#include "../read_write_functions.cpp"

//#include "PardisoSolver.h"
//#include "RGFSolver.h"

//#include <likwid-marker.h>



using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;


using namespace LBFGSpp;

/* ===================================================================== */

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

    std::cout << "reading in example. " << std::endl;

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
    std::cout << "Solver : " << solver_type << std::endl;
    
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

        // casting no_s as integer
        no = std::stoi(no_s);
        std::cout << "total number of observations : " << no << std::endl;
      
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

        // doesnt require no to be read, can read no from Ax
        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();
        std::cout << "total number of observations : " << no << std::endl;


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
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();
        std::cout << "total number of observations : " << no << std::endl;

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
        exit(1);
    }

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);


    /* ----------------------- initialise random theta -------------------------------- */

    Vector theta(dim_th);
    Vector theta_prior_param(dim_th);
    Vector theta_original(dim_th);       // only relevant for synthetic dataset

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

        // =========== synthetic data set =============== //
        std::cout << "using SYNTHETIC DATASET" << std::endl;        
        theta_original << 1.4, -5.9,  1,  3.7;  // here exact solution, here sigma.u = 4
        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //theta_prior << 1.386294, -5.594859,  1.039721,  3.688879; // here sigma.u = 3
        //theta_prior << 1.386294, -5.594859, 1.039721,  3.688879; // here sigma.u = 3
        std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        //theta << 1.4, -5.9,  1,  3.7; 
        theta << 1, -3, 1, 3;
        //theta << 0.5, -1, 2, 2;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;

        // =========== temperature data set =============== //
        /*std::cout << "using TEMPERATURE DATASET" << std::endl; 
        theta_prior << -0.294769, -5.670050, -3.452297,  5.627084;       // EU only (solution from INLA)
        //theta_original << 5, -10, 2.5, 1;
        std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        theta << -0.2, -2, -2, 3;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;*/
    }

    Vector b(nb);

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
    param.max_iterations = 30;
    // TODO: stepsize too small? seems like it almost always accepts step first step.
    // changed BFGS convergence criterion, now stopping when abs(f(x_k) - f(x_k-1)) < delta
    // is this sufficiently bullet proof?!
    param.delta = 1e-1;

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
        std::cout << "\ncall spatial constructor." << std::endl;
        // PostTheta fun(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior_param, solver_type);
    } else {
        std::cout << "\ncall spatial-temporal constructor." << std::endl;
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type);
    }

    #if 1
    double fx;

    //Vector grad_test(dim_th);
    //fx = fun(theta, grad_test);
    //std::cout <<  "f(x) = " << fx << std::endl;

    std::cout << "\nCall BFGS solver now. " << std::endl;

    //LIKWID_MARKER_INIT;
    //LIKWID_MARKER_THREADINIT;

    double time_bfgs = -omp_get_wtime();
    int niter = solver.minimize(*fun, theta, fx);

    //LIKWID_MARKER_CLOSE;

    time_bfgs += omp_get_wtime();

    std::cout << niter << " iterations" << std::endl;
    std::cout << "BFGS solver time             : " << time_bfgs << " sec" << std::endl;

    std::cout << "\nf(x)                         : " << fx << std::endl;

    /*int fct_count = fun->get_fct_count();
    std::cout << "function counts thread zero  : " << fct_count << std::endl;*/

    Vector grad = fun->get_grad();
    std::cout << "grad                         :" << grad.transpose() << std::endl;

    /*std::cout << "\nestimated mean theta         : " << theta.transpose() << std::endl;
    std::cout << "original theta               : " << theta_prior.transpose() << "\n" << std::endl;*/

    /*double eps = 0.005;
    VectorXd temp(4);
    temp << -5,2,3,-2;
    double f_temp = fun->f_eval(temp);
    std::cout << "f eval test : " << f_temp << endl;
    MatrixXd cov = fun->get_Covariance(temp, eps);
    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;*/

    std::cout << "\norig. mean parameters        : " << theta_original.transpose() << std::endl;
    std::cout << "est.  mean parameters        : " << theta.transpose() << std::endl;

    // convert between different theta parametrisations
    if(dim_th == 4){
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

    Vector theta_max(dim_th);
    //theta_max << 2.675054, -2.970111, 1.537331;    // theta
    //theta_max = theta_prior;
    theta_max = theta;
    //theta_max << 1.382388, -5.626002,  1.156931,  3.644319;
    //theta_max << 1.388921, -5.588113,  0.985369,  3.719458;
    //theta_max << 1.299205, -5.590766,  0.943657,  3.746657;
    //theta_max << 1.4608052, -5.8996978,  0.6805342,  3.8358287; 

    /*std::cout << "Estimated Covariance Matrix INLA : " << std::endl;
    MatrixXd Cov_INLA(4,4);

    Cov_INLA << 0.023833160, 0.01486733, 0.004853688, 0.005288554,
                0.014867325, 0.12749968, 0.056587582, 0.048833876,
                0.004853688, 0.05658758, 0.025517230, 0.022059932,
                0.005288554, 0.04883388, 0.022059932, 0.019274723;

    std::cout << Cov_INLA << std::endl;*/

    // in what parametrisation are INLA's results ... ?? 
    double eps;
    MatrixXd cov(dim_th,dim_th);

    /*eps = 0.01;
    cov = fun->get_Covariance(theta_max, eps);
    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;*/

    eps = 0.005;
    //cov = fun->get_Covariance(theta_max, sqrt(eps));
    cov = fun->get_Covariance(theta_max, eps);
    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;

    /*eps = 0.001;
    cov = fun->get_Covariance(theta_max, eps);
    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;*/
    std::cout << "estimated variances theta    :  " << cov.diagonal().transpose() << std::endl;
    std::cout << "estimated standard dev theta :  " << cov.cwiseSqrt().diagonal().transpose() << std::endl;

    //convert to interpretable parameters
    // order of variables : gaussian obs, range t, range s, sigma u
    Vector interpret_theta(4);
    interpret_theta[0] = theta_max[0];
    fun->convert_theta2interpret(theta_max[1], theta_max[2], theta_max[3], interpret_theta[1], interpret_theta[2], interpret_theta[3]);
    std::cout << "est.  mean interpret. param. : " << interpret_theta[0] << " " << interpret_theta[1] << " " << interpret_theta[2] << " " << interpret_theta[3] << std::endl;

    cov = fun->get_Cov_interpret_param(interpret_theta, eps);

    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;


    #endif

    #if 0

    Vector mu(n);
    fun->get_mu(theta, mu);
    std::cout << "\nestimated mean fixed effects : " << mu.tail(nb).transpose() << std::endl;
    
    // when the range of u is large the variance of b0 is large.
    Vector marg(n);
    fun->get_marginals_f(theta, marg);
    std::cout << "est. variances fixed eff.    :  " << marg.tail(nb).transpose() << std::endl;
    std::cout << "est. standard dev fixed eff  :  " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;


    #endif


    delete fun;
    return 0;
    
}
