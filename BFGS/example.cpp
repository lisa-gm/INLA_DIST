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

#include "theta_function.cpp"
#include "../read_write_functions.cpp"


using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;


using namespace LBFGSpp;

// for now use armadillo ... do better once we switch to binary

/*MatrixXd read_matrix(const std::string filename,  int n_row, int n_col){

    arma::mat X(n_row, n_col);
    X.load(filename, arma::raw_ascii);
    // X.print();

    return Eigen::Map<MatrixXd>(X.memptr(), X.n_rows, X.n_cols);
}*/

/*void file_exists(std::string file_name)
{
    if (std::fstream{file_name}) ;
    else {
      std::cerr << file_name << " couldn\'t be opened (not existing or failed to open)\n"; 
      exit(1);
    }
    
}*/


void rnorm_gen(int no, double mean, double sd,  Eigen::VectorXd * x, int seed){
  // unsigned int seed = 2;
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution (mean, sd);
 
  for (int i=0; i< x->size(); ++i){
        (*x)(i)  = distribution(generator);
  }

}

void generate_ex_regression( int nb,  int no, double tau, Eigen::MatrixXd *B, Vector *b, Vector *y){

    /* ---------------------- construct random matrix of covariates --------------------- */
    Vector B_ones(no); B_ones.setOnes();

    // require different random seed here than in noise -> otherwise cancels each other out
    // val_l will then equal val_d ... 
    Vector B_random(no*(nb-1));
    rnorm_gen(no, 0.0, 1, &B_random, 2);

    Vector B_tmp(no*nb);
    B_tmp << B_ones, B_random;
    //std::cout << B_tmp << std::endl;

    // TODO: fix this!
    Eigen::Map<Eigen::MatrixXd> tmp(B_tmp.data(), no,nb);
    *B = tmp;
    //*B(B_tmp.data());
    //Eigen::MatrixXd::Map(*B) = B_tmp.data(); 
    //std::cout << *B << std::endl;

    /* -------  construct random solution vector of fixed effects & observations -------- */
    *b = 2*(Vector::Random(nb) + Vector::Ones(nb)); 

    double mean = 0.0;
    double sd = 1/sqrt(exp(tau));
    Vector noise_vec(no);

    rnorm_gen(no, mean, sd, &noise_vec, 4);

    *y = (*B)*(*b) + noise_vec;

    /*std::cout << "noise vec " << std::endl;
    std::cout << noise_vec << std::endl; 

    std::cout << "B*b " << std::endl;
    std::cout << (*B)*(*b) << std::endl;

    std::cout << "y " << std::endl;
    std::cout << *y << std::endl; */

}


class my_function
{
private:
    int no;
    int nb;
    MatrixXd B;
    VectorXd y;

public:
    my_function(int no_, int nb_, MatrixXd B_, VectorXd y_) : no(no_), nb(nb_), B(B_), y(y_) {}
    double operator()(const VectorXd& x, VectorXd& grad)
    {

        // double fx = 0.0;

        // double t1 = 1.0 - x[0];
        // double t1 = (1.0 - x[0])*(1 - x[0]);
        // double t2 = 10 * (x[0 + 1] - x[0] * x[0]);
        // grad[1] = 20 * t2;
        // grad[0]     = -2.0 * (x[0] * grad[0 + 1] + t1);
        // fx += t1 * t1 + t2 * t2;

        double fx = 0.0;
        double t1 = (1.0 - x[0])*(1 - x[0]);
        double t2 = 100*(x[1] - x[0]*x[0])*(x[1] - x[0]*x[0]);
        fx = t1 + t2;

        gradient(x,grad);
        // grad[0] = -2.0*(1.0 - x[0]) - 4*100.0*x[0]*(x[1] - x[0]*x[0]);
        // grad[1] = 2.0*100.0*(x[1] - x[0]*x[0]);
        

        return fx;
    }

    void gradient(const VectorXd& x, VectorXd& grad){
        grad[0] = -2.0*(1.0 - x[0]) - 4*100.0*x[0]*(x[1] - x[0]*x[0]);
        grad[1] = 2.0*100.0*(x[1] - x[0]*x[0]);
    }
    
};


int main(int argc, char* argv[])
{

    
    #if 0
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

    #if 1

    if(argc != 1 + 4){
        std::cout << "wrong number of input parameters. " << std::endl;
        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns = atoi(argv[1]);
    //int nt = atoi
    size_t nb = atoi(argv[2]);
    size_t no = atoi(argv[3]);

    // also save as string
    std::string ns_s = std::to_string(ns);
    //std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string nu_s = std::to_string(ns + nb);


    std::string base_path = argv[4];

    // dimension hyperparamter initialised to 1
    int dim_th = 1;

    /* ---------------- read in matrices ---------------- */

    MatrixXd B;
    SpMat Ax; 
    SpMat c0; 
    SpMat g1; 
    SpMat g2;

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);

    Vector y = read_matrix(y_file, no, 1);

    if(ns == 0 ){
        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        B = read_matrix(B_file, no, nb);
    }

    // std::cout << "y : \n"  << y << std::endl;    
    // std::cout << "B : \n" << B << std::endl;

    // read in spatial component of the model
    if(ns > 0){
        dim_th = 3;

        // check spatial FEM matrices
        std::string c0_file       =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file       =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file       =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + nu_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);

        Ax = readCSC(Ax_file);

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    }

    #endif

    Vector theta(dim_th);
    // initialise theta
    if(ns == 0){
        // Initial guess
        theta[0] = 3;
    } else {
        theta << 0.5, 0.5, 0.5;
        std::cout << "theta : \n"  << theta.transpose() << std::endl;    

    }

    //exit(1);

    Vector b(nb);
  
    // Set up parameters
    LBFGSParam<double> param;    
    // set convergence criteria
    // stop if norm of gradient smaller than :
    param.epsilon = 1e-1;
    // or if objective function has not decreased by more than  
    param.epsilon_rel = 1e-1;
    // in the past ... steps
    param.past = 1;
    // maximum line search iterations
    param.max_iterations = 10;


    // Create solver and function object
    LBFGSSolver<double> solver(param);

    /*std::optional<PostTheta> fun;

    if(ns == 0){
        fun.emplace(nb, no, B, y);
    }*/

    // PostTheta fun(nb, no, B, y);
    PostTheta fun(ns, nb, no, Ax, y, c0, g1, g2);
       
    double fx;

    // Vector grad(1);
    // fx = fun(theta, grad);
    // std::cout <<  "f(x) = " << fx << std::endl;

    int niter = solver.minimize(fun, theta, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    Vector grad = fun.get_grad();
    std::cout << "grad = " << grad << std::endl;

    // std::cout << "original theta             : " << tau << std::endl;
    std::cout << "estimated theta            : " << theta.transpose() << std::endl;

    //MatrixXd cov = fun.get_Covariance(theta);
    //std::cout << "estimated covariance theta : " << cov << std::endl;

    //std::cout << "original fixed effects     : " << b.transpose() << std::endl;
    Vector mu = fun.get_mu();    
    std::cout << "estimated fixed & random effects    : " << mu.transpose() << std::endl;

    //Vector marg = fun.get_marginals_f(theta);
    //std::cout << "est. marginals fixed eff.  : " << marg.transpose() << std::endl;


    return 0;
}
