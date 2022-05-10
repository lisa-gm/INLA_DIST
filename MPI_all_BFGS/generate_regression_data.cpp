// probably don't need all of them.
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


using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;

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

void generate_ex_regression( size_t nb,  size_t no, double tau, Eigen::MatrixXd& B, Vector& b, Vector& y){
    
    std::cout << "generates random sample" << std::endl;

    /* ---------------------- construct random matrix of covariates --------------------- */
    
    // require different random seed here than in noise -> otherwise cancels each other out
    // val_l will then equal val_d ... 
    Vector B_random(no*(nb-1));
    rnorm_gen(no, 0.0, 1, &B_random, 2);

    Vector B_tmp(no*nb);
    B_tmp << Vector::Ones(no), B_random;
    std::cout << B_tmp << std::endl;

    // TODO: fix this!
    Eigen::Map<Eigen::MatrixXd> tmp(B_tmp.data(), no,nb);
    B = tmp;
    //*B(B_tmp.data());
    //Eigen::MatrixXd::Map(*B) = B_tmp.data(); 
    //std::cout << *B << std::endl;

    /* -------  construct random solution vector of fixed effects & observations -------- */
    b = 2*(Vector::Random(nb) + Vector::Ones(nb)); 

    double mean = 0.0;
    double sd = 1/sqrt(exp(tau));
    Vector noise_vec(no);

    rnorm_gen(no, mean, sd, &noise_vec, 4);

    y = B*b + noise_vec;

    std::cout << "b = " << b.transpose() << std::endl;

    /*std::cout << "noise vec " << std::endl;
    std::cout << noise_vec << std::endl; 

    std::cout << "B*b " << std::endl;
    std::cout << (*B)*(*b) << std::endl;

    std::cout << "y " << std::endl;
    std::cout << *y << std::endl; */

}