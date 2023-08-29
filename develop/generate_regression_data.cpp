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

typedef Eigen::VectorXd Vect;

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


void rnorm_gen(int no, double mean, double sd,  Vect& x, int seed){
  // unsigned int seed = 2;
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution (mean, sd);
 
  for (int i=0; i< x.size(); ++i){
        x[i]  = distribution(generator);
  }

}


void generate_ex_regression_constr(size_t nb, size_t no, double& tau, MatrixXd& D, Vect& e, MatrixXd& Prec, MatrixXd& B, Vect& b, Vect& y){


    if(e.sum() != 0){
      std::cout << "For now only SUM-TO-ZERO constraints possible!!" << std::endl;
      exit(1);
    }

    // generate covariance matrix
    MatrixXd M = MatrixXd::Random(nb,nb);
    Prec = M*M.transpose();

    LLT<MatrixXd> lltOfA(Prec); // compute the Cholesky decomposition of A
    MatrixXd L = lltOfA.matrixL();

    int seed = 4;
    Vect z(nb);
    rnorm_gen(nb, 0.0, 1.0, z, seed);
    //std::cout << "z = " << z.transpose() << std::endl;

    b = L.triangularView<Lower>().solve(z);
    //std::cout << "norm(L*b - z) = " << (L*b - z).norm() << std::endl;

    MatrixXd V = lltOfA.solve(D.transpose());
    MatrixXd W = D*V;
    MatrixXd U = W.inverse()*V.transpose();
    Vect c = D*b - e;
    b = b - U.transpose()*c;
    //std::cout << "D*b = " << D*b << ", b = " << b.transpose() << std::endl;
 
    B.resize(no, nb);
    B << Vect::Ones(no), MatrixXd::Random(no, nb-1);
    //std::cout << "B = \n" << B << std::endl;

    Vect noise_vec(no);
    double sd = 1/sqrt(exp(tau));
    // use different seeds!!
    rnorm_gen(no, 0.0, sd, noise_vec, seed+1); 

    // generates data y, using fixed effects b with b ~ N(0,Sigma*) where D*b = e, 
    // Sigma* constraint covariance matrix (see GMRF book p.38)
    y = B*b + noise_vec;


}

void generate_ex_regression( size_t nb,  size_t no, double& tau, Eigen::MatrixXd& B, Vect& b, Vect& y){
    
    std::cout << "generates random sample" << std::endl;

    /* ---------------------- construct random matrix of covariates --------------------- */
    
    // require different random seed here than in noise -> otherwise cancels each other out
    // val_l will then equal val_d ... 
    Vect B_random(no*(nb-1));
    rnorm_gen(no, 0.0, 1, B_random, 2);

    Vect B_tmp(no*nb);
    B_tmp << Vect::Ones(no), B_random;
    //std::cout << B_tmp << std::endl;

    // TODO: fix this!
    Eigen::Map<Eigen::MatrixXd> tmp(B_tmp.data(), no,nb);
    B = tmp;
    //*B(B_tmp.data());
    //Eigen::MatrixXd::Map(*B) = B_tmp.data(); 
    //std::cout << *B << std::endl;

    /* -------  construct random solution vector of fixed effects & observations -------- */
    b = 2*(Vect::Random(nb) + Vect::Ones(nb)); 

    double mean = 0.0;
    double sd = 1/sqrt(exp(tau));
    Vect noise_vec(no);

    rnorm_gen(no, mean, sd, noise_vec, 4);

    y = B*b + noise_vec;

    std::cout << "b                  : " << b.transpose() << std::endl;

    /*std::cout << "noise vec " << std::endl;
    std::cout << noise_vec << std::endl; 

    std::cout << "B*b " << std::endl;
    std::cout << (*B)*(*b) << std::endl;

    std::cout << "y " << std::endl;
    std::cout << *y << std::endl; */

}

// theta vector should be 3-dimensional : precision_observations, kappa, tau
void generate_ex_spatial_constr(size_t ns, size_t nb, size_t no, Vect& theta, MatrixXd& Qs, SpMat& Ax, MatrixXd& Ds, Vect& e, MatrixXd& Prec, MatrixXd& B, Vect& b, Vect& u, Vect& y){


    if(e.sum() != 0){
      std::cout << "For now only SUM-TO-ZERO constraints possible!!" << std::endl;
      exit(1);
    }

    // fixed effects remain UNCONSTRAINT
    // generate random covariance matrix for fixed effects 
    MatrixXd M = MatrixXd::Random(nb,nb);
    Prec = M*M.transpose();

    LLT<MatrixXd> lltOfA(Prec); // compute the Cholesky decomposition of A
    MatrixXd L_b = lltOfA.matrixL();

    int seed = 4;
    Vect z_b(nb);
    rnorm_gen(nb, 0.0, 1.0, z_b, seed);
    //std::cout << "z = " << z.transpose() << std::endl;

    b = L_b.triangularView<Lower>().solve(z_b);
    //std::cout << "norm(L*b - z) = " << (L*b - z).norm() << std::endl;

    // generate random covariates
    B.resize(no, nb);
    B << Vect::Ones(no), MatrixXd::Random(no, nb-1);
    //std::cout << "B = \n" << B << std::endl;

    // handle CONSTRAINED spatial part

    LLT<MatrixXd> lltOfQ(Qs); // compute the Cholesky decomposition of A
    MatrixXd L_s = lltOfQ.matrixL();

    Vect z_s(ns);
    rnorm_gen(nb, 0.0, 1.0, z_s, seed+1);
    //std::cout << "z = " << z.transpose() << std::endl;

    u = L_s.triangularView<Lower>().solve(z_s);

    MatrixXd V = lltOfQ.solve(Ds.transpose());
    MatrixXd W = Ds*V;
    MatrixXd U = W.inverse()*V.transpose();
    Vect c = Ds*u - e;
    u = u - U.transpose()*c;
    std::cout << "Ds*u = " << Ds*u << std::endl;
    // ", b = " << b.transpose()

    Vect noise_vec(no);
    double sd = 1/sqrt(exp(theta[0]));
    // use different seeds!!
    rnorm_gen(no, 0.0, sd, noise_vec, seed+2); 

    // generates data y, using fixed effects b with b ~ N(0,Sigma*) where D*b = e, 
    // Sigma* constraint covariance matrix (see GMRF book p.38)
    Vect x(ns+nb);
    x << u, b;
    y = Ax*x + noise_vec;

}


// theta vector should be 4-dimensional 
void generate_ex_spatial_temporal_constr(size_t ns, size_t nt, size_t nb, size_t no, Vect& theta, MatrixXd& Qst, SpMat& Ax, MatrixXd& Dst, Vect& e, MatrixXd& Prec, Vect& b, Vect& u, Vect& y){


    if(e.sum() != 0){
      std::cout << "For now only SUM-TO-ZERO constraints possible!!" << std::endl;
      exit(1);
    }

    // fixed effects remain UNCONSTRAINT
    // generate random covariance matrix for fixed effects 
    MatrixXd M = MatrixXd::Random(nb,nb);
    Prec = M*M.transpose();

    LLT<MatrixXd> lltOfA(Prec); // compute the Cholesky decomposition of A
    MatrixXd L_b = lltOfA.matrixL();

    int seed = 199;
    Vect z_b(nb);
    rnorm_gen(nb, 0.0, 1.0, z_b, seed);
    //std::cout << "z_b = " << z_b.transpose() << std::endl;

    b = L_b.triangularView<Lower>().solve(z_b);
    //std::cout << "b : " << b.transpose() << std::endl;
    //std::cout << "norm(L*b - z) = " << (L_b*b - z_b).norm() << std::endl;

    // handle CONSTRAINED spatial part

    LLT<MatrixXd> lltOfQ(Qst); // compute the Cholesky decomposition of A
    MatrixXd L_st = lltOfQ.matrixL();

    Vect z_st(ns*nt);
    rnorm_gen(nb, 0.0, 1.0, z_st, seed+1);
    //std::cout << "z = " << z_st.transpose() << std::endl;

    u = L_st.triangularView<Lower>().solve(z_st);

    MatrixXd V = lltOfQ.solve(Dst.transpose());
    MatrixXd W = Dst*V;
    MatrixXd U = W.inverse()*V.transpose();
    Vect c = Dst*u - e;
    u = u - U.transpose()*c;
    if((Dst*u).norm() > 1e-10){
      std::cout << "Dst*u = " << Dst*u << std::endl;
      std::cout << "This should be zero?!" << std::endl;
      exit(1);
    }
    // ", b = " << b.transpose()

    Vect noise_vec(no);
    double sd = 1/sqrt(exp(theta[0]));
    // use different seeds!!
    rnorm_gen(no, 0.0, sd, noise_vec, seed+2); 

    // generates data y, using fixed effects b with b ~ N(0,Sigma*) where D*b = e, 
    // Sigma* constraint covariance matrix (see GMRF book p.38)
    Vect x(ns*nt+nb);
    x << u, b;
    y = Ax*x + noise_vec;

}





