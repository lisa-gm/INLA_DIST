#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <functional> // to pass functions as arguments in other functions

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SparseExtra>   // includes saveMarket

#include <armadillo>
//#include "generate_testMat_st_s_field.cpp"
#include "../../read_write_functions.cpp"

#include "../../RGF/RGF.H"
//#include "../../develop/PardisoSolver.h"
#include "../../linSolverInterfaces/PardisoSolver.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SpRmMat;

using namespace std;

//#define PRINT_MSG

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif


// ************************************************************************************************ //
// functions for VB correction

/* what do I need? 
- quadrature points and weights (ideally a flexible setup)
- selected inverse of Q or full inverse
- beta mode
- neg log likelihood
- test data
- p1 & p2 -> functions to compute the derivatives ...

*/

// store quadrature weights manually for now
void gaussHermQuadrature(int m, Vect& nodes, Vect& weights){
    
    if(m == 5){
        nodes   << -2.856970e+00, -1.355626e+00,  2.410361e-16,  1.355626e+00,  2.856970e+00;
        weights << 0.01125741, 0.22207592, 0.53333333, 0.22207592, 0.01125741;
    } else if(m == 15){
        nodes << -6.363948, -5.190094, -4.196208, -3.289082, -2.432437, -1.606710, -7.991291e-01, -2.194426e-16,  
                7.991291e-01, 1.606710, 2.432437,  3.289082,  4.196208,  5.190094,  6.363948;
        weights << 8.589650e-10, 5.975420e-07, 5.642146e-05, 1.567358e-03, 1.736577e-02, 8.941780e-02, 2.324623e-01, 
                3.182595e-01, 2.324623e-01, 8.941780e-02, 1.736577e-02, 1.567358e-03, 5.642146e-05, 5.975420e-07, 8.589650e-10;
    } else {
      printf("invalid number of quadrature points!\n");
      exit(1);
    }
}

MatrixXd p1_mat(MatrixXd& Z, Vect& varObs){

    MatrixXd p1Z = Z.array().colwise() / varObs.array().sqrt();
    return p1Z;
}

MatrixXd p2_mat(MatrixXd& Z, Vect& varObs){

    MatrixXd p2Z = (Z.array().square() - 1).colwise() / varObs.array();
    return p2Z;
}

// ************************************************************************************************ //


void construct_Q_spatial(SpMat& Qs, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2){

    // Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
    // return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
    double exp_theta1 = exp(theta[0]);
    double exp_theta2 = exp(theta[1]);
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


void construct_Q_spat_temp(SpMat& Qst, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
                                      SpMat& M0, SpMat& M1, SpMat& M2){

    std::cout << "theta : " << theta.transpose() << std::endl;

    double exp_theta1 = exp(theta[0]);
    double exp_theta2 = exp(theta[1]);
    double exp_theta3 = exp(theta[2]);

    /*double exp_theta1 = exp(-5.594859);
    double exp_theta2 = exp(1.039721);
    double exp_theta3 = exp(3.688879);*/

    std::cout << "exp(theta) : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << " " << std::endl;   

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
        Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

        //std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
}


void construct_Qprior(SpMat& Qx, int ns, int nt, int nss, int nb, Vect theta, 
                      SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
                      SpMat& M0, SpMat& M1, SpMat& M2, \
                      SpMat& c0_s, SpMat& g1_s, SpMat& g2_s){

    int nst = ns*nt;
    int nu  = nst + nss;

    if(ns > 0){
        SpMat Qst(nst, nst);
        SpMat Qss(nss, nss);
        // TODO: find good way to assemble Qx

        if(nt > 1){
            construct_Q_spat_temp(Qst, theta, c0, g1, g2, g3, M0, M1, M2);
            if(nss > 0){
                // check if nss == c0_s.rows() ...
                if(nss != c0_s.rows()){
                    std::cout << "nss != c0_s.rows() ... " << std::endl;
                    exit(1);
                }
                construct_Q_spatial(Qss, theta(seq(3,4)), c0_s, g1_s, g2_s);
                std::cout << "Qss : \n" << Qss.block(0,0,10,10) << std::endl;
            }
        } else {    
            construct_Q_spatial(Qst, theta, c0, g1, g2);
        }   

        std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;

        //Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
        // construct Qx from Qs values, extend by zeros 
        size_t n = ns*nt + nss + nb;

        int nnz_st = Qst.nonZeros();
        Qx.reserve(nnz_st);

        for (int k=0; k<Qst.outerSize(); ++k)
          for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
          {
            Qx.insert(it.row(),it.col()) = it.value();                 
          }

        if(nss > 0){
            for (int k=0; k<Qss.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Qss,k); it; ++it)
                {
                    Qx.insert(it.row()+nst,it.col()+nst) = it.value();                 
                }

        }

        for(int i=nu; i<(n); i++){
            Qx.coeffRef(i,i) = 1e-3;
        }

        Qx.makeCompressed();

    #ifdef PRINT_MSG
        //std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
        //std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;
    #endif

    #ifdef PRINT_MSG
        std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
        std::cout << "theta : \n" << theta.transpose() << std::endl;
    #endif
    }

    /*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
    std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}

void construct_Q(SpMat& Q, int ns, int nt, int nss, int nb, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
                                      SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

    double exp_theta0 = exp(theta[0]);
    int nst = ns*nt;
    int nu  = nst + nss;

    SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
    /*std::cout << "Q_b " << std::endl;
    std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

    if(ns > 0){
        SpMat Qst(nst, nst);
        SpMat Qss(nss, nss);
        // TODO: find good way to assemble Qx

        if(nt > 1){
            construct_Q_spat_temp(Qst, theta(seq(1,3)), c0, g1, g2, g3, M0, M1, M2);
            if(nss > 0){
                construct_Q_spatial(Qss, theta(seq(4,5)), c0, g1, g2);
            }
        } else {    
            construct_Q_spatial(Qst, theta(seq(1,2)), c0, g1, g2);
        }   

        std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
        std::cout << "Qss : \n" << Qss.block(0,0,10,10) << std::endl;

        //Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
        // construct Qx from Qs values, extend by zeros 
        size_t n = ns*nt + nss + nb;
        SpMat Qx(n,n);         // default is column major           

        int nnz_st = Qst.nonZeros();
        Qx.reserve(nnz_st);

        for (int k=0; k<Qst.outerSize(); ++k)
          for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
          {
            Qx.insert(it.row(),it.col()) = it.value();                 
          }

        if(nss > 0){
            for (int k=0; k<Qss.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Qss,k); it; ++it)
                {
                    Qx.insert(it.row()+nst,it.col()+nst) = it.value();                 
                }

        }

        for(int i=nu; i<(n); i++){
            Qx.coeffRef(i,i) = 1e-3;
        }

        Qx.makeCompressed();

        #ifdef PRINT_MSG
            //std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
            //std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;
        #endif

        Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

        #ifdef PRINT_MSG
            std::cout << "exp(theta0) : " << exp_theta0 << std::endl;
            std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;

            std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
            //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
            std::cout << "theta : \n" << theta.transpose() << std::endl;

        #endif
    }

    /*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
    std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}


void compute_diagAQinvAT(SpMat& Qinv, SpRmMat& Ax_all, Vect& projMargVar){
       
    if(projMargVar.size() != Ax_all.rows()){
        printf("In compute marginals y. Dimensions don't match! dim(projMargVar) = %ld, nrows(Ax) = %ld\n", projMargVar.size(), Ax_all.rows());
        exit(1);
    }
    //SpRmMat Ax_all = Ax;
    SpRmMat temp = Ax_all;

    std::cout << "nnz(Ax_all) = " << Ax_all.nonZeros() << ", nnz(Qinv) = " << Qinv.nonZeros();
    std::cout << ", dim(Qinv) = " << Qinv.rows() << " " << Qinv.cols() << ", dim(Ax_all) = " << Ax_all.rows() << " " << Ax_all.cols() << std::endl;

    // write my own multiplication
    // for each row in Ax_all iterate through the columns of QinvSp -> only consider nonzero entries of Ax_all
    double t_firstMult = - omp_get_wtime();

    #pragma omp parallel 
    {

    #pragma omp parallel for default(shared)
    for (int k=0; k<temp.outerSize(); ++k){
        //printf("number of threads %d\n", omp_get_thread_num());
        for (SparseMatrix<double, RowMajor>::InnerIterator it(temp,k); it; ++it)
    {
            it.valueRef() = 0.0;
            for (SparseMatrix<double, RowMajor>::InnerIterator it_A(Ax_all,k); it_A; ++it_A)
            {     
                //printf("A(%ld, %ld) = %f\n", it_A.row(), it_A.col(), it_A.value());
                it.valueRef() += it_A.value()* Qinv.coeff(it_A.col(), it.col());
    }
                //temp2.coeffRef(it.row(), it.col()) = (Ax_all.row(it.row())).dot(QinvSp.col(it.col()));
}

    } // end outer for loop

    } // end parallel region

    t_firstMult += omp_get_wtime();
    //printf("time 1st Mult innerIter : %f\n", t_firstMult);

    //printf("norm(temp - temp2) = %f\n", (temp-temp2).norm());        
    double t_secondMult = - omp_get_wtime();
    for(int i=0; i<Ax_all.rows(); i++){
        projMargVar(i) = (temp.row(i)).dot(Ax_all.row(i));
    }
    t_secondMult += omp_get_wtime();
    //printf("time 2nd Mult: %f\n", t_secondMult);

    // find minimum and maximum value
    double minVal = projMargVar.minCoeff();
    printf("min(projMargVar) = %f\n", minVal);
    printf("max(projMargVar) = %f\n", projMargVar.maxCoeff());
}


/////////////////// Prior /////////////////
double cond_LogPriorLat(SpMat& Q, Vect& beta){
    Vect mu = Vect::Zero(beta.size());

    double f_val = -0.5 * (beta - mu).transpose() * Q * (beta - mu);
    return f_val;
}

/////////////////// Likelihoods /////////////////
// Poisson
double cond_LogPoisLik(Vect& eta, Vect& y){
    double f_val = eta.dot(y) - eta.array().exp().sum();
    return f_val;
}

// TODO: include scaling constant E for each eta -> will also be required in input ...
// 
double cond_negLogPoisLik(Vect& eta, Vect& y, Vect& E){
    // actually link function fixed here but to make input the same ...
    double f_val = eta.dot(y) - (E.array()*(eta.array().exp())).sum();
    return -1*f_val;
}

/*
diagHess_cond_negLpoisLik <- function(eta){
  if(is.null(E)){
    # assuming that E is all ones
    #f_val <- dot(y, drop(eta)) - sum(exp(eta))
    diagHess <- - exp(eta)
  } else {
    diagHess <- - E*exp(eta)
  }  
  return(-diagHess)
}
*/

void diagHess_cond_negLogPoisLik(Vect& eta, Vect& E, Vect& diagHess){
    diagHess = E.array() * eta.array().exp();
}


// takes matrix as input for vectorized evaluation of "multiple eta's"
// each column of Z corresponds to one input vector eta
void cond_negLogPoisLikMat(MatrixXd& Z, Vect& y, Vect& E, MatrixXd& negLogPoisZ){
    //double f_val = eta.dot(y) - (E.array()*(eta.array().exp())).sum();
    negLogPoisZ = Z.array().exp();

    // sum over columns
    negLogPoisZ = negLogPoisZ.array().colwise() * E.array();

    //Z = - (Z.transpose() * y).transpose() + Zexp;
    negLogPoisZ = -1 * (Z.array().colwise() * y.array()) + negLogPoisZ.array();
    //std::cout << "negLogPoisLik = \n" << negLogPoisZ.rowwise().sum().transpose() << std::endl;
}

double cond_LogPois(SpMat& Q, SpMat& A, Vect& E, Vect& beta, Vect& y){

    double f_val_prior = cond_LogPriorLat(Q, beta);

    Vect eta = A * beta;
    double f_val_neg_lik = cond_negLogPoisLik(eta, y, E);

    // times -1: want to minimize
    double f_val = -1 * f_val_prior + f_val_neg_lik;
    return f_val;
}

// LINK FUNCTIONS

void link_f_sigmoid(Vect& x, Vect& sigmoidX){
    //  1/(1 + e^-x)
        sigmoidX = 1.0 / (1.0 + (-1 * x).array().exp());
}

//void link_f_sigmoid(Vect& x, Vect& sigmoidX){
void link_f_sigmoid(MatrixXd& X, MatrixXd& SigmoidX){
    //  1/(1 + e^-x)
        SigmoidX = 1.0 / (1.0 + (-1 * X).array().exp());
}

// Binomial
double cond_negLogBinomLik(Vect& eta, Vect& y, Vect& ntrials){
    int m = eta.size();

    Vect linkEta(m);
    // hardcode sigmoid for now
    link_f_sigmoid(eta, linkEta);    
    Vect logLinkEta = linkEta.array().log();
    Vect tmpLinkEta = (Vect::Ones(m) - linkEta).array().log();

    double f_val = y.dot(logLinkEta) + (ntrials - y).dot(tmpLinkEta);
    return -f_val;
}

void cond_negLogBinomLikMat(MatrixXd& Z, Vect& y, Vect& ntrials, MatrixXd& negLogBinomZ){

    MatrixXd linkZ(Z.rows(), Z.cols());
    // hardcode sigmoid for now
    link_f_sigmoid(Z, linkZ);    

    /*
    std::cout << "shifted nodes: \n" << Z.block(0,0,10,Z.cols()) << std::endl;
    std::cout << "sum(sum(Z)) = " << Z.sum() << std::endl;
    std::cout << "linkZ: \n" << linkZ.block(0,0,10,Z.cols()) << std::endl;
    std::cout << "sum(sum(linkZ)) = " << linkZ.sum() << std::endl;
    std::cout << "sum(linkZ.col(0)) = " << linkZ.col(0).sum() << std::endl;
    */
    MatrixXd logLinkZ = linkZ.array().log();
    //std::cout << "sum(sum(logLinkZ)) = " << logLinkZ.sum() << std::endl;
    MatrixXd tmpLinkZ = (MatrixXd::Ones(Z.rows(), Z.cols()) - linkZ).array().log();

    negLogBinomZ = -1 * (logLinkZ.array().colwise() * y.array() + tmpLinkZ.array().colwise() * (ntrials - y).array());
    //std::cout << "sum(negLogBinomZ) = " << negLogBinomZ.sum() << std::endl;
}

double cond_logBinom(SpMat& Q, SpMat& A, Vect& ntrials, Vect& beta, Vect& y){
    double f_val_prior = cond_LogPriorLat(Q, beta);

    Vect eta = A * beta;
    double f_val_neg_lik = cond_negLogBinomLik(eta, y, ntrials);

    double f_val = -1 * f_val_prior + f_val_neg_lik;

    return f_val;
}

// general formulation for evaluating log conditional distribution
// pass likelihood & link function as an argument
// later fix these things in class constructor ...
double cond_logDist(SpMat &Q, SpMat& A, Vect& beta, Vect& y, 
                    function<double(Vect&, Vect&, Vect&)> lik_func){

    double f_val = 0;
    return f_val;
}


// naive FIRST ORDER CENTRAL DIFFERENCE (to be improved ...?)
// will simplify once we have this inside class ... only beta will be a variable ...
// expects as input
void FD_gradient(Vect& eta, Vect& y, Vect& E, Vect& grad, function<double(Vect&, Vect&, Vect&)> lik_func){
    int m = eta.size();
    double h = 1e-4;

    // probably better way to do this ...
    SpMat epsId(m,m);
    epsId.setIdentity();
    epsId = h * epsId;

    #pragma omp parallel for
    for(int i=0; i<m; i++){
        Vect eta_forward      = eta + epsId.col(i);
        double f_eta_forward  = lik_func(eta_forward, y, E);

        Vect eta_backward     = eta - epsId.col(i);
        double f_eta_backward = lik_func(eta_backward, y, E);

        grad(i) = (f_eta_forward - f_eta_backward) / (2*h);
    }
}

// naive SECOND ORDER DIFFERENCE: DIAGONAL of Hessian
// expects as input cond_LogPois (generalize ... )
void FD_diag_hessian(Vect& eta, Vect& y, Vect& E, Vect& diag_hess, function<double(Vect&, Vect&, Vect&)> lik_func){
    int m = eta.size();
    double h = 1e-5;

    // probably better way to do this ...
    SpMat epsId(m,m);
    epsId.setIdentity();
    epsId = h * epsId;

    double f_eta = lik_func(eta, y, E);
    printf("in FD_diag_hessian: f_eta = %f\n", f_eta);

    // 2nd order 3-point stencil
    /*#pragma omp parallel for
    for(int i=0; i<m; i++){
        Vect eta_forward    = eta + epsId.col(i);
        double f_eta_forward  = lik_func(eta_forward, y, E);

        Vect eta_backward     = eta - epsId.col(i);
        double f_eta_backward = lik_func(eta_backward, y, E);

        diag_hess(i) = (f_eta_forward - 2*f_eta + f_eta_backward) / (h*h);
    }*/

    // 2nd order 5-point stencil : (−𝑓(𝑥−2ℎ)+16𝑓(𝑥−ℎ)−30𝑓(𝑥)+16𝑓(𝑥−ℎ)−𝑓(𝑥+2ℎ)) / 12ℎ2
    
    /*
    #pragma omp parallel for
    for(int i=0; i<m; i++){
        Vect eta_forward_1      = eta + epsId.col(i);
        double f_eta_forward_1  = lik_func(eta_forward_1, y, E);

        Vect eta_forward_2      = eta + 2*epsId.col(i);
        double f_eta_forward_2  = lik_func(eta_forward_2, y, E);        

        Vect eta_backward_1     = eta - epsId.col(i);
        double f_eta_backward_1 = lik_func(eta_backward_1, y, E);

        Vect eta_backward_2     = eta - 2*epsId.col(i);
        double f_eta_backward_2 = lik_func(eta_backward_2, y, E);

        diag_hess(i) = (-f_eta_backward_2 + 16*f_eta_backward_1 - 30*f_eta + 16*f_eta_forward_1 - f_eta_forward_2) / (12*h*h);
    }
    */

   diagHess_cond_negLogPoisLik(eta, E, diag_hess);

}


// within class less inputs required
void NewtonIter(SpMat& Qprior, SpMat& A, Vect& E, Vect& y, Vect& beta, function<double(Vect&, Vect&, Vect&)> lik_func){

    int n = beta.size();
    int m = y.size();

    // prepare for iteration
    Vect beta_new = beta;
    Vect beta_old = Vect::Random(n);

    Vect eta(m);
    Vect gradLik(m);
    Vect diag_hess_eta(m);

    SpMat hess_eta(m,m);
    hess_eta.setIdentity();

    Vect beta_update(n);
    Vect beta_update2(n);

    // negative gradient: negFoD
    Vect negFoD(n);
    SpMat SoD(n,n);

    double t_FD_grad = 0;
    double t_FD_hess = 0;

    int MPI_rank   = 0;
    double log_det = 0;
    double t_chol  = 0;
    double t_solve = 0;
    PardisoSolver solverQ(MPI_rank); 

    // iteration
    int counter = 0;
    while((beta_new - beta_old).norm() > 1e-5){
        printf("\niter = %d, norm(beta_new - beta_old) = %f\n", counter, (beta_new - beta_old).norm());
        beta_old = beta_new;
        std::cout << "beta: " << beta_new.head(min((int) n, 10)).transpose() << std::endl;
        counter += 1;

        if(counter > 20){
            printf("max number of iterations reached in inner Iteration!\n");
            exit(1);
        }

        eta = A * beta_new;

        // compute gradient
        t_FD_grad = -omp_get_wtime();
        FD_gradient(eta, y, E, gradLik, lik_func);
        t_FD_grad += omp_get_wtime();

        // gradient of negative Log conditional  (minimization)
        negFoD = -1 * (Qprior * beta_new + A.transpose() * gradLik);

        // compute hessian
        t_FD_hess = -omp_get_wtime();
        FD_diag_hessian(eta, y, E, diag_hess_eta, lik_func);
        t_FD_hess += omp_get_wtime();
        hess_eta.diagonal() = diag_hess_eta;
        // hessian of negative log conditional (minimization)
        SpMat hess = Qprior + A.transpose() * hess_eta * A;
        //std::cout << "hess = \n" << hess.block(0,0,min(10,n),min(10,n)) << std::endl;

        // Newton step hess(x_k)*(x_k+1 - x_k) = - grad(x_k)
        solverQ.factorize_solve(hess, negFoD, beta_update, log_det, t_chol, t_solve);
        printf("t_FD_grad = %f, t_FD_hess = %f. PardisoSolver: log_det = %f, t_chol = %f, t_solve = %f\n", t_FD_grad, t_FD_hess, log_det, t_chol, t_solve);

        beta_new    = beta_update + beta_old;
    }

    beta = beta_new;

}

// within class less inputs required
void NewtonIter_VB(SpMat& Qprior, SpMat& Qxy, SpMat& A, SpMat& M, Vect& E, Vect& y, Vect& beta, function<void(MatrixXd&, Vect&, Vect&, MatrixXd&)> lik_func, Vect& varObs){

    int n = beta.size();
    int m = y.size();
    int p = M.cols();  // size of update

    // prepare for iteration
    Vect beta_new = beta;
    Vect beta_old = Vect::Random(n);

    // iterate over update delta -> size of p
    Vect delta_new = Vect::Zero(p);
    Vect delta_old = Vect::Random(p);
    Vect update(p);
    Vect update2(p);

    // Eigen solver for now
    /*
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Qxy);
    MatrixXd M_tilde2 = solverQ.solve(M);
    */

    int MPI_rank   = 0;
    double log_det = 0;
    double t_chol  = 0;
    double t_solve = 0;
    PardisoSolver solverQ_M(MPI_rank);
    PardisoSolver solverQQ(MPI_rank);

    //Eigen::Map<Vect> M_vec((MatrixXd(M)).data(), M.rows()*M.cols());
    Vect M_vec = MatrixXd(M).reshaped(M.rows()*M.cols(), 1);
    Vect M_tilde_vec(M_vec.size());

    solverQ_M.factorize_solve(Qxy, M_vec, M_tilde_vec, log_det, t_chol, t_solve);
    printf("PardisoSolver. Compute M_tilde: log_det = %f, t_chol = %f, t_solve = %f\n", log_det, t_chol, t_solve);

    Eigen::Map<Eigen::MatrixXd> M_tilde(M_tilde_vec.data(), M.rows(), M.cols());
    //printf("M_tilde - M_tilde2 = %f\n", (M_tilde - M_tilde2).norm());

    MatrixXd AM_tilde = A * M_tilde;

    Vect negFoD(n);
    MatrixXd SoD(n,n);
    SpMat SoD_sparse(n,n);

    int degree = 15;
    Vect quadNodes(degree);
    Vect quadWeights(degree);
    gaussHermQuadrature(degree, quadNodes, quadWeights);

    MatrixXd quadNodesMat   = quadNodes.replicate(1,m).transpose();
    MatrixXd quadWeightsMat = quadWeights.replicate(1,m).transpose();
    //std::cout << "quad Nodes : \n" << quadNodesMat << std::endl;

    Vect eta = A * beta;
    std::cout << "eta : " << eta.head(min((int) m, 10)).transpose() << std::endl;
    std::cout << "varObs : " << varObs.head(min((int) m, 10)).transpose() << std::endl;
    //std::cout << "eta replicate: \n" << eta.head(m).replicate(1,degree) << std::endl; 

    MatrixXd shiftedNodes(m, degree);
    shiftedNodes = quadNodesMat.array().colwise() * varObs.array().sqrt();
    shiftedNodes = shiftedNodes + eta.replicate(1,degree);
    //std::cout << "shifted Nodes: \n" << shiftedNodes << std::endl;

    MatrixXd sigmoidShiftedN = MatrixXd::Zero(m, degree);
    link_f_sigmoid(shiftedNodes, sigmoidShiftedN);
    //std::cout << "sigmoid(shiftedNodes) = \n" << sigmoidShiftedN << std::endl;

    MatrixXd negLogLikShiftedN = MatrixXd::Zero(m, degree);
    lik_func(shiftedNodes, y, E, negLogLikShiftedN);

    double quadRes = quadWeights.dot(negLogLikShiftedN.colwise().sum());
    printf("quad. Result : %f\n", quadRes);

    MatrixXd p1Z = p1_mat(quadNodesMat, varObs);
    //std::cout << "p1(quadNodes) = \n" << p1Z << std::endl;

    MatrixXd p2Z = p2_mat(quadNodesMat, varObs);
    //std::cout << "p2(quadNodesMat) = \n" << p2Z << std::endl;

    // 1st derivative: quadWeightsMat
    MatrixXd quadWp1Z = p1Z.array().rowwise() * quadWeights.transpose().array();
    Vect resGauss_1stDeriv = (negLogLikShiftedN.array() * quadWp1Z.array()).rowwise().sum();
    //std::cout << "resGauss_1stDeriv: \n" << resGauss_1stDeriv.transpose() << std::endl;
    
    // 2nd derivative
    MatrixXd quadWp2Z = p2Z.array().rowwise() * quadWeights.transpose().array();
    Vect resGauss_2ndDeriv = (negLogLikShiftedN.array() * quadWp2Z.array()).rowwise().sum();
    MatrixXd resGauss_2ndDeriv_mat = resGauss_2ndDeriv.asDiagonal();
    //std::cout << "resGauss_2ndDeriv: \n" << resGauss_2ndDeriv.transpose() << std::endl;

    //std::cout << "delta new = " << delta_new.transpose() << std::endl;
    negFoD = -1 * (M_tilde.transpose() * Qxy * (M_tilde * delta_new + beta) + AM_tilde.transpose() * resGauss_1stDeriv);
    //std::cout << "FoD : " << FoD.transpose() << std::endl;

    SoD = M_tilde.transpose() * Qxy * M_tilde + AM_tilde.transpose() * resGauss_2ndDeriv.asDiagonal() * AM_tilde;
    //std::cout << "SoD : \n" << SoD<< std::endl;

    MatrixXd newShiftedNodes(shiftedNodes.rows(), shiftedNodes.cols());

    // A*D vs D*A with D being diagonal

    // iteration
    int counter = 0;
    // make stopping criterion based on relative change: (delta_new / sd) < tol
    // while()
    while((M_tilde * (delta_new - delta_old)).norm() > 1e-6){
        printf("\niter: %d, norm(M_tilde*(delta_new - delta_old)) = %f\n", counter, (M_tilde * (delta_new - delta_old)).norm());
        delta_old = delta_new;
        counter += 1;

        if(counter > 20){
            printf("max number of iterations reached in inner Iteration!\n");
            exit(1);
        }

        newShiftedNodes = shiftedNodes + (AM_tilde*delta_new).replicate(1,degree);
        //std::cout << "shifted Nodes: \n" << shiftedNodes << std::endl;
  
        lik_func(newShiftedNodes, y, E, negLogLikShiftedN);
        double quadRes = quadWeights.dot(negLogLikShiftedN.colwise().sum());
        printf("quad. Result : %f\n", quadRes);

        // 1st derivative
        resGauss_1stDeriv = (negLogLikShiftedN.array() * quadWp1Z.array()).rowwise().sum();
        //std::cout << "resGauss_1stDeriv: \n" << resGauss_1stDeriv.transpose() << std::endl;
        
        // 2nd derivative
        double t_mult = -omp_get_wtime();
        resGauss_2ndDeriv = (negLogLikShiftedN.array() * quadWp2Z.array()).rowwise().sum();
        t_mult += omp_get_wtime();
        printf("t compute resGauss_2ndDeriv: %f\n", t_mult);
        //std::cout << "resGauss_2ndDeriv: \n" << resGauss_2ndDeriv.transpose() << std::endl;        
        
        t_mult = -omp_get_wtime();
        negFoD = -1 * (M_tilde.transpose() * Qprior * (M_tilde * delta_new + beta) + AM_tilde.transpose() * resGauss_1stDeriv);
        t_mult += omp_get_wtime();
        printf("t compute gradient: %f\n", t_mult);
        
        t_mult = -omp_get_wtime();
        // TODO: can I optimize this? second multiplication takes a long time ...
        SoD    = M_tilde.transpose() * Qprior * M_tilde + AM_tilde.transpose() * resGauss_2ndDeriv.asDiagonal() * AM_tilde;
        t_mult += omp_get_wtime();        
        printf("t compute Hessian: %f\n", t_mult);

        /*
        Eigen::LDLT<MatrixXd> solverHess(SoD);
        if(solverHess.info()!=Success) {
            cout << "Oh: Very bad. Hessian not pos. definite." << endl;
            exit(1);
        }
        update2    = solverHess.solve(negFoD);
        */

        // Newton step hess(x_k)*(x_k+1 - x_k) = - grad(x_k)
        // SoD is dense
        SoD_sparse = SoD.sparseView();
        solverQQ.factorize_solve(SoD_sparse, negFoD, update, log_det, t_chol, t_solve);
        printf("PardisoSolver. Compute M_tilde: log_det = %f, t_chol = %f, t_solve = %f\n", log_det, t_chol, t_solve);
        //printf("norm(update - update2) = %f\n", (update - update2).norm());

        delta_new = delta_new + update;
    }

    beta = beta + M_tilde * delta_new;
}

/* ===================================================================== */

int main(int argc, char* argv[])
{

size_t i; // iteration variable

#if 0
    // generate dummy test case for Poisson distributed data
    /*
    int ns=3;
    int nt=3;
    int nss=1;
    int nb=2;
    int n = ns*nt + nss + nb;
    */

    //SpMat Q = gen_test_mat_base4(ns, nt, nss, nb);
    //SpMat Q = gen_test_mat_base4_prior(ns, nt, nss);
    //std::cout << "Q : \n" << MatrixXd(Q) << std::endl;

    if(argc != 1 + 4){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "[integer: n]                number of latent variables " << std::endl;
        std::cerr << "[integer: m]                number of observations " << std::endl;
        std::cout << "[string:likelihood]         assumed distribution data " << std::endl;
        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        exit(1);
    }

    size_t n               = atoi(argv[1]);
    size_t no              = atoi(argv[2]);
    std::string likelihood = argv[3];

    if(likelihood.compare("Poisson") == 0 || likelihood.compare("poisson") == 0){
        likelihood     = "Poisson";
    } else if(likelihood.compare("Binomial") == 0 || likelihood.compare("binomial") == 0) {
        likelihood = "Binomial";
    } else {
        std::cout << "unknown likelihood: " << likelihood << std::endl;
        exit(1);
    }

    std::string base_path = argv[4];    
    std::cout << "n = " << n << ", m = " << m << ", likelihood: " << likelihood << ", base path: " << base_path << std::endl;

    std::string A_file        =  base_path + "/Ax_" + to_string(m) + "_" + to_string(n) + ".dat";
    file_exists(A_file); 
    //MatrixXd A_dense = read_matrix(A_file, m, n);
    //SpMat A = A_dense.sparseView();
    SpMat Ax = readCSC(A_file);
    //std::cout << "A: \n" << MatrixXd(A) << std::endl;
    //std::cout << "A = \n" << A << std::endl;
    std::cout << "A(1:10,1:n) = \n" << A.block(0,0,10,n) << std::endl;

    std::string y_file        =  base_path + "/y_" + to_string(no) + "_1" + ".dat";
    file_exists(y_file);
    Vect y = read_matrix(y_file, m, 1);
    //std::cout << "y: " << y.transpose() << std::endl;

    //Vect E = Vect::Ones(m);
    std::string extraCoeff_file        =  base_path + "/extraCoeff_" + to_string(no) + "_1" + ".dat";
    file_exists(extraCoeff_file);
    Vect E = read_matrix(extraCoeff_file, no, 1);
    std::cout << "E: " << E.head(min((int) n, 10)).transpose() << std::endl;

    /*std::string beta_file        =  base_path + "/beta_original_" + to_string(n) + "_1" + ".dat";
    file_exists(beta_file);
    Vect beta_original = read_matrix(beta_file, n, 1);
    std::cout << "beta original: " << beta_original.transpose() << std::endl;*/

    //std::string xMode_INLA_file        =  base_path + "/xMode_INLA_" + to_string(n) + "_1" + ".dat";
    std::string xMode_INLA_file        =  base_path + "/mean_latent_noVB_INLA_" + to_string(n) + "_1" + ".dat";
    file_exists(xMode_INLA_file);
    Vect xMode_INLA = read_matrix(xMode_INLA_file, n, 1);
    std::cout << "x Mode INLA: " << xMode_INLA.head(min((int) n, 10)).transpose() << std::endl;

    // std::string x_VB_INLA_file        =  base_path + "/x_VB_INLA_" + to_string(n) + "_1" + ".dat";
    std::string x_VB_INLA_file        =  base_path + "/mean_latent_VB_INLA_" + to_string(n) + "_1" + ".dat";
    file_exists(x_VB_INLA_file);
    Vect x_VB_INLA = read_matrix(x_VB_INLA_file, n, 1);
    std::cout << "x VB INLA: " << x_VB_INLA.head(min((int) n, 10)).transpose() << std::endl;

#if 0
    // number of regression coefficients
    int n = 3;
    // number of data samples/observations
    int m = 10;

    // regression coefficients beta
    Vect beta = 2*Vect::Random(n);

    // matrix with covariates A
    MatrixXd A_dense = 0.5*MatrixXd::Random(m,n);
    A_dense.col(0)   = Vect::Ones(m); // set first column to zero for offset
    SpMat A = A_dense.sparseView();

    // linear predictor mu
    Vect muTmp = A*beta;
    Vect mu = muTmp.array().exp();
    std::cout << "mu: " << mu.transpose() << std::endl;

    Vect y(m);

    std::default_random_engine generator;

    for (int i=0; i<m; ++i) {
        std::poisson_distribution<int> distribution(mu[i]);
        y(i) = distribution(generator);
    }

    std::cout << "y: " << y.transpose() << std::endl;
#endif

    Vect rhs(n);
    rhs.setOnes(n);

#else

    if(argc != 1 + 8){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points add. spatial field " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:likelihood]         Gaussian/Poisson/Binomial" << std::endl;
        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;
        std::cerr << "[string:solver_type]        BTA or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns  = atoi(argv[1]);
    size_t nt  = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb  = atoi(argv[4]);
    size_t no  = atoi(argv[5]);

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    int n = ns*nt + nss + nb;
    std::cout << "ns = " << ns << ", nt = " << nt << ", nb = " << nb << ", n = " << n << std::endl;
    
    //std::string no_s = argv[5];
    // to be filled later

    // also save as string
    std::string ns_s  = std::to_string(ns);
    std::string nt_s  = std::to_string(nt);
    std::string nss_s = std::to_string(nss);
    std::string nb_s  = std::to_string(nb);
    std::string no_s  = std::to_string(no); 
    std::string n_s   = std::to_string(ns*nt + nss + nb);

    std::string likelihood = argv[6];

    std::function<double(SpMat&, SpMat&, Vect&, Vect&, Vect&)> cond_LogLik;
    std::function<double(Vect&, Vect&, Vect&)> cond_negLogLik;
    std::function<void(MatrixXd&, Vect&, Vect&, MatrixXd&)> cond_negLogLikMat;

    if(likelihood.compare("Poisson") == 0 || likelihood.compare("poisson") == 0){
        likelihood        = "Poisson";
        cond_LogLik       = cond_LogPois;
        cond_negLogLik    = cond_negLogPoisLik;
        cond_negLogLikMat = cond_negLogPoisLikMat;

    } else if(likelihood.compare("Binomial") == 0 || likelihood.compare("binomial") == 0) {
        likelihood        = "Binomial";
        cond_LogLik       = cond_logBinom;
        cond_negLogLik    = cond_negLogBinomLik;
        cond_negLogLikMat = cond_negLogBinomLikMat;

    } else {
        std::cout << "unknown likelihood: " << likelihood << std::endl;
        exit(1);
    }

    std::string base_path = argv[7];    

    std::string solver_type = argv[8];
    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("BTA") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nBTA" << std::endl;
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

    // add. spatial field
    SpMat c0_s;
    SpMat g1_s;
    SpMat g2_s;

    // data component / fixed effects
    MatrixXd B;
    SpMat Ax; 
    Vect y;

    if(ns == 0 && nt == 0){

        // no point to run this with Gaussian likelihood
        dim_th = 0;

        // read in design matrix 
        // files containing B
        std::string Ax_file        =  base_path + "/Ax_" + no_s + "_" + nb_s + ".dat";
        file_exists(Ax_file); 
        Ax = readCSC(Ax_file);

        // casting no_s as integer
        no = std::stoi(no_s);
        std::cout << "total number of observations : " << no << std::endl;
    
    } else if(ns > 0 && nt == 1){

        std::cout << "spatial model." << std::endl;
        dim_th = 2;

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

        std::cout << "spatial-temporal model. Reading in matrices." << std::endl;

        if(nss == 0){
            dim_th = 3;
        } else if(nss > 0){
            dim_th = 5;
        } else {
            printf("nss invalid!! nss = %ld\n", nss);
            exit(1);
        }

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

        if(nss > 0 && nss != ns){
            std::string c0_s_file      =  base_path + "/c0_s_" + nss_s + ".dat";
            file_exists(c0_s_file);
            std::string g1_s_file      =  base_path + "/g1_s_" + nss_s + ".dat";
            file_exists(g1_s_file);
            std::string g2_s_file      =  base_path + "/g2_s_" + nss_s + ".dat";
            file_exists(g2_s_file);

            c0_s = read_sym_CSC(c0_s_file);
            g1_s = read_sym_CSC(g1_s_file);
            g2_s = read_sym_CSC(g2_s_file);
        }

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

    Vect theta(dim_th);
    Vect theta_prior(dim_th);

    if(nt == 1){
        //theta << -5,2.64;
        std::string theta_file = base_path + "/theta_INLA_modelScale_" + to_string(dim_th) + "_1.dat";
        file_exists(theta_file);
        theta = read_matrix(theta_file, dim_th, 1);
        std::cout << "theta : " << theta.transpose() << std::endl;
        //theta.print();
    } else if(nt > 1){
        // prec noise, theta_st_1, theta_st_2, theta_st_3, theta_s_1, theta_s_2
        //theta << -9.682992,  3.342306,  6.684612;
        //theta << 4.000000, -3.344954,  1.039721,  1.386294, 3.0, 2.0; // equals 4,0,0,0 in param scale        
        //theta << 1.3539441, -4.4696240,  0.6342557,  1.6739764,  -4.6078180, 2.2436936;
        // std::string theta_file = base_path + "/theta_" + ".dat";
        std::string theta_file = base_path + "/theta_INLA_modelScale_" + to_string(dim_th) + "_1.dat";
        file_exists(theta_file);
        theta = read_matrix(theta_file, dim_th, 1);
        std::cout << "theta : " << theta.transpose() << std::endl;
        //theta = {3, -5, 1, 2};
        //theta.print();
    }

#endif

#if 1

    //Vect E = Vect::Ones(m);
    std::string extraCoeff_file        =  base_path + "/extraCoeff_" + to_string(no) + "_1" + ".dat";
    file_exists(extraCoeff_file);
    Vect E = read_matrix(extraCoeff_file, no, 1);
    std::cout << "E: " << E.head(min((int) n, 10)).transpose() << std::endl;

    // std::string xMode_INLA_file        =  base_path + "/xMode_INLA_" + to_string(n) + "_1" + ".dat";
    std::string xMode_INLA_file        =  base_path + "/mean_latent_INLA_noVB_" + to_string(n) + "_1" + ".dat";
    file_exists(xMode_INLA_file);
    Vect xMode_INLA = read_matrix(xMode_INLA_file, n, 1);
    std::cout << "x Mode INLA: " << xMode_INLA.head(min((int) n, 10)).transpose() << std::endl;

    //std::string x_VB_INLA_file        =  base_path + "/x_VB_INLA_" + to_string(n) + "_1" + ".dat";
    std::string x_VB_INLA_file         =  base_path + "/mean_latent_INLA_VB_" + to_string(n) + "_1" + ".dat";
    file_exists(x_VB_INLA_file);
    Vect x_VB_INLA = read_matrix(x_VB_INLA_file, n, 1);
    std::cout << "x VB INLA: " << x_VB_INLA.head(min((int) n, 10)).transpose() << std::endl;

    Vect sigmoidBeta(n);
    //link_f_sigmoid(beta_original, sigmoidBeta);
    link_f_sigmoid(xMode_INLA, sigmoidBeta);
    std::cout << "sigmoid(beta) = " << sigmoidBeta.head(min((int) n, 10)).transpose() << std::endl;

    // define prior Q
    SpMat Qprior(n,n);

    if(ns == 0 && nt == 0){
        Qprior.setIdentity();
        Qprior = 0.001 * Qprior;
    } else {
        construct_Qprior(Qprior, ns, nt, nss, nb, theta, c0, g1, g2, g3, M0, M1, M2, c0_s, g1_s, g2_s);
    }
    
    //double f_val_prior = cond_LogPriorLat(Qprior, beta_original);
    double f_val_prior = cond_LogPriorLat(Qprior, xMode_INLA);
    printf("val LogPriorLat : %f\n", f_val_prior);

    std::cout << "dim(Ax) = " << Ax.rows() << " " << Ax.cols() << ", dim(beta original) = " << xMode_INLA.rows() << std::endl;
    //Vect eta = A * beta_original;
    Vect eta = Ax * xMode_INLA;
    std::cout << "eta(1:10) : " << eta.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "y(1:10)   : " << y.head(min((int) n, 10)).transpose() << std::endl;

    std::cout << "likelihood : " << likelihood << ", likelihood.compare(Poisson) = " << likelihood.compare("Poisson") << std::endl;

    // *************************************************************************** //

    // construct Qprior, Qxy for comparison
    std::string Qprior_file = base_path + "/Qprior_thetaINLA_" + to_string(n) + ".dat";
    file_exists(Qprior_file);
    SpMat Qprior_INLA = read_sym_CSC(Qprior_file);

    printf("norm(Qprior - Qprior_INLA) = %f\n", (Qprior - Qprior_INLA).norm());

    std::string Q_file = base_path + "/Qxy_thetaINLA_" + to_string(n) + ".dat";
    file_exists(Q_file);
    SpMat Qxy_INLA = read_sym_CSC(Q_file);

    Vect diagHessEta(no);
    eta = Ax * xMode_INLA;
    FD_diag_hessian(eta, y, E, diagHessEta, cond_negLogLik);
    //cout << "diag hessian: " << diag_hess_eta.transpose() << std::endl;
    
    std::cout << "eta[1:10] = " << eta.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "E[1:10] = " << E.head(min((int) n, 10)).transpose() << std::endl;

    Vect diagHessAnalytic = E.array() * eta.array().exp();
    std::cout << "diagHessAnalytic[1:10] = " << diagHessAnalytic.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "diagHessEta[1:10] = " << diagHessEta.head(min((int) n, 10)).transpose() << std::endl;
    SpMat Qxy_test = Qprior + Ax.transpose() * diagHessAnalytic.asDiagonal() * Ax;

    SpMat Qxy = Qprior + Ax.transpose() * diagHessEta.asDiagonal() * Ax;
    printf("norm(Qxy - Qxy_INLA) = %f\n", (Qxy - Qxy_INLA).norm());

    printf("norm(Qxy_INLA - Qxy_test) = %f\n", (Qxy_INLA - Qxy_test).norm());
    printf("norm(Qxy - Qxy_test) = %f\n", (Qxy - Qxy_test).norm());


    std::cout << "Qxy_INLA : \n" << MatrixXd(Qxy_INLA.block(0,0,10,10)) << std::endl;
    std::cout << "Qxy      : \n" << MatrixXd(Qxy.block(0,0,10,10)) << std::endl;

    // print last 10x10 submatrix of Qxy_INLA & Qxy
    std::cout << "Qxy_INLA : \n" << MatrixXd(Qxy_INLA.block(n-10,n-10,10,10)) << std::endl;
    std::cout << "Qxy      : \n" << MatrixXd(Qxy.block(n-10,n-10,10,10)) << std::endl;

    // Find indices of maximum differences
    Eigen::MatrixXd diff = (Qxy - Qxy_INLA).cwiseAbs();
    Eigen::Index maxRow, maxCol;
    double maxDiff = diff.maxCoeff(&maxRow, &maxCol);
    printf("maxDiff : %f, Indices of maximum difference: (%ld, %ld)\n", maxDiff, maxRow, maxCol); 


    std::cout << "Qxy - Qxy_INLA : \n" << MatrixXd(Qxy - Qxy_INLA).block(n-10,n-10,10,10) << std::endl;
    std::cout << "Qxy_test - Qxy_INLA : \n" << MatrixXd(Qxy_test - Qxy_INLA).block(n-15,n-15,15,15) << std::endl;

    // *************************************************************************** //

    //Vect beta = beta_original + 0.5*Vect::Random(n);
    Vect beta = xMode_INLA + 0.1*Vect::Random(n);

    double f_val_lik = cond_negLogLik(eta, y, E);
    printf("val negLogLik : %f\n", f_val_lik);
    //double f_val = cond_LogPois(Qprior, A, E, beta_original, y);
    double f_val = cond_LogLik(Qprior, Ax, E, xMode_INLA, y);
    printf("val Log : %f\n", f_val);
    
    Vect gradLik(no);
    FD_gradient(eta, y, E, gradLik, cond_negLogLik);
    //Vect grad = Qprior * beta_original + A.transpose() * gradLik;
    Vect grad = Qprior * xMode_INLA + Ax.transpose() * gradLik;
    //cout << "gradient: " << grad.transpose() << std::endl;

    //Vect eta = A * beta_original;
    Vect diag_hess_eta(no);
    FD_diag_hessian(eta, y, E, diag_hess_eta, cond_negLogLik);
    // cout << "diag hessian: " << diag_hess_eta.transpose() << std::endl;
    
    SpMat hess_eta(no,no);
    hess_eta.setIdentity();
    hess_eta.diagonal() = diag_hess_eta;

    SpMat hess = Qprior + Ax.transpose() * hess_eta * Ax;
    //std::cout << "Hessian: \n" << MatrixXd(hess.block(0,0,5,5)) << std::endl;
    //cout << "Hessian: \n" << MatrixXd(hess) << std::endl;

    NewtonIter(Qprior, Ax, E, y, beta, cond_negLogLik);
    std::cout << "final x estimate: " << beta.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "xMode INLA      : " << xMode_INLA.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "norm(x - x_INLA): " << (beta - xMode_INLA).norm() << std::endl;

    exit(1);

   // ------------------------------ VB correction ---------------------------- // 
    printf("\nstart VB correction.\n");

    // initialize quadrature points
    int degree = 15;
    std::cout << "number of used quadrature points: " << degree << std::endl;
    Vect quadNodes(degree);
    Vect quadWeights(degree);
    gaussHermQuadrature(degree, quadNodes, quadWeights);
    // std::cout << "quadNodes:  " << quadNodes.transpose() << std::endl;
    // std::cout << "quadWeights: " << quadWeights.transpose() << std::endl;

    // compute Q at the mode
    eta = Ax * xMode_INLA;
    FD_diag_hessian(eta, y, E, diag_hess_eta, cond_negLogLik);

    //FD_diag_hessian(eta, y, E, diag_hess_eta, &cond_negLogBinomLik);
    //cout << "diag hessian: " << diag_hess_eta.transpose() << std::endl;

    hess_eta.setIdentity();
    hess_eta.diagonal() = diag_hess_eta;

    SpMat Q = Qprior + Ax.transpose() * hess_eta * Ax;
    // cout << "Hessian: \n" << MatrixXd(Q.block(0,0,min((int) n,10),min((int) n,10))) << std::endl;

    double t_cov;
    
    t_cov = -omp_get_wtime();
    // compute covariance matrix (full/selected inverse)
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Qxy_INLA);
    MatrixXd Cov2 = solverQ.solve(MatrixXd::Identity(n,n));

   if(solverQ.info()!=Success) {
        cout << "Oh: Very bad. Hessian not pos. definite." << endl;
        exit(1);
    }

    //cout << "Cov: \n" << Cov << std::endl;
    t_cov += omp_get_wtime();



    printf("time compute Covariance: %f\n", t_cov);
    

    int MPI_rank = 0;
    MatrixXd Cov(n,n);
    MatrixXd Q_dense = MatrixXd(Qxy_INLA);

    t_cov = -omp_get_wtime();
    PardisoSolver solverQQ(MPI_rank);
    //solverQQ.compute_inverse_pardiso(Q_dense, Cov);
    solverQQ.compute_full_inverse(Qxy_INLA, Cov);
    t_cov += omp_get_wtime();
    printf("PardisoSolver. Compute Cov: %f\n", t_cov);

    printf("min(diagonal(Cov)) = %f\n", Cov.diagonal().minCoeff());
    printf("max(diagonal(Cov)) = %f\n", Cov.diagonal().maxCoeff());

    printf("min(diagonal(Cov)) = %f\n", Cov2.diagonal().minCoeff());

    printf("norm(Cov - Cov2) = %f\n", (Cov - Cov2).norm());

    printf("norm(I - Q_dense*Cov) = %f\n", (MatrixXd::Identity(n,n) - Q_dense*Cov).norm());
    printf("norm(I - Q_dense*Cov2) = %f\n", (MatrixXd::Identity(n,n) - Q_dense*Cov2).norm());
    exit(1);
    /*
    t_cov = -omp_get_wtime();
    Vect varObs2 = (Ax * Cov * Ax.transpose()).diagonal();
    t_cov += omp_get_wtime();
    printf("time compute Ax Cov Ax^T: %f\n", t_cov);
    */

    SpMat Cov_sparse = Cov2.sparseView();
    Vect varObs(no);    
    SpRmMat Ax_rm = Ax;
    t_cov = -omp_get_wtime();
    compute_diagAQinvAT(Cov_sparse, Ax_rm, varObs);
    t_cov += omp_get_wtime();
    printf("time diagAQinvAt(): %f\n", t_cov);
    //printf("norm(varObs - varObs2) = %f\n", (varObs - varObs2).norm());

    Vect sdObs = varObs.cwiseSqrt();

    // TODO: this will have to be adapted for large cases
    int temp_m = Ax.rows();
    std::cout << "sdObs : " << sdObs.head(min((int) n,10)).transpose() << std::endl;

    MatrixXd quadNodesMat   = quadNodes.replicate(1,temp_m).transpose();
    MatrixXd quadWeightsMat = quadWeights.replicate(1,temp_m).transpose();
    //std::cout << "quad Nodes : \n" << quadNodesMat.block(0,0,10,quadNodesMat.cols()) << std::endl;

    eta = Ax * beta;
    //std::cout << "eta replicate: \n" << eta.head(temp_m).replicate(1,degree) << std::endl; 

    MatrixXd shiftedNodes(temp_m, degree);
    shiftedNodes = quadNodesMat.array().colwise() * sdObs.head(temp_m).array();
    //std::cout << "shifted Nodes sd scaling: \n" << shiftedNodes << std::endl;

    shiftedNodes = shiftedNodes + eta.head(temp_m).replicate(1,degree);
    //std::cout << "shifted Nodes: \n" << shiftedNodes << std::endl;
    MatrixXd negLogLikShiftedN = MatrixXd::Zero(temp_m, degree);

    // approximate integral neg. log likelihood for comparison
    Vect y_short = y.head(temp_m);
    Vect E_short = E.head(temp_m);
    Vect eta_short = eta.head(temp_m);
    SpMat A_short =  Ax.block(0,0,temp_m,n);
    Vect varObs_short = varObs.head(temp_m);

    cond_negLogLikMat(shiftedNodes, y_short, E_short, negLogLikShiftedN);
    //std::cout << "A(1:temp_m,:) = \n" << MatrixXd(A_short) << std::endl;

    double quadRes = quadWeights.dot(negLogLikShiftedN.colwise().sum());
    printf("quad. Result : %f\n", quadRes);

    MatrixXd negLogLikZ(shiftedNodes.rows(), shiftedNodes.cols());
    cond_negLogLikMat(quadNodesMat, y, E, negLogLikZ);
    //std::cout << "negLogBinomZ : \n" << negLogBinomZ.block(0,0,10,degree) << std::endl;

    MatrixXd p1Z = p1_mat(quadNodesMat, varObs_short);
    //std::cout << "p1(quadNodes) = \n" << p1Z << std::endl;

    MatrixXd p2Z = p2_mat(quadNodesMat, varObs_short);
    //std::cout << "p2(quadNodesMat) = \n" << p2Z << std::endl;

    // 1st derivative: quadWeightsMat
    MatrixXd quadWp1Z = p1Z.array().rowwise() * quadWeights.transpose().array();
    Vect resGauss_1stDeriv = (negLogLikShiftedN.array() * quadWp1Z.array()).rowwise().sum();
    //std::cout << "resGauss_1stDeriv: \n" << resGauss_1stDeriv.transpose() << std::endl;
    
    // 2nd derivative
    MatrixXd quadWp2Z = p2Z.array().rowwise() * quadWeights.transpose().array();
    Vect resGauss_2ndDeriv = (negLogLikShiftedN.array() * quadWp2Z.array()).rowwise().sum();
    //std::cout << "resGauss_2ndDeriv: \n" << resGauss_2ndDeriv.transpose() << std::endl;

    // replace this by particular entries that we correct for
    bool read_index_list = true;
    SpMat M;

    if(read_index_list){
        std::cout << "read index list ... " << std::endl;
        std::string index_list_file = base_path + "/M_indices_40_1.dat";
        file_exists(index_list_file);
        Vect index_list = read_matrix(index_list_file, 37, 1);
        int p = index_list.size();
        M.resize(n, p);
        M.reserve(p);
        for(int i=0; i<p; i++){
            M.insert(index_list(i), i) = 1;
        }
    } else {
        std::cout << "setting M to the identity matrix ..." << std::endl;
        M.resize(n, n);
        M.setIdentity();
    }
    //std::cout << "M : \n" << M.block(0,0,10,10) << std::endl;

    // Newton iteration
    std::cout << "beta : " << beta.head(min((int) n, 10)).transpose() << std::endl;
    NewtonIter_VB(Qprior, Q, A_short, M, E_short, y_short, beta, cond_negLogLikMat, varObs_short);

    std::cout << "final beta + M_tilde*delta_new : " << beta.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "x VB INLA                      : " << x_VB_INLA.head(min((int) n, 10)).transpose() << std::endl;
    std::cout << "norm(x - x_VB_INLA)            : " << (beta - x_VB_INLA).norm() << std::endl;


#endif
    return 0;

  }


