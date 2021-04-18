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

#include "theta_function.cpp"


using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;


using namespace LBFGSpp;


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


class Rosenbrock
{
private:
    int n;
public:
    Rosenbrock(int n_) : n(n_) {}
    double operator()(const VectorXd& x, VectorXd& grad)
    {
        double fx = 0.0;
        for(int i = 0; i < n; i += 2)
        {
            double t1 = 1.0 - x[i];
            double t2 = 10 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1] = 20 * t2;
            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        return fx;
    }
};

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


int main()
{

    int no = 5000;
    int nb = 3;

    double tau = 1.0;

    Eigen::MatrixXd B(no, nb);
    Vector b(nb);
    Vector y(no);

    generate_ex_regression(nb, no, tau, &B, &b, &y);
    std::cout << "original fixed effects : " << b.transpose() << std::endl;

    /*my_function fun(no, nb, B, y);

    VectorXd x = VectorXd::Zero(2);
    VectorXd grad = VectorXd::Zero(2);
    // const int n = 2;

    double fx = fun(x, grad);

    //std::cout << "fx : " << fx << std::endl;

    /*double yTy = y.dot(y);
    post_theta f_eval(no, nb, B, y, yTy);

    Vector theta(1);
    theta[0] = 2.0;
    Vector grad(1);

    double f_theta = f_eval(theta, grad);
    std::cout << "f theta : " << f_theta << std::endl;
    std::cout << "grad    : " << grad << std::endl;
    // std::cout << "mu      : " << mu.transpose() << std::endl;*/


    
    // Set up parameters
    /*BFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;

    // Create solver and function object
    LBFGSSolver<double> solver(param);

    // Initial guess
    // x will be overwritten to be the best point found
    //double fx;
    //int niter = solver.minimize(f_eval, theta, fx
    int niter = solver.minimize(fun, x, fx);


    std::cout << niter << " iterations" << std::endl;
    //std::cout << "theta = \n" << theta << std::endl;
    std::cout << "x = \n" << x << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    

    return 0;*/

    const int n = 10;
    // Set up parameters
    LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 10;

    // Create solver and function object
    LBFGSSolver<double> solver(param);
    post_theta fun(no, nb, B, y);

    // Initial guess
   Vector theta(1);
   theta[0] = 1.2;
   // Vector grad(1);    // x will be overwritten to be the best point found
    double fx;
    int niter = solver.minimize(fun, theta, fx);

    Vector mu = fun.get_mu();
    std::cout << "mu : " << mu.transpose() << std::endl;

    std::cout << niter << " iterations" << std::endl;
    std::cout << "theta = \n" << theta.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    Vector grad = fun.get_grad();
    std::cout << " grad = " << grad << std::endl;

    return 0;
}
