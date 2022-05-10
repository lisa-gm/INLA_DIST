#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;

using namespace LBFGSpp;

class Rosenbrock
{
private:
    int n;
    int counter;

    double fx_min;

    VectorXd x_prev;
    MatrixXd X_diff;
    bool Xdiff_initialized;

public:
    Rosenbrock(int n_) : n(n_) {

        Xdiff_initialized = false;
        counter = 1;

        fx_min = 10e10;

    }
    double operator()(const VectorXd& x, VectorXd& grad)
    {   

        //std::cout << "iter = " << counter << std::endl;
        //counter += 1;

        double fx = f_eval(x);

        if(fx < fx_min){
            std::cout << "x : " << std::right << std::fixed << x.transpose() << ",    fx : " << std::right << std::fixed << fx << std::endl;
            fx_min = fx;
        }
        //grad_exact(x, grad);
        //std::cout << "analytic gradient = " << grad.transpose() << std::endl;

        //Eigen::VectorXd grad = Eigen::VectorXd::Zero(grad.size());
        //grad_standard_FD(x, grad);
        //std::cout << "FD gradient     : " << grad.transpose() << std::endl;

        grad_smart_FD(x, grad);
        //std::cout << "smart gradient  : " << grad.transpose() << std::endl;

        return fx;
    }

    double f_eval(const VectorXd& x){

        double fx = 0.0;
        for(int i = 0; i < n; i += 2)
        {
            double t1 = 1.0 - x[i];
            double t2 = 10 * (x[i + 1] - x[i] * x[i]);
            fx += t1 * t1 + t2 * t2;
        }
        return fx;

    }

    void grad_exact(const VectorXd& x, VectorXd& grad){

        for(int i = 0; i < n; i += 2)
        {
            double t1 = 1.0 - x[i];
            double t2 = 10 * (x[i + 1] - x[i] * x[i]);

            grad[i + 1] = 20 * t2;
            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
            
        }

    }

    void grad_standard_FD(const VectorXd& x, VectorXd& grad){

        int n = x.size();
        double eps = 1e-3;
        Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(n, n);

        for(int i = 0; i<n; i++){
            grad[i] = (f_eval(x+eps*Id.col(i)) - f_eval(x-eps*Id.col(i)))/(2*eps);
        }
    }

    void grad_smart_FD(const VectorXd& x, VectorXd& grad){

        int n = x.size();
        double eps = 1e-3;
        MatrixXd Id = MatrixXd::Identity(n, n);

        MatrixXd G(n,n);
        computeG(x, G);

        for(int i = 0; i<n; i++){
            grad[i] = (f_eval(x+eps*G*Id.col(i)) - f_eval(x-eps*G*Id.col(i)))/(2*eps);
        }

        grad = G.transpose().fullPivLu().solve(grad);

    }

    void computeG(const VectorXd& x, MatrixXd& G){

        // construct/update X_diff matrix
        // go to else in first call otherwise true
        if(Xdiff_initialized == true){

            VectorXd temp_col(n);
            for(int i=n-1; i>0; i--){
                temp_col = X_diff.col(i-1);
                X_diff.col(i) = temp_col;
            }

            X_diff.col(0) = x - x_prev;
            //std::cout << "X_diff = \n" << X_diff << std::endl;

            // add small noise term to diagonal, in case columns are linearly dependent
            double eps = 10e-6;
            X_diff = X_diff + eps*MatrixXd::Identity(n,n);

        } else {
            X_diff = MatrixXd::Identity(n,n);
            //std::cout << "X_diff = \n" << X_diff << std::endl;

            Xdiff_initialized = true;
        }

        // store current iterate
        x_prev = x;

        // do modified GRAM-SCHMIDT-ORTHONORMALIZATION
        G.Zero(n, n);
        MatrixXd R = Eigen::MatrixXd::Zero(n, n);


        for(int k=0; k<n; k++){
            G.col(k) = X_diff.col(k);
            for(int i = 0; i<k; i++){
                R(i,k) = G.col(i).transpose()*G.col(k);
                G.col(k) = G.col(k) - R(i,k)*G.col(i);
            }
            R(k,k) = G.col(k).norm();
            G.col(k) = G.col(k)/R(k,k);
        }

        // check if X_diff = G*R
        //std::cout << "norm(X_diff - G*R) = " << (X_diff - G*R).norm() << std::endl;
        //std::cout << "G :\n" << G << std::endl;

    }
    
};




int main()
{
    const int n = 6;
    // Set up parameters
    LBFGSParam<double> param;
    param.epsilon = 1e-3;
    param.epsilon_rel = 1e-3;
    param.delta = 1e-3;
    param.max_iterations = 50;

    // Create solver and function object
    LBFGSSolver<double> solver(param);
    Rosenbrock fun(n);

    // Initial guess
    VectorXd x = 0*VectorXd::Ones(n);
    // x will be overwritten to be the best point found
    double fx;
    int niter = solver.minimize(fun, x, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    return 0;
}