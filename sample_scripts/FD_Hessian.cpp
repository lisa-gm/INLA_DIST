// Finite Difference Hessian approximation

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Dense>

typedef Eigen::VectorXd Vector;

using namespace Eigen;


/*
# hessian dummy
# for (d^2f / dx^2) 
# (f(x_i+1) - 2*f(x_i) + f(x_i-1))/(delta x)^2

# for (d^2f / dx dy )
# (f(x_i+1,j+1) - f(x_i+1,j-1) - f(x_i-1,j+1) + f(x_i-1,j-1))/(4*delta x * delta y)


# 2D
# f = f(x) with x=(x1,x2)

f_eval <- function(x){
  x[1]^3*x[2]^2*x[3]
}

# for comparison
# d^2f / dx^2  : 6*y^2*x
# d^2f / dy^2  : 2*x^3
# d^2f / dx dy : 6*x^2*y 

f_hess_analytic <- function(x){
  d2fdx2  <- 6*x[1]*x[2]^2*x[3]
  d2fdy2  <- 2*x[1]^3*x[3]
  d2fdz2  <- 0
  
  d2fdxdy <- 6*x[1]^2*x[2]*x[3]
  d2fdxdz <- 3*x[1]^2*x[2]^2
  d2fdydz <- x[1]^3*2*x[2]
  
  return(matrix(data=c(d2fdx2, d2fdxdy, d2fdxdz, d2fdxdy, d2fdy2, d2fdydz, d2fdxdz, d2fdydz, d2fdz2), nrow=3))
}


f_hess_2D <- function(x, eps=0.005){
  epsId = diag(x = eps, nrow=length(x))
  
  d2fdx2  <- (f_eval(x+epsId[1,]) - 2 * f_eval(x) + f_eval(x-epsId[1,]))/(eps^2)
  d2fdy2  <- (f_eval(x+epsId[2,]) - 2 * f_eval(x) + f_eval(x-epsId[2,]))/(eps^2)
  d2fdxdy <- (  f_eval(x+epsId[1,]+epsId[2,]) 
              - f_eval(x+epsId[1,]-epsId[2,]) - f_eval(x-epsId[1,]+epsId[2,])
              + f_eval(x-epsId[1,]-epsId[2,]))/(4*eps^2)
  
  return(matrix(data=c(d2fdx2, d2fdxdy, d2fdxdy, d2fdy2), nrow=2))

}

f_hess <- function(x, eps=0.005){
  dim_x = length(x)
  epsId = diag(x = eps, nrow=dim_x)
  
  hess <- matrix(NA, nrow=dim_x, ncol=dim_x)
  
  for(i in 1:length(x)){
    for(j in i:length(x)){
      
        if(i == j){
          hess[i,i] = (f_eval(x+epsId[i,]) - 2 * f_eval(x) + f_eval(x-epsId[i,]))/(eps^2)
        } else {
          hess[i,j] = (  f_eval(x+epsId[i,]+epsId[j,]) 
                         - f_eval(x+epsId[i,]-epsId[j,]) - f_eval(x-epsId[i,]+epsId[j,])
                         + f_eval(x-epsId[i,]-epsId[j,]))/(4*eps^2)
        }
    }
  }
  # make symmetric from upper triangular matrix
  hess <- forceSymmetric(hess)
  return(hess)
}


x = c(1,2,3)

f_hess_analytic(x)
#f_hess_2D(x)
f_hess(x)

*/

// HESSIAN 1D

#if 0

// OLD 1D VERSION
		// construct 2nd order cetnral difference
		Vector eps(1);
		eps[0] = 0.005;

		Vector theta_forw(1);
		theta_forw = theta + eps;

		Vector theta_back(1);
		theta_back = theta - eps;

		Vector mu_dummy(1);

		double f_theta = eval_post_theta(theta, mu);
		double f_theta_forw = eval_post_theta(theta_forw, mu_dummy);
		double f_theta_back = eval_post_theta(theta_back, mu_dummy);

		MatrixXd hess(1,1);
		// careful : require negative hessian (swapped signs in eval post theta) 
		// but then precision negative hessian -> no swapping
		hess << (1.0)/(eps[0]*eps[0]) * (f_theta_forw - 2*f_theta + f_theta_back);
		
		std::cout << "hess " << hess << std::endl;

		MatrixXd cov(1,1);
		cov << 1.0/hess(0,0);


#endif


double f_eval(Vector& x){
	// x[1]^3*x[2]^2*x[3]

	return(pow(x[0],3)*pow(x[1],2)*x[2]);
}

MatrixXd hess_eval(Vector& x){

	double eps = 0.005;

	int dim_x = x.size();
	MatrixXd epsId(dim_x, dim_x); 
	epsId = eps*epsId.setIdentity();

	MatrixXd hessUpper = MatrixXd::Zero(dim_x, dim_x);

	for(int i=0; i < dim_x; i++){
		for(int j=i; j < dim_x; j++){

			if(i == j){
				Vector x_forw_i = x+epsId.col(i);
				Vector x_back_i = x-epsId.col(i);

				hessUpper(i,i) = (f_eval(x_forw_i) - 2 * f_eval(x) + f_eval(x_back_i))/(eps*eps);
			} else {
				Vector x_forw_i_j 		= x+epsId.col(i)+epsId.col(j);
				Vector x_forw_i_back_j = x+epsId.col(i)-epsId.col(j);
				Vector x_back_i_forw_j = x-epsId.col(i)+epsId.col(j);
				Vector x_back_i_j 		= x-epsId.col(i)-epsId.col(j);

    		hessUpper(i,j) = (  f_eval(x_forw_i_j) \
                       - f_eval(x_forw_i_back_j) - f_eval(x_back_i_forw_j) \
                       + f_eval(x_back_i_j)) / (4*eps*eps);
       }
		}
	}

	MatrixXd hess = hessUpper.selfadjointView<Upper>();
	return hess;

}


int main(int argc, char* argv[]){


	Vector x(3);
	x << -5,2,3;

	double f = f_eval(x);

	std::cout << "x    : " << x[0] << " " << x[1] << " " << x[2] << std::endl;
	std::cout << "f(x) : " << f << std::endl;

	MatrixXd hess = hess_eval(x);
	std::cout << "hessian : \n" << hess << std::endl;

	return 0;
}
