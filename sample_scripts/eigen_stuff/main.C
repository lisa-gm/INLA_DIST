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
#include <Eigen/SparseCholesky>


using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRmMat;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;


// compute "score" for y_predict vs y_observed
/*
scoresFunc <- function(y, m, v) {
    ds <- function(y, m, s) {
        (y - m)^2 / v + log(v) 
    }

    }
    scrps.g <- function(y, m, s) {
        md <- y - m
        -0.5 * log(2 * s/sqrt(pi)) - sqrt(pi) *
            (s * dnorm(md/s) - 
             md/2 + md * pnorm(md/s))/s
    }
    data.frame(DS = ds(y, m, v),
               CRPS = crps.g(y, m, v),
               SCRPS = scrps.g(y, m, v))
}
*/

// y_predict_var expects: diag(A*Qinv*A^T) + 1/precision_obs -> need to do exp() of theta[0]
// returns: 1st column: DS, 2nd column: CRPS, 3rd column: SCRPS
MatrixXd scoreFunction(Vect& y_observed, Vect& y_predict_mean, Vect& y_predict_var){
    int dimy = y_observed.size();
    MatrixXd M(dimy, 3);
    M.setZero();

    Vect y_diff = y_observed - y_predict_mean;
    // need to work with arrays
    // ds = y_diff.^2 ./ y_predict_var + log(v)
    M.col(0) = y_diff.array() * y_diff.array() / y_predict_var.array() + y_predict_var.array().log();

    // crps.g = y_predict_var / sqrt(pi) - 2*y_predict_var*dnorm(y_diff/s) + y_diff*(1-2*pnorm(y_diff/s))
    // dnorm -> evaluate standard normal distr. with zero mean, sd = 1, at x = y_diff/s
    Vect y_diffbyVar = y_diff.array() / y_predict_var.array();
    // dnorm: 1/sqrt(2*M_PI) exp(x^2 / 2)
    Vect dnorm_ydiffVar(dimy);
    dnorm_ydiffVar = 1 / sqrt(2*M_PI) * exp(-y_diffbyVar.array()*y_diffbyVar.array()/2);
    // pnorm: 1/2 (1 + erf(x / sqrt(2))) -> erf(x)
    Vect pnorm_yDiffvar(dimy);
    for(int i=0; i<dimy; i++){
      pnorm_yDiffvar[i] = 0.5 * (1 + erf(y_diffbyVar[i]/sqrt(2)));
    }
    printf("y/s = %f, dnorm = %f, pnorm = %f\n", y_diffbyVar[0], dnorm_ydiffVar[0], pnorm_yDiffvar[0]);

    M.col(1) =1/sqrt(M_PI) * y_predict_var.array() 
               - 2*y_predict_var.array() * dnorm_ydiffVar.array()
               + y_diff.array()*(ArrayXd::Ones(dimy) - 2*pnorm_yDiffvar.array());

    // scrps.g 
    // md = y_obs - y_pred
    //  -0.5 * log(2 * s/sqrt(pi)) - sqrt(pi) * (s * dnorm(md/s) -  md/2 + md * pnorm(md/s))/s
    M.col(2) = - 0.5*(2*y_predict_var.array()/sqrt(M_PI)).log()
               - sqrt(M_PI)/y_predict_var.array()*(y_predict_var.array()*dnorm_ydiffVar.array() - 0.5*y_diff.array() + y_diff.array() * pnorm_yDiffvar.array());


    return M;
}


int main(int argc, char *argv[]) {

#if 0 
int nrows = 4;
int ncols = 4;

// generate sparse matrix A with random sparsity pattern
MatrixXd A_dense = MatrixXd::Random(nrows,ncols);
A_dense = A_dense*A_dense.transpose();

//std::cout << "A: \n" << A_dense << std::endl;

 for(int i=0; i<nrows; i++){
    for(int j=0; j<ncols; j++){
        if(A_dense(i,j) < 0.4){
            A_dense(i,j) = 0;
        }
    }
 }

 MatrixXd V = MatrixXd::Random(ncols,1);
 std::cout << "V: " << V.transpose() << std::endl;
 MatrixXd S = 2.0*MatrixXd::Random(1, ncols);
 std::cout << "V: " << S << std::endl;

 std::cout << "A_dense:\n" << A_dense << std::endl;

SpMat A = A_dense.sparseView();

for (int k=0; k<A.outerSize(); ++k)
  for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
  {
    std::cout << it.value() << " ";
    std::cout << it.row() << " ";   // row index
    std::cout << it.col() << " ";   // col index (here it is equal to k)
    std::cout << it.index() << std::endl; // inner index, here it is equal to it.row()

    std::cout << "V.row(" << it.row() << "): " << V.row(it.row()) << std::endl;
    std::cout << "S.col(" << it.col() << "): " << S.col(it.col()) << std::endl;

    it.valueRef() = V.row(it.row())*S.col(it.col());
  }

std::cout << "A:\n" << MatrixXd(A) << std::endl;
std::cout << "V*S = \n" << V*S << std::endl;
std::cout << "A - V*S = \n" << A - V*S << std::endl;

#endif

#if 0
int m = 8;

Vect y = Vect::Random(m);

Vect y_diff(m);
y_diff.setOnes();
y_diff[2] = 0;
y_diff[4] = 0;

std::cout << "y      = " << y.transpose() << std::endl;
std::cout << "y diff = " << y_diff.transpose() << std::endl;
// compute MSE
std::cout << "norm(y) = " << 1.0/y.size() * y.squaredNorm() << std::endl;

Vect y_diff1(m);
y_diff1.array() = y.array() * y_diff.array();
std::cout << "y diff 1 = " << y_diff1.transpose() << std::endl;
std::cout << "norm(y_diff1) = " << 1.0/y_diff.sum() * y_diff1.squaredNorm() << std::endl;

Vect y_diff0(m);
Vect y_diff_opposite = Vect::Ones(m) - y_diff;
y_diff0.array() = y.array() * y_diff_opposite.array();
std::cout << "y diff 0 = " << y_diff0.transpose() << std::endl;
std::cout << "norm(y_diff0) = " << 1.0/y_diff_opposite.sum() * y_diff0.squaredNorm() << std::endl;

#endif

// implement scoring function 
#if 1

int m = 5;
Vect y_observed = 1.1*Vect::Random(m);
std::cout << "y_observed : " << y_observed.transpose() << std::endl;
Vect y_predict  = 1.3*Vect::Random(m);
std::cout << "y_predict  : " << y_predict.transpose() << std::endl;
Vect v          = Vect::Random(m) + Vect::Ones(m);
std::cout << "v          : " << v.transpose() << std::endl;


MatrixXd M = scoreFunction(y_observed, y_predict, v);

std::cout << "M: \n" << M << std::endl;


#endif


}