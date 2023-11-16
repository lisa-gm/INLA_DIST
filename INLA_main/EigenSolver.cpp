#include "EigenSolver.h"


EigenSolver::EigenSolver(int& MPI_rank){
    printf("in constructor Eigen solver.\n");
}


void symbolic_factorization(SpMat& Q, int& init){
    printf("in symbolic factorization Eigen solver.\n");

    if(init == 0){
        printf("symbolic factorization flag already set. init = %d\n", init);
    } else {
        solverEigenQ.analyzePattern(Q);
        init = 1;
    }

}


void EigenSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol){
    printf("in factorize Eigen solver.\n");

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    solverEigenQ.factorize(Q);
}

