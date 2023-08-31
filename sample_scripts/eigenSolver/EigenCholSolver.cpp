#include "EigenCholSolver.h"


EigenCholSolver::EigenCholSolver(int& MPI_rank_) : MPI_rank(MPI_rank_){
    printf("in constructor Eigen solver.\n");

    init = 0;
    //SimplicialLLT<SpMat> solverEigenQ;
    CholmodSupernodalLLT<SpMat> solverEigenQ;
}


void EigenCholSolver::symbolic_factorization(SpMat& Q, int& init){
    printf("in symbolic factorization Eigen solver.\n");

    if(init == 1){
        printf("symbolic factorization flag already set. init = %d\n", init);
    } else {
        solverEigenQ.analyzePattern(Q);
        init = 1;
    }

}


void EigenCholSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol){
    printf("in factorize Eigen solver.\n");

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    t_priorLatChol = - omp_get_wtime();

    solverEigenQ.factorize(Q);

    log_det = solverEigenQ.logDeterminant();
    //log_det = log(solverEigenQ.determinant());

    t_priorLatChol += omp_get_wtime();
    //std::cout << "log det = " << log(solverEigenQ.determinant()) << ", comp = " << log_det << std::endl;
}


void EigenCholSolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){
    printf("in factorize w/ constraints Eigen solver DUMMY.\n");
    exit(1);
}

void EigenCholSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve){
    printf("in factorize solve Eigen solver.\n");

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    t_condLatChol = - omp_get_wtime();
    solverEigenQ.factorize(Q);

    log_det = solverEigenQ.logDeterminant();
    //log_det = log(solverEigenQ.determinant());
    t_condLatChol += omp_get_wtime();

    t_condLatSolve = - omp_get_wtime();
    sol = solverEigenQ.solve(rhs);
    t_condLatSolve += omp_get_wtime();

}

void EigenCholSolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){
    printf("in factorize_solve_w_constr Eigen solver DUMMY.\n");
    exit(1);      
}

void EigenCholSolver::selected_inversion(SpMat& Q, Vect& inv_diag){
    printf("in selected inversion Eigen solver.\n");

    SpMat spId(Q.rows(), Q.cols());
    spId.setIdentity();

    SpMat Qinv = solverEigenQ.solve(spId);

    std::cout << "norm(Q*Qinv - I) = " << (Q*Qinv - spId).norm() << std::endl;
    inv_diag = Qinv.diagonal();
}

void EigenCholSolver::selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){
    printf("in selected inversion w/ constraints Eigen solver DUMMY.\n");
    exit(1);        
}

void EigenCholSolver::compute_full_inverse(MatrixXd& H, MatrixXd& C){
    printf("in compute full inverse Eigen solver.\n");

    C = H.inverse();
    //std::cout << "inv(H) = \n" << C << std::endl;
}

EigenCholSolver::~EigenCholSolver(){
    printf("in EigenCholSolver destructor.\n");
}

