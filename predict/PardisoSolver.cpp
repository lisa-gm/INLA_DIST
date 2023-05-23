#include "PardisoSolver.h"

PardisoSolver::PardisoSolver(int& MPI_rank_) : MPI_rank(MPI_rank_){

    mtype  = -2;             /* set to positive semi-definite */

    /* -------------------------------------------------------------------- */
    /* ..  Setup Pardiso control parameters.                                */
    /* -------------------------------------------------------------------- */

    error  = 0;
    solver = 0;              /* use sparse direct solver */
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error); 

    if (error != 0) 
    {
        if (error == -10 )
           printf("No license file found \n");
        if (error == -11 )
           printf("License is expired \n");
        if (error == -12 )
           printf("Wrong username or hostname \n");
       exit(1); 
    } else {
        #ifdef PRINT_PAR
            printf("[PARDISO]: License check was successful ... \n");
        #endif
    }

    int threads_level2 = omp_get_max_threads();

//#ifdef PRINT_OMP
        if(MPI_rank == 0){
            //char* var = getenv("OMP_NUM_THREADS");
            //std::cout << "OMP_NUM_THREADS = " << var << std::endl;
            std::cout << "Pardiso will be called with " << threads_level2 << " threads per solver. " << std::endl;
        }
        // printf("Thread rank: %d out of %d threads.\n", omp_get_thread_num(), omp_get_num_threads());
//#endif

    // make sure that this is called inside upper level parallel region 
    // to get number of threads on the second level 
    iparm[2] = threads_level2;

    iparm[33] = 1;      /* always returns the same result, even when executed in parallel,
                           becomes a problem when doing multiple solves at once */

    maxfct = 1;         /* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */

    msglvl = 0;         /* Print statistical information  */

#ifdef MEAS_GFLOPS
    if(MPI_rank == 0)
        msglvl = 1;
#endif

    error  = 0;         /* Initialize error flag */
    init = 0;           /* switch that determines if symbolic factorisation already happened */

} // end constructor


void PardisoSolver::symbolic_factorization(SpMat& Q, int& init){

#ifdef PRINT_PAR
    std::cout << "in symbolic factorization. thread ID: " << omp_get_thread_num() << std::endl;
#endif

    n = Q.rows();
    int nrhs   = 1;              /* in symbolic factorization only dummy variable */

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // this time require CSR format

    nnz = Q_lower.nonZeros();

    #ifdef PRINT_PAR
        std::cout << "number of non zeros : " << nnz << std::endl;
    #endif

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    } 

    nnz = ia[n];

    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    /* -------------------------------------------------------------------- */
    /* ..  Reordering and Symbolic Factorization.  This step also allocates */
    /*     all memory that is necessary for the factorization.              */
    /* -------------------------------------------------------------------- */
    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
     &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, &ddum, &ddum, &error, dparm);

    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }

    #ifdef PRINT_PAR
        printf("\nReordering completed ... ");
        printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
        printf("\nNumber of factorization GFLOPS = %d\n", iparm[18]);
    #endif

    // set init to 1 to indicate that symbolic factorisation happened
    init = 1;

    delete[] ia;
    delete[] ja;
    delete[] a;

}


void PardisoSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol){

    int nrhs   = 1;              /* here only dummy variable */

#ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
#endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // first measure flops of factorize_solve then factorize ... 
    // so that they don't get mixed up in output
/*#ifdef MEAS_GFLOPS
    msglvl = 0;
#endif*/

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);

        for(i = 0; i < nnz; i++){
            if(isnan(a[i])){
                std::cout << "In factorize!Found NaN value in *a. a[" << i << "] = " << a[i] << std::endl;
            }

            if(isinf(a[i])){
                std::cout << "In factorize!Found Inf value in *a. a[" << i << "] = " << a[i] << std::endl;
            }
        }

        exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    t_priorLatChol = -omp_get_wtime();

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);

    t_priorLatChol += omp_get_wtime();
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

    log_det = dparm[32];

#ifdef PRINT_MSG
    std::cout << std::fixed << std::setprecision(12) << "in factorize. log det : " << dparm[32] << ", 0.5*log_det : " << 0.5*dparm[32] << std::endl;
#endif

    delete[] ia;
    delete[] ja;
    delete[] a;

} // end factorize function

void PardisoSolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){

    int nrhs = D.rows();

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    double *b;
    double *x;

    b = new double [nrhs*n];
    x = new double [nrhs*n];

    // want rows of D to become vector ... 
    MatrixXd Dt = D.transpose();
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));
    
    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);

        for(i = 0; i < nnz; i++){
            if(isnan(a[i])){
                std::cout << "Found NaN value in *a. a[" << i << "] = " << a[i] << std::endl;
            }
        }

    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    double t_factorise = - omp_get_wtime();

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }

    t_factorise += omp_get_wtime();
    //std::cout << "time factorise : " << t_factorise << std::endl;
    //printf("\nFactorization completed ...\n");

#ifdef PRINT_PAR
    std::cout << std::fixed << std::setprecision(12) << "in factorize. log det       : " << dparm[32] << std::endl;
    
#endif
    log_det = dparm[32];

    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    // map solution back
    memcpy(V.data(), x, nrhs*n*sizeof(double));
    //std::cout << "norm(Q*V   - t(D) = " << (Q*V - D.transpose()).norm() << std::endl;

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete [] b;
    delete [] x;

} // end factorize_w_constr

 
void PardisoSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve){

    int nrhs   = 1;              /* only one solve necessary */

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    // first measure flops of factorize_solve then factorize ... 
    // so that they don't get mixed up in output
#ifdef MEAS_GFLOPS
    msglvl = 0;
#endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    b = new double [n];
    x = new double [n];

    /* Set right hand side to i. */
    for (int i = 0; i < n; i++) {
        b[i] = rhs[i];
    } 


    /* -------------------------------------------------------------------- */
    /* ..  pardiso_chkvec(...)                                              */
    /*     Checks the given vectors for infinite and NaN values             */
    /*     Input parameters (see PARDISO user manual for a description):    */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }


    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);

        for(i = 0; i < nnz; i++){
            if(isnan(a[i])){
                std::cout << "In factorize solve!\nFound NaN value in *a. a[" << i << "] = " << a[i] << std::endl;
            }
        }

    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    t_condLatChol = -omp_get_wtime();

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);

    t_condLatChol += omp_get_wtime();
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

#ifdef PRINT_MSG
    std::cout << "in factorize solve. log det : " << std::setprecision(12) << dparm[32] << ", 0.5*log_det : " << 0.5*dparm[32] << std::endl;
#endif
    
    log_det = dparm[32];

    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */

    t_condLatSolve = -omp_get_wtime();
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);

    t_condLatSolve += omp_get_wtime();
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    //printf("\nSolve completed ... ");
    for (i = 0; i < n; i++) {
        //printf("\n x [%d] = % f", i, x[i] );
        sol(i) = x[i];
    }

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end factorise solve function


// perform all solves at the same time
void PardisoSolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){
    
#ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
#endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    int nrhs;

    nrhs = Dxy.rows() + 1;  // from constraints + the regular one
    //std::cout << "nrhs = " << nrhs << std::endl;


#ifdef PRINT_PAR
    std::cout << "After symbolic_factorization" << std::endl;
#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    b = new double [nrhs*n];
    x = new double [nrhs*n];

    //std::cout << "Dxy    = " << Dxy << std::endl;
    //std::cout << "constr = " << constr << std::endl;

    memcpy(b, rhs.data(), n*sizeof(double));
    
    //std::cout << "in constr is true" << std::endl;
    MatrixXd Dt = Dxy.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b + n, Dt.data(), n*Dxy.rows()*sizeof(double));


    /* -------------------------------------------------------------------- */
    /* ..  pardiso_chkvec(...)                                              */
    /*     Checks the given vectors for infinite and NaN values             */
    /*     Input parameters (see PARDISO user manual for a description):    */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }


    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);

        for(i = 0; i < nnz; i++){
            if(isnan(a[i])){
                std::cout << "In factorize solve!\nFound NaN value in *a. a[" << i << "] = " << a[i] << std::endl;
            }
        }

    exit(1);
    }

#ifdef PRINT_PAR
    std::cout << "b : " << std::endl;
    for(int i = 0; i<(Q.rows()*nrhs); i++){
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
#endif


    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

#ifdef PRINT_PAR
    std::cout << "in factorize solve. log det : " << std::setprecision(12) << dparm[32] << ", 0.5*log_det : " << 0.5*dparm[32] << std::endl;
#endif
    
    log_det = dparm[32];

    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 0;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    // map solution back
    memcpy(sol.data(), x, n*sizeof(double));
    //std::cout << "norm(Q*sol - rhs) = " << (Q*sol - rhs).norm() << std::endl;


    memcpy(V.data(), x+n, n*Dxy.rows()*sizeof(double));
    //std::cout << "norm(Q*V   - t(D) = " << (Q*V   - Dxy.transpose()).norm() << std::endl;

    // TODO: too inaccurate??
    /*
    Vect temp1 = Q*sol;
    Vect temp2 = Q*V;

    for(int i=0; i<n; i++){
        std::cout << std::right << std::fixed << temp1[i] << " " << rhs[i] << "       ";
        std::cout << std::right << std::fixed << temp2[i] << " " << Dxy(0,i) << std::endl;
    }
    */

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;


} // end factorize solve with constraints


void PardisoSolver::selected_inversion_diag(SpMat& Q, Vect& inv_diag){

    int nrhs = 1; // only dummy?

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    //msglvl = 1;

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // TODO: make already one-based in the above loop
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

#ifdef PRINT_MSG
    if(MPI_rank == 0)
        std::cout << "MPI rank : " << MPI_rank << ". Calling PARDISO with " << iparm[2] << " threads."<< std::endl;
#endif
    /* -------------------------------------------------------------------- */    
    /* ... Inverse factorization.                                           */                                       
    /* -------------------------------------------------------------------- */  

    // what do we need both b & x for here?
    b = new double [n];
    x = new double [n];

    //printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35]  = 0; /*  no not overwrite internal factor L, CRASHES FOR LARGE MATRICES */ 
    //printf("iparm[35] = %d\n", iparm[35]);

    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

    if (error != 0) {
        printf("\nERROR %d", error);
        exit(1);
    }

    /* print diagonal elements */
    for (k = 0; k < n; k++)
    {
        int j = ia[k]-1;
        //printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
        inv_diag(k) = a[j];
    }

    //std::cout << "indDiag(1:10) = " << inv_diag.head(10).transpose() << std::endl;
    
    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end selected inversion function


void PardisoSolver::selected_inversion_diag_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    //msglvl = 0;

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    int nrhs = D.rows();

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // for Q*V = D
    b = new double [nrhs*n];
    x = new double [nrhs*n];

    // MatrixXd regularly in column-major, hence
    MatrixXd Dt = D.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));


    // TODO: make already one-based in the above loop
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");


    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    // map solution back
    // V has column-major format
    memcpy(V.data(), x, nrhs*n*sizeof(double));
    //std::cout << "norm(Q*V   - t(D)) = " << (Q*V - D.transpose()).norm() << std::endl;

    /* -------------------------------------------------------------------- */    
    /* ... Inverse factorization.                                           */                                       
    /* -------------------------------------------------------------------- */  

    //printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35]  = 0; /*  no not overwrite internal factor L */ 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

    /* print diagonal elements */
    for (k = 0; k < n; k++)
    {
        int j = ia[k]-1;
        //printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
        inv_diag(k) = a[j];
    }

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end selected inversion with constraints function


void PardisoSolver::selected_inversion_full(SpMat& Q, SpMat& Qinv){

    int nrhs = 1; // only dummy?

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    //msglvl = 1;

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // TODO: make already one-based in the above loop
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

#ifdef PRINT_MSG
    if(MPI_rank == 0)
        std::cout << "MPI rank : " << MPI_rank << ". Calling PARDISO with " << iparm[2] << " threads."<< std::endl;
#endif
    /* -------------------------------------------------------------------- */    
    /* ... Inverse factorization.                                           */                                       
    /* -------------------------------------------------------------------- */  

    // what do we need both b & x for here?
    b = new double [n];
    x = new double [n];

    //printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35]  = 0; /*  no not overwrite internal factor L, CRASHES FOR LARGE MATRICES */ 
    //printf("iparm[35] = %d\n", iparm[35]);

    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

    if (error != 0) {
        printf("\nERROR %d", error);
        exit(1);
    }

    SpMat Qinv_lower(n,n);
    Qinv_lower.reserve(nnz);

    // return to 0-based notation
    for (int i = 0; i < n+1; ++i){
        Qinv_lower.outerIndexPtr()[i] = ia[i]-1;
    }  

    for (int i = 0; i < nnz; ++i){
        Qinv_lower.innerIndexPtr()[i] = ja[i]-1;
    }  

    for (int i = 0; i < nnz; ++i){
        Qinv_lower.valuePtr()[i] = a[i];
    }

    Qinv = Qinv_lower.selfadjointView<Lower>();

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end selected inversion function



void PardisoSolver::selected_inversion_full_w_constr(SpMat& Q, const MatrixXd& D, SpMat& Qinv, MatrixXd& V){

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif
    //msglvl = 1;

    int nrhs = D.rows();

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // for Q*V = D
    b = new double [nrhs*n];
    x = new double [nrhs*n];

    // MatrixXd regularly in column-major, hence
    MatrixXd Dt = D.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));

    // TODO: make already one-based in the above loop
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

#ifdef PRINT_MSG
    if(MPI_rank == 0)
        std::cout << "MPI rank : " << MPI_rank << ". Calling PARDISO with " << iparm[2] << " threads."<< std::endl;
#endif


    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */   

    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    // map solution back
    // V has column-major format
    memcpy(V.data(), x, nrhs*n*sizeof(double));
    //std::cout << "norm(Q*V   - t(D)) = " << (Q*V - D.transpose()).norm() << std::endl;

    /* -------------------------------------------------------------------- */    
    /* ... Inverse factorization.                                           */                                       
    /* -------------------------------------------------------------------- */  

    //printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35]  = 0; /*  no not overwrite internal factor L, CRASHES FOR LARGE MATRICES */ 
    //printf("iparm[35] = %d\n", iparm[35]);

    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

    if (error != 0) {
        printf("\nERROR %d", error);
        exit(1);
    }

    SpMat Qinv_lower(n,n);
    Qinv_lower.reserve(nnz);

    // return to 0-based notation
    for (int i = 0; i < n+1; ++i){
        Qinv_lower.outerIndexPtr()[i] = ia[i]-1;
    }  

    for (int i = 0; i < nnz; ++i){
        Qinv_lower.innerIndexPtr()[i] = ja[i]-1;
    }  

    for (int i = 0; i < nnz; ++i){
        Qinv_lower.valuePtr()[i] = a[i];
    }

    Qinv = Qinv_lower.selfadjointView<Lower>();

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end selected inversion function



// function to completely invert small n x n matrix using identity as rhs
// assumes function to be SYMMETRIC (which is the case for hessian)
void PardisoSolver::compute_full_inverse(SpMat& H, MatrixXd& C){

    // convert to sparse matrix for pardiso
    //SpMat H = Q.sparseView();

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    if(init == 0){
        symbolic_factorization(H, init);
    }

// check if n and H.size() match
    if(n != H.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(H) = %ld.\n", n, H.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat H_lower = H.triangularView<Lower>(); 

    // check if nnz and H_lower.nonZeros match
    if(nnz != H_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %ld.\n nnz(H_lower) = %ld\n", nnz, H_lower.nonZeros());
    }

    // IMPORTANT : change the number of right-hand sides
    int nrhs = n;

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    H_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = H_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = H_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = H_lower.valuePtr()[i];
    }

    int n2 = n*n;
    b = new double [n2];
    x = new double [n2];

    /* Set right hand side to Identity. */
    for (int i = 0; i < n2; i++) {
        if(i % (n+1) == 0){
            b[i] = 1.0;
        } else{
            b[i] = 0;
        }
    } 

    /* -------------------------------------------------------------------- */
    /* ..  pardiso_chkvec(...)                                              */
    /*     Checks the given vectors for infinite and NaN values             */
    /*     Input parameters (see PARDISO user manual for a description):    */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }


    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    //printf("\nSolve completed ... ");
    /*for (i = 0; i < n2; i++) {
        printf("\n x [%d] = % f", i, x[i] );
        //sol(i) = x[i];
    }*/

    // fill C with x values
    C = MatrixXd::Map(x, n, n);

    //std::cout << "\nC = \n" << C << std::endl;


    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;


}


PardisoSolver::~PardisoSolver(){

    int nrhs;

    /* -------------------------------------------------------------------- */    
    /* ..  Termination and release of memory.                               */
    /* -------------------------------------------------------------------- */    
    phase = -1;                 /* Release internal memory. */
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);

}
