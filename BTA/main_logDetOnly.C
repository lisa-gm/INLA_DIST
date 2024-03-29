// main_const_ind

#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include "RGF.H"

#include <omp.h>
#include <armadillo>

using namespace std;

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

/*
Start simulations with RGF NBlock Bmin.dat Bmax.dat M.dat
NBlock: number of blocks of the matrix
Bmin.dat: name of file that contains the first index of each block starting with 0
Bmax.dat: name of file that contains the last index of each block with Bmax[i]=Bmin[i+1]
M.dat: name of file that contains the matrix to work on.
Data stored:
size (matrix size)
n_nonzeros (number of non-zero elements)
fortran index (0 or 1)
index_i index_j real imag (4 columns per matrix entry)
*/


  void readCSR(std::string filename, int &n, int &nnz, int* ia, int* ja, double* a)
{

  fstream fin(filename, ios::in);
  fin >> n;
  fin >> n;
  fin >> nnz;

   // allocate memory
   ia = new int [n+1];
   ja = new int [nnz];
   a = new double [nnz];
  
  for (int i = 0; i <= n; i++){
    fin >> ia[i];
  }

  for (int i = 0; i < ia[n]; i++){
    fin >> ja[i];
  }

  for (int i = 0; i < ia[n]; i++){
    fin >> a[i];
  }

  fin.close();
} 

arma::sp_mat readCSC(std::string filename){
  int n_rows; int n_cols;
  int nnz;

  ifstream fin(filename);

  fin >> n_rows;
  fin >> n_cols;
  fin >> nnz;

   // allocate memory
  arma::uvec row_ind(nnz);
  arma::uvec col_ptr(n_cols+1);
  arma::vec a(nnz);

  for (int i = 0; i < nnz; i++){
    fin >> row_ind[i];
  }

  for (int i = 0; i < n_cols+1; i++){
    fin >> col_ptr[i];
    //std::cout <<col_ptr[i] << std::endl;
  }

  for (int i = 0; i < nnz; i++){
    fin >> a[i];
   // std::cout <<a[i] << std::endl;
    }

  fin.close();

  arma::sp_mat A(row_ind, col_ptr, a, n_rows, n_cols);

  //std::cout << "nonzeros A " << A.n_nonzero << std::endl;
  return A;
} 

arma::sp_mat readCSC_sym(std::string filename)
{

  int n;
  int nnz;

  fstream fin(filename, ios::in);
  fin >> n;
  fin >> n;
  fin >> nnz;

   // allocate memory
  arma::uvec row_ind(nnz);
  arma::uvec col_ptr(n+1);
  arma::vec a(nnz);

  for (int i = 0; i < nnz; i++){
    fin >> row_ind[i];}

  for (int i = 0; i < n+1; i++){
    fin >> col_ptr[i];}

  for (int i = 0; i < nnz; i++){
    fin >> a[i];}

  fin.close();

  arma::sp_mat A_lower(row_ind, col_ptr, a, n, n);
  // create entire matrix
  arma::sp_mat A = arma::symmatl(A_lower);

  return A;
} 

arma::mat read_matrix(std::string filename,  int n_row, int n_col){

  arma::mat B(n_row, n_col);
  //std::cout << "size(matrix) : " << arma::size(B) << std::endl;
  B.load(filename, arma::raw_ascii);  // check later if there is a faster alternative
  //B.print();

  return B;
}

void file_exists(std::string file_name)
{
    if (std::fstream{file_name}) ;
    else {
      std::cerr << file_name << " couldn\'t be opened (not existing or failed to open)\n"; 
      exit(1);
    }
    
}

/* ---------------------------------------------------------------------------------------- */
// functions to construct precision matrices, rhs, etc.

#if 1
// SPDE discretisation -- matrix construction
/* void construct_Q_spatial(arma::vec& theta, arma::sp_mat* Qs, \
                         arma::sp_mat* c0, arma::sp_mat* g1, arma::sp_mat* g2){ */
arma::sp_mat construct_Q_spatial(arma::vec theta_u, \
                         arma::sp_mat c0, arma::sp_mat g1, arma::sp_mat g2){

  arma::sp_mat Qs(c0.n_rows, c0.n_cols);
  // Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
  // return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
  Qs = pow(theta_u[0],2)*(pow(theta_u[1], 4) * c0 + 2*pow(theta_u[1],2) * g1 + g2);
  // extract triplet indices and insert into Qx

  return Qs;

}
#endif

// SPDE discretisation -- matrix construction
/* void construct_Q_spatial(arma::vec& theta, arma::sp_mat* Qs, \
                        arma::sp_mat* c0, arma::sp_mat* g1, arma::sp_mat* g2){ */
arma::sp_mat construct_Q_spat_temp(arma::vec theta_u, \
                         arma::sp_mat c0, arma::sp_mat g1, arma::sp_mat g2, arma::sp_mat g3, \
                         arma::sp_mat M0, arma::sp_mat M1, arma::sp_mat M2){

  int n_st = c0.n_rows * M0.n_rows;
  arma::sp_mat Qst(n_st, n_st);

  // g^2 * fem$c0 + fem$g1
  arma::sp_mat q1s = pow(theta_u[1],2) * c0 + g1;
  //std::cout << "q1s :" << std::endl;
  //arma::mat(q1s).submat(0,0,10,10).print();

  // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
  arma::sp_mat q2s = pow(theta_u[1], 4) * c0 + 2 * pow(theta_u[1],2) * g1 + g2;
  //std::cout << "q2s :" << std::endl;
  //arma::mat(q2s).submat(0,0,10,10).print();

  // g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
  arma::sp_mat q3s = pow(theta_u[1], 6) * c0 + 3 * pow(theta_u[1],4) * g1 + 3 * pow(theta_u[1],2) * g2 + g3;
  //std::cout << "q3s :" << std::endl;
  //arma::mat(q3s).submat(0,0,10,10).print();

  // assemble overall precision matrix Q.st
  Qst = pow(theta_u[0],2)*(kron(M0, q3s) + kron(2 * theta_u[2] * M1, q2s) +  kron(theta_u[2] * theta_u[2] * M2, q1s));
  //std::cout << "Qst :" << std::endl;
  //arma::mat(Qst).submat(0,0,10,10).print();

  return Qst;

}


int main(int argc, char* argv[]) 
{ 

 if (argc != 5 + 1) {
    std::cerr << "RGF Call : path_to_folder ns nt nb no" << std::endl;


    std::cerr << "[string:base_path]          path to folder containing all files " << std::endl;      
    std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;      
    std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;      
    std::cerr << "[integer:nb]                number of fixed effects" << std::endl;  

    std::cerr << "[integer:no]                number of data samples" << std::endl;      
      
    exit(1);
  } 

  // start timer
  double overall_time = -omp_get_wtime();


  std::string base_path = argv[1];
  size_t ns                = atoi(argv[2]);
  size_t nt                = atoi(argv[3]);
  size_t nb                = atoi(argv[4]);
  size_t no                = atoi(argv[5]);

  size_t nu                = ns*nt;

  // also save as string
  std::string ns_s = std::to_string(ns);
  std::string nt_s = std::to_string(nt);
  std::string nb_s = std::to_string(nb);
  std::string no_s = std::to_string(no);  

  std::string nu_s = std::to_string(nu);

  std::cout << "./RGF CALL  " << ns_s << " " << nt_s << " " << nb_s << " " << no_s << "\n" << std::endl;

  // ------------------- construct file names and check if files exist --------------------------

  // files to construct Q.u depending on HYPERPARAMETERS theta
  std::string c0_file      =  base_path + "/c0_" + ns_s + ".dat";
  file_exists(c0_file);
  std::string g1_file      =  base_path + "/g1_" + ns_s + ".dat";
  file_exists(g1_file);
  std::string g2_file      =  base_path + "/g2_" + ns_s + ".dat";
  file_exists(g2_file);


  std::string g3_file;
  std::string M0_file; 
  std::string M1_file;
  std::string M2_file;  

  #if 1
  if(nt > 1){
    g3_file      =  base_path + "/g3_" + ns_s + ".dat";
    file_exists(g3_file);

    M0_file      =  base_path + "/M0_" + nt_s + ".dat";
    file_exists(M0_file);
    M1_file      =  base_path + "/M1_" + nt_s + ".dat";
    file_exists(M1_file);
    M2_file      =  base_path + "/M2_" + nt_s + ".dat";
    file_exists(M2_file);  
  }
  #endif

  // INDEPENDENT OF HYPERPARAMETERS

  // Ax = cbind(As, B)
  std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + std::to_string(nu+nb) + ".dat";
  file_exists(Ax_file);

  // temperature values y
  std::string y_file        =  base_path + "/y_" + no_s + "_1.dat";
  file_exists(y_file);
  //std::cout << y_file << std::endl;

  // ------------------------------- read in files  --------------------------------- //

  // READ IN MATRICES

  // spatial matrices
  arma::sp_mat c0 = readCSC_sym(c0_file);
  arma::sp_mat g1 = readCSC_sym(g1_file);
  arma::sp_mat g2 = readCSC_sym(g2_file);

  arma::sp_mat g3;
  arma::sp_mat M0;
  arma::sp_mat M1;
  arma::sp_mat M2;

  if(nt > 1){ 
    g3 = readCSC_sym(g3_file); 
    //arma::mat(g3).submat(0,0,10,10).print();


    // temporal matrices 
    M0 = readCSC_sym(M0_file);
    //arma::mat(M0).submat(0,0,nt-1,nt-1).print();
    M1 = readCSC_sym(M1_file);
    //arma::mat(M1).submat(0,0,nt-1,nt-1).print();
    M2 = readCSC_sym(M2_file); 
    //arma::mat(M2).submat(0,0,nt-1,nt-1).print();

  }
  


  // Ax (sparse)
  arma::sp_mat Ax = readCSC(Ax_file);
  //std::cout << "non zeros A_st " << Ax.n_nonzero << std::endl;
  //arma::mat(M0).submat(0,0,nt-1,nt-1).print();

  //y (vector)
  arma::vec y = read_matrix(y_file, no, 1);

  /* ----------------------- initialise random theta -------------------------------- */

  arma::vec theta;

  if(nt == 1){
    theta = {-1.5,-5,-2};
    //theta.print();
  } else {
    theta = {1.386, -5.883, 1.04, 3.689};
    //theta = {5, -10, 2.5, 1};
    //theta.print();
  }


  arma::sp_mat Qu;

 // assemble Qs for given theta
  if(nt == 1){

    Qu = construct_Q_spatial(exp(theta(arma::span(1,2))), c0, g1, g2);
    // arma::mat(Qu).submat(0,0,10,10).print();
  
  } else {

    std::cout << "theta u : " << std::endl;
    std::cout << exp(theta(arma::span(1,3))) << std::endl;
    Qu = construct_Q_spat_temp(exp(theta(arma::span(1,3))), c0, g1, g2, g3, M0, M1, M2);
    //arma::mat(Qu).submat(0,0,10,10).print();

  }

  // Q.b : diag(1e-5), dim : nb nb
  arma::sp_mat Qb = 1e-5 * arma::speye(nb, nb); 
  //Qb.print(); 

  // Qub0  --> create here, all zeros 
  arma::sp_mat Qub0(nb, ns*nt); Qub0.zeros();
    
  /* ------------- ASSEMBLE MATRICES  ----------------- */

  // assemble Q.x = block_diagonal(Q.u, Q.b)
  size_t n = size(Qu)[1] + size(Qb)[1];
  std::cout << "n : " << n << std::endl;
  arma::sp_mat Qx(n,n);
  Qx(0,0, size(Qu))           = Qu;
  Qx(0,nu, size(Qub0.t()))    = Qub0.t();
  Qx(nu, 0, size(Qub0))       = Qub0;
  Qx(nu, nu, size(Qb))        = Qb;
  //arma::mat(Qx).submat(0,0,10,10).print();

  // Q.x|y = Q.x + t(A.x), Q.e*A.x
  arma::sp_mat Qxy(size(Qx));
  Qxy = Qx + exp(theta[0])*Ax.t()*Ax;
  //std::cout << "Qxy : " << std::endl;
  //arma::mat(Qxy).submat(0,0,10,10).print();


  arma::vec bxy = exp(theta[0])*Ax.t()*y;
  //bxy.print();

  //std::cout << "size(Qxy)" << size(Qxy) << std::endl;
  //std::cout << "size(bxy)" << size(bxy) << std::endl;

  // TAKE ENTIRE MATRIX FOR THIS SOLVER
  arma::sp_mat Qxy_lower = arma::trimatl(Qxy);

  // require CSR format
  size_t nnz = Qxy_lower.n_nonzero;

  //std::cout << "number of non zeros : " << nnz << std::endl;

  size_t* ia; 
  size_t* ja;
  double* a; 

  // allocate memory
  ia = new size_t [n+1];
  ja = new size_t [nnz];
  a = new double [nnz];

  std::cout << n << std::endl;
  std::cout << n << std::endl;
  std::cout << nnz << std::endl;


  for (int i = 0; i < nnz; ++i){
    ja[i] = Qxy_lower.row_indices[i];
    //std::cout << ja[i] << std::endl;     

  } 

  for (int i = 0; i < n+1; ++i){
    ia[i] = Qxy_lower.col_ptrs[i]; 
    //std::cout << ia[i] << std::endl;   
  }  

  for (int i = 0; i < nnz; ++i){
    a[i] = Qxy_lower.values[i];
    //std::cout << a[i] << std::endl;     

  }  

  printf("\nAll matrices assembled. Passing to RGF solver now.\n");

  // ------------------------------------------------------------------------------------------- // 
  // -------------------------------------- RGF SOLVER ----------------------------------------- //
  // ------------------------------------------------------------------------------------------- // 

  int i;
  double data;
  double t_factorise1, t_factorise2, t_solve, t_inv;
  T *b;
  T *x;
  T *invDiag;
  RGF<T> *solver;

  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  printf ("The current date/time is: %s\n",asctime(timeinfo));

  b      = new T[n];
  x      = new T[n];
  invDiag= new T[n];

  // assign b to correct format
  for (int i = 0; i < n; i++){
    b[i] = bxy[i];
    //printf("%f\n", b[i]);
  }
  
  solver = new RGF<T>(ia, ja, a, ns, nt, nb);

  t_factorise1 = get_time(0.0);
  //solver->solve_equation(GR);
  double logDet_new;
  double flops_factorize = solver->factorize_noCopyHost(logDet_new);
  t_factorise1 = get_time(t_factorise1);

  t_factorise2 = get_time(0.0);

  double flops_factorize2 = solver->factorize();
  double log_det = solver->logDet();

  t_factorise2 = get_time(t_factorise2);
  //double log_det = solver->logDet();
  printf("log det new : %f\n", logDet_new);
  printf("logdet      : %f\n", log_det);

  printf("\n\n");


  printf("flops factorize: %f and %f\n", flops_factorize, flops_factorize2);

  // write this to file
  /*std::string L_factor_file_name = base_path + "/L_factor_RGF"  + "_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s + ".dat";
  std::ofstream L_factor_file(L_factor_file_name,    std::ios::out | std::ios::trunc);

  L_factor_file << n << std::endl;
  L_factor_file << n << std::endl;
  L_factor_file << M->n_nonzeros << std::endl;

  for (int i = 0; i < M->size+1; ++i){
    L_factor_file << M->edge_i[i] << std::endl;
  }
   for (int i = 0; i < M->n_nonzeros; ++i){
    L_factor_file << M->index_j[i] << std::endl;
  }  
  for (int i = 0; i < M->n_nonzeros; ++i){
    L_factor_file << MF[i] << std::endl;
  }  

  L_factor_file.close(); */

  t_solve = get_time(0.0); 
  double flops_solve = solver->solve(x, b, 1);
  t_solve = get_time(t_solve);
  printf("flops solve:     %f\n", flops_solve);


  t_inv = get_time(0.0);
  double flops_inv = solver->RGFdiag(invDiag);
  t_inv = get_time(t_inv);
  printf("flops inv:      %f\n", flops_inv);


  //exit(1);

  printf("RGF factorise1 time: %lg\n",t_factorise1);
  printf("RGF factorise2 time: %lg\n",t_factorise2);
  printf("RGF solve     time: %lg\n",t_solve);
  printf("RGF sel inv   time: %lg\n",t_inv);


  printf("Residual norm: %e\n", solver->residualNorm(x, b));
  printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));

  /*for (int i = 0; i < nrhs*ns*nt; i++){
    printf("x[%d] = %f\n", i, b[i]);
  }*/

  // create file with solution vector
  std::string sol_x_file_name = base_path + "/x_sol_RGF"  + "_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
  std::ofstream sol_x_file(sol_x_file_name,    std::ios::out | std::ios::trunc);

  for (i = 0; i < n; i++) {
    sol_x_file << x[i] << std::endl;
    // sol_x_file << x[i] << std::endl; 
  }

  sol_x_file.close();

  std::string log_file_name = base_path + "/log_RGF_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
  std::ofstream log_file(log_file_name);
  log_file << ns << std::endl;
  log_file << nt << std::endl;
  log_file << nb << std::endl;
  log_file << no << std::endl;
  log_file << n << std::endl;
  log_file << nnz << std::endl;
  log_file << "RGF" << std::endl;
  log_file << log_det << std::endl;
  log_file << "0.0" << std::endl;
  log_file << t_factorise2 << std::endl;
  log_file << t_solve << std::endl;
  log_file << t_inv << std::endl;
  log_file << flops_factorize << std::endl;
  log_file << flops_solve << std::endl;
  log_file << flops_inv << std::endl;

  log_file.close(); 

    // print/write diag 
  string sel_inv_file_name = base_path +"/RGF_sel_inv_ns"+to_string(ns)+"_nt"+to_string(nt)+"_nb"+ nb_s + "_no" + no_s +".dat";
  cout << sel_inv_file_name << endl;
  ofstream sel_inv_file(sel_inv_file_name,    ios::out | ::ios::trunc);
  
    for (int i = 0; i < n; i++){
      sel_inv_file << invDiag[i] << endl;
    }

  sel_inv_file.close();
  cout << "after writing file " << endl;
  
  // free memory
  delete[] invDiag;
  delete solver;
  delete[] ia;
  delete[] ja;
  delete[] a;
  delete[] b;
  delete[] x;

  #if 0
  #endif
    
  return 0;


  }
