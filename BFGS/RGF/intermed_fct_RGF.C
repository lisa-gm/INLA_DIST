// intermediate function

#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <armadillo>

#include "RGF.H"

using namespace std;


#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

void call_RGF_solver(size_t ns, size_t nt, size_t nb, size_t no, arma::sp_mat& Qxy, arma::vec bxy, string base_path){
	
  int n = size(Qxy)[0];

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
  double t_factorise; double t_solve; double t_inv;
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

  t_factorise = get_time(0.0);
  //solver->solve_equation(GR);
  double flops_factorize = solver->factorize();
  t_factorise = get_time(t_factorise);


  double log_det = solver->logDet();
  printf("logdet: %f\n", log_det);

  printf("flops factorize: %f\n", flops_factorize);

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


  printf("RGF factorise time: %lg\n",t_factorise);
  printf("RGF solve     time: %lg\n",t_solve);
  printf("RGF sel inv   time: %lg\n",t_inv);


  printf("Residual norm: %e\n", solver->residualNorm(x, b));
  printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));

  /*for (int i = 0; i < nrhs*ns*nt; i++){
    printf("x[%d] = %f\n", i, b[i]);
  }*/

  #if 0 

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
  log_file << t_factorise << std::endl;
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

  #endif
  
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

}