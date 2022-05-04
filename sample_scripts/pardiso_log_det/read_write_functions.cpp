// read & write functions

#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// require armadillo for read dense matrix for now
#include <armadillo>

#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

// attention expects complete matrix (not just lower triangular part)
SpMat readCSC(std::string filename){
  int n_rows; int n_cols;
  int nnz;

  fstream fin(filename, ios::in);
  fin >> n_rows;
  fin >> n_cols;
  fin >> nnz;

  int* innerIndices;
  int* outerIndexPtr;
  double* values; 

  // allocate memory
  innerIndices  = new int [nnz];
  outerIndexPtr = new int [n_cols+1];
  values        = new double [nnz];

   // allocate memory
  /*int innerIndices[nnz];
   std::cout << "inner index ptr" << std::endl;
  int outerIndexPtr[n_cols+1];
  double val[nnz];
     std::cout << "inner index ptr" << std::endl;
  double values[nnz];*/

  for (int i = 0; i < nnz; i++){
    fin >> innerIndices[i];
  }

  for (int i = 0; i < n_cols+1; i++){
    fin >> outerIndexPtr[i];}

  for (int i = 0; i < nnz; i++){
    fin >> values[i];}

  fin.close();

  // 
  SpMat A = Eigen::Map<Eigen::SparseMatrix<double> >(n_rows,n_cols,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

  delete[] innerIndices;
  delete[] outerIndexPtr;
  delete[] values;


  return A;
} 

// expects indices for lower triangular matrix
SpMat read_sym_CSC(std::string filename)
{

  int n;
  int nnz;

  fstream fin(filename, ios::in);
  fin >> n;
  fin >> n;
  fin >> nnz;

   // allocate memory
  int outerIndexPtr[n+1];
  int innerIndices[nnz];
  double values[nnz];

  for (int i = 0; i < nnz; i++){
    fin >> innerIndices[i];}

  for (int i = 0; i < n+1; i++){
    fin >> outerIndexPtr[i];}

  for (int i = 0; i < nnz; i++){
    fin >> values[i];}

  fin.close();

  //
  SpMat A_lower =  Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

  // TODO: more efficient way to do this?
  SpMat A = A_lower.selfadjointView<Lower>();
  //std::cout << "input A : " << std::endl;
  //std::cout << A << std::endl;

  return A;
} 


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


// for now use armadillo ... do better once we switch to binary

MatrixXd read_matrix(const string filename,  int n_row, int n_col){

    arma::mat X(n_row, n_col);
    X.load(filename, arma::raw_ascii);
    //X.print();

    return Eigen::Map<MatrixXd>(X.memptr(), X.n_rows, X.n_cols);
}



void file_exists(std::string file_name)
{
    if (std::fstream{file_name}) ;
    else {
      std::cerr << file_name << " couldn\'t be opened (not existing or failed to open)\n"; 
      exit(1);
    }
    
}


void read_matrix_binary(){

}

void read_sparse_CSC_binary(){
	
}


/* ==================================================================================== */
// write functions


void write_vector(std::string full_file_name, Eigen::VectorXd x, int n){

  ofstream sol_file(full_file_name,    ios::out | ::ios::trunc);
  
  for (int i = 0; i < n; i++){
    sol_file << x[i] << endl;
  }
  sol_file.close();

  std::cout << "wrote to file : " << full_file_name << std::endl;
}

void write_log_file(std::string full_file_name, int ns, int nt, int nb, int no, int nnz, \
                    std::string solver_type, double log_det, \
                    double t_sym_fact, double t_factorise, double t_solve, double t_inv, \
                    double flops_factorize, double flops_solve, double flops_inv){

  int n = ns*nt + nb;

  std::ofstream log_file(full_file_name);
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
}


/*
int main(int argc, char** argv)
{

	 if (argc != 3 + 1) {
    	std::cerr << "test read functions : path_to_folder_file" << std::endl;
    	std::cerr << "[string:path_to_file]  file containing matrix in CSC format (as a list)" << std::endl;      
      std::cerr << "[integer]: n_rows" << std::endl;     
		  std::cerr << "[integer]: n_cols" << std::endl;  

    exit(1);
  } 

  // read matrix from file
  std::string path_to_file      = argv[1];
  int n_rows					= atoi(argv[2]);
  int n_cols 					= atoi(argv[3]);
  
  MatrixXd B = read_matrix(path_to_file, n_rows, n_cols);
  std::cout << "B : " << std::endl;
  std::cout << B << std::endl; 


  SpMat A = read_sym_CSC(path_to_file);
  std::cout << "A : " << std::endl;
  std::cout << MatrixXd(A) << std::endl; */

  // SpMat A = read_sym_CSC(path_to_file);

  // return 0;

//
