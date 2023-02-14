// read & write functions

#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

// std::setwd print out
#include <iomanip>

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
  int* outerIndexPtr;
  int* innerIndices;
  double* values;

  outerIndexPtr = new int[n+1];
  innerIndices  = new int[nnz];
  values        = new double[nnz];

  /*int outerIndexPtr[n+1];
  int innerIndices[nnz];
  double values[nnz];*/

  for (int i = 0; i < nnz; i++){
    fin >> innerIndices[i];
  }

  for (int i = 0; i < n+1; i++){
    fin >> outerIndexPtr[i];
  }

  for (int i = 0; i < nnz; i++){
    fin >> values[i];
  }

  fin.close();

  //
  SpMat A_lower =  Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

  // TODO: more efficient way to do this?
  SpMat A = A_lower.selfadjointView<Lower>();
  //std::cout << "input A : " << std::endl;
  //std::cout << A << std::endl;

  delete [] innerIndices;
  delete [] outerIndexPtr;
  delete [] values;

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
    sol_file << std::setprecision(15) << x[i] << endl;
  }
  sol_file.close();

  std::cout << "wrote to file : " << full_file_name << std::endl;
}

void write_matrix(std::string full_file_name, Eigen::MatrixXd A){
   
    ofstream sol_file(full_file_name);
    if(sol_file){
    	sol_file << std::setprecision(15) << A;
    	sol_file.close();
        std::cout << "wrote to file : " << full_file_name << std::endl;
    } else {
    	std::cout << "There was an error writing " << full_file_name << " to file." << std::endl;
        exit(1);
    }
}

#if 1
// only takes lower triangular part of matrix & CSC format or
// upper triangular & CSR format
void write_sym_CSC_matrix(std::string full_file_name, SpMat A){

    SpMat A_lower = A.triangularView<Lower>();

    int n = A_lower.cols();
    int nnz = A_lower.nonZeros();
   
    ofstream sol_file(full_file_name);
    if(sol_file){
        sol_file << n << "\n";
        sol_file << n << "\n";
        sol_file << nnz << "\n";

        for (int i = 0; i < nnz; i++){
            sol_file << A_lower.innerIndexPtr()[i] << "\n";
        }   

        for (int i = 0; i < n+1; i++){
             sol_file << A_lower.outerIndexPtr()[i] << "\n";
        }     

        for (int i = 0; i < nnz; i++){
            sol_file << std::setprecision(15) << A_lower.valuePtr()[i] << "\n";
        }

        sol_file.close();
        std::cout << "wrote to file : " << full_file_name << std::endl;
    } else {
        std::cout << "There was an error writing " << full_file_name << " to file." << std::endl;
        exit(1);
    }
}

#endif

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


void create_folder(std::string dir_name){
    
    char dir_name_char[dir_name.length() + 1]; 
    strcpy(dir_name_char, dir_name.c_str());

    if(mkdir(dir_name_char, 0777) == -1){
            cerr << "Error :  " << strerror(errno) << endl;
    } else {
            cout << "Results parameters directory created under name : " << dir_name << std::endl;
    }

}


// create folder -> don't overwrite, create new one if already exists
/*
std::string create_folder(std::string initial_dir_name){
    
    int n = 0;
    bool dir_created = false;
    string dir_name = initial_dir_name;

    while(!dir_created){

        char dir_name_char[dir_name.length() + 1];
        strcpy(dir_name_char, dir_name.c_str());
        if(mkdir(dir_name_char, 0777) == -1){
            cerr << "Error :  " << strerror(errno) << endl;
            n += 1;
            dir_name = initial_dir_name + "_" + to_string(n);
        } else {
            dir_created = true;
            cout << "Results parameters directory created under name : " << dir_name << std::endl;
            return dir_name;
	}

    }

    return "Problem. Something went wrong in create folder."; 	
}
*/

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
