// read in partial matrices, do Cholesky factorisation
// build preconditioner from partial matrices

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <vector>

using namespace Eigen;
using namespace std;

typedef SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

// attention expects complete matrix (not just lower triangular part)
SpMat readCSC(std::string filename)
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

  Map<SparseMatrix<double> > A(n,n,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

  return A;
} 




int main(int argc, char** argv)
{

	 if (argc != 1 + 1) {
    	std::cerr << "build preconditioner : path_to_folder_file" << std::endl;
    	std::cerr << "[string:path_to_file]  file containing matrix in CSC format (as a list)" << std::endl;      
      
    exit(1);
  } 

  // read matrix from file
  std::string path_to_file      = argv[1];
  SpMat A = readCSC(path_to_file);

  /*MatrixXd dMat;
  dMat = MatrixXd(A_lower);
  std::cout << "Here is the matrix m:\n" << dMat << std::endl;*/

  // call Cholesky factorisation
  Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int> > cholesky;  // performs a Cholesky factorization of A
  cholesky.analyzePattern(A);
  cholesky.factorize(A);
  SpMat L = cholesky.matrixL();

  std::cout << Eigen::MatrixXd(A) << std::endl;
  std::cout << Eigen::MatrixXd(L) << std::endl;





}