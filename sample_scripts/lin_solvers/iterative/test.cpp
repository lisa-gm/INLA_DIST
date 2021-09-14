// test eigen library 

/*#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
} */

#include <Eigen/Sparse>
#include <vector>
#include <iostream>

#include <vector>

using namespace Eigen;
 
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
 
void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
  int n = int(boundary.size());
  int id1 = i+j*n;
 
        if(i==-1 || i==n) b(id) -= w * boundary(j); // constrained coefficient
  else  if(j==-1 || j==n) b(id) -= w * boundary(i); // constrained coefficient
  else  coeffs.push_back(T(id,id1,w));              // unknown coefficient
}
 
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
  b.setZero();
  Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0,M_PI).sin().pow(2);
  for(int j=0; j<n; ++j)
  {
    for(int i=0; i<n; ++i)
    {
      int id = i+j*n;
      insertCoefficient(id, i-1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i+1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j-1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j+1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j,    4, coefficients, b, boundary);
    }
  }
}
 
 
int main(int argc, char** argv)
{
  if(argc!=1) {
    std::cerr << "Error: expected no input argument.\n";
    return -1;
  }
  
  int n = 3;  // size of the image
  int m = n*n;  // number of unknowns (=number of pixels)
 
  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);
 
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  std::cout << "Dim A : " << A.rows() << " " << A.cols() << std::endl;
  std::cout << "non-zeros A : " << A.nonZeros() << std::endl;
  std::cout << "inner size A : " << A.innerSize() << std::endl;
  std::cout << "outer size A : " << A.outerSize() << std::endl;

  std::cout << " A is compressed : " << A.isCompressed() << std::endl;
  A.makeCompressed();
  std::cout << " A is compressed : " << A.isCompressed() << std::endl;


  std::cout << "Dim A : " << A.rows() << " " << A.cols() << std::endl;
  std::cout << "storage index A : " << A.innerIndexPtr() << std::endl;
  std::cout << "inner size A : " << A.innerSize() << std::endl;
  std::cout << "outer size A : " << A.outerSize() << std::endl;

  int nnz = A.nonZeros();
  for (int i = 0; i < nnz; i++){
  		// innerIndexPtr is the "long one" equivalent to armadillos row index
  		// outIndexPtr is the "short one" equivalent to armadillos column index (in CSC format)
  		std::cout << A.innerIndexPtr()[i] << " " <<  A.valuePtr()[i] << std::endl;
  	
  }

  for (int i = 0; i < n; i++){
  	  std::cout << A.outerIndexPtr()[i] << std::endl;
  	}


  int rows = 3;
  int cols = 3;

  Eigen::SparseMatrix<double> B(rows, cols);
  // 
  n = 3;
  nnz = 2;
  int row_index[nnz] = {0, 2};
  int col_ptr[n]   = {0, 2, 2};
  double val[nnz]   = {10, 11};


  int outerIndexPtr[cols+1] = {0, 2, 2, 2};
  int innerIndices[nnz] =  {0, 2};
  double values[nnz] = {10, 11};
  Map<SparseMatrix<double> > sm1(rows,cols,nnz,outerIndexPtr, // read-write
                               innerIndices,values);
  //Map<const SparseMatrix<double> > sm2(...);                  // read-only


   MatrixXd dMat;
   dMat = MatrixXd(sm1);

   std::cout << "Here is the matrix m:\n" << dMat << std::endl;


  //typedef Map<SparseMatrix> cscMap;

  //Map<Eigen::SparseMatrix<double> > B(n,n,nnz,col_ptr,row_index,val);

  //B.Map(n, n, nnz, *col_ptr, *row_index, *val);




  /*for(int i = 0; i < nnz; i++){
  		B.innerIndexPtr()[i] = row_index[i];
  		B.valuePtr()[i]      = val[i];
  }

  for(int i = 0; i < n; i++){
  	B.outerIndexPtr()[i] = col_ptr[i];
  }*/


 
  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side
 
  return 0;
}






//Eigen::SimplicialLLT <Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholesky;
//cholesky.compute(XX);