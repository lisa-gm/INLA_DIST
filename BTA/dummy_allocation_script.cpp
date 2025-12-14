#include <iostream>
#include <Eigen/Dense>  // Main Eigen header
#include <Eigen/SparseCore>  // Required for sparse matrices
#include <unsupported/Eigen/KroneckerProduct>  // For kroneckerProduct
#include <chrono>
#include <unistd.h> // For memory usage tracking

// compile with
// g++ -std=c++14 -O3 -I/users/lgaedkem/applications/eigen -o eigen_demo dummy_allocation_script.cpp


typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

void construct_Q_spat_temp(SpMat& Qst, SpMat& A, SpMat& B, SpMat& C, SpMat& D){

    double c1 = 5.0;
    SpMat tmp1 = c1 * Eigen::KroneckerProductSparse<SpMat, SpMat>(A, B);
    SpMat tmp2 = pow(c1, 2) * Eigen::KroneckerProductSparse<SpMat, SpMat>(C, B);

    Qst = tmp1 + tmp2;
    Qst.makeCompressed();

    printf("after kronecker\n");

}

SpMat generateRandomSparseMatrix(int rows, int cols, double sparsity) {
    Eigen::MatrixXd dense = Eigen::MatrixXd::Random(rows, cols); // Generate a dense random matrix
    dense = (dense.array() < sparsity).select(dense, 0); // Apply sparsity by zeroing out elements
    return dense.sparseView(); // Convert to sparse matrix
}

int main() {
    // Scalable test parameters
    const int base_size = 100; // Base size for matrices
    const int max_scale = 4; // Number of scaling steps

    for (int scale = 1; scale <= max_scale; ++scale) {
        int rows_A = base_size * scale;
        int cols_A = base_size * scale;
        int rows_B = base_size * scale;
        int cols_B = base_size * scale;

        // Generate random sparse matrices
        SpMat A = generateRandomSparseMatrix(rows_A, cols_A, 0.02); // 20% sparsity
        SpMat B = generateRandomSparseMatrix(rows_B, cols_B, 0.01);
        SpMat C = generateRandomSparseMatrix(rows_A, cols_A, 0.02);
        SpMat D = generateRandomSparseMatrix(rows_B, cols_B, 0.01);

        // Output matrix
        SpMat Qst(rows_A * rows_B, cols_A * cols_B);

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        construct_Q_spat_temp(Qst, A, B, C, D);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Scale: " << scale
                  << ", Matrix size: " << rows_A << "x" << cols_A
                  << ", Time: " << elapsed.count() << " seconds\n";

        // Verify output dimensions
        std::cout << "Qst dimensions: " << Qst.rows() << "x" << Qst.cols() << "\n";
    }

    return 0;
}


