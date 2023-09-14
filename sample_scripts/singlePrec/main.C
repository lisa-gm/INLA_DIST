#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "magma_v2.h"
#include "cusolverDn.h"

#include <armadillo>

#include <Eigen/Core>
#include <Eigen/Dense>

#define MAGMA

#if 1
typedef float T;
#define assign_T(val) val
#else
typedef double T;
#define assign_T(val) val
#endif

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

using namespace std;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
        abort();
    }
}

MatrixXd read_matrix(const string filename,  int n_row, int n_col){

    arma::mat X(n_row, n_col);
    X.load(filename, arma::raw_ascii);
    //X.print();

    return Eigen::Map<MatrixXd>(X.memptr(), X.n_rows, X.n_cols);
}

/*
magma_int_t magma_tpotrf_expert(){
    
    magma_dpotrf_expert_gpu_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info,
    magma_mode_t mode, // MagmaHybrid MagmaNative
    magma_int_t nb, // block size -> 
    magma_int_t recnb, // within nb -> double: 64/128, float 128/256. put recnb 32. then increase nb.
    void* host_work,   magma_int_t *lwork_host, -> lwork_host set to negative number, in code
    void* device_work, magma_int_t *lwork_device, -> lwork_device set to negative, in code
    magma_event_t events[2], magma_queue_t queues[2] ) -> 
}
*/


// new magma -- double precision
magma_int_t magma_tpotrf_gpu(magma_uplo_t magma_uplo, magma_int_t n, magmaDouble_ptr dA, magma_int_t ldda, magma_int_t * info)
{
    magma_int_t potrfErr = magma_dpotrf_gpu(magma_uplo,n,dA,ldda,info);    
    return potrfErr;
}	


magma_int_t magma_tpotrf_gpu(magma_uplo_t magma_uplo, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t* info )	
{
    magma_int_t potrfErr = magma_spotrf_gpu(magma_uplo,n,dA,ldda,info);
    return potrfErr;
}

                            
// TODO use different magma call for single precision
void tpotrf_magma_dev(char uplo,int n,T *a,int lda,int *info)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    printf("in magma ptorf\n");

    //magma_int_t potrfErr = magma_dpotrf_gpu(magma_uplo,n,a,lda,info); // hybrid version
    //magma_int_t potrfErr = magma_dpotrf_native(magma_uplo,n,a,lda,info); // gpu only version  
    magma_int_t potrfErr = magma_tpotrf_gpu(magma_uplo,n,a,lda,info);

    if(potrfErr != 0){
        std::cout << "magma potrf error = " << potrfErr << std::endl;
        exit(1);
    }

}

void ttrsm_dev(magma_side_t magma_side, magma_uplo_t magma_uplo, magma_trans_t magma_trans, magma_diag_t magma_diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb, magma_queue_t queue)
{
    magma_dtrsm(magma_side, magma_uplo, magma_trans, magma_diag, m, n, alpha, a, lda, b, ldb, queue);
}

void ttrsm_dev(magma_side_t magma_side, magma_uplo_t magma_uplo, magma_trans_t magma_trans, magma_diag_t magma_diag, int m, int n, float alpha, float *a, int lda, float *b, int ldb, magma_queue_t queue)
{
    magma_strsm(magma_side, magma_uplo, magma_trans, magma_diag, m, n, alpha, a, lda, b, ldb, queue);
}

void ttrsm_magma_dev(char side, char uplo, char trans, char diag, int m, int n, T alpha, T *a, int lda, T *b, int ldb, magma_queue_t queue){
    
    magma_side_t magma_side = magma_side_const(side);
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    magma_trans_t magma_trans = magma_trans_const(trans);
    magma_diag_t magma_diag = magma_diag_const(diag);

    ttrsm_dev(magma_side, magma_uplo, magma_trans, magma_diag, m, n, alpha, a, lda, b, ldb, queue);
}


void magma_tgemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, double alpha,
                magmaDouble_const_ptr dA, magma_int_t ldda, magmaDouble_const_ptr dB, magma_int_t lddb,
                double beta, magmaDouble_ptr dC, magma_int_t lddc, magma_queue_t queue)
{
    magma_dgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);	

}


void magma_tgemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, float alpha,
                magmaFloat_const_ptr dA, magma_int_t ldda, magmaFloat_const_ptr dB, magma_int_t lddb, float beta, magmaFloat_ptr dC,
                magma_int_t lddc, magma_queue_t queue)
{
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);	
}	

void tgemm_magma_dev(int n, T alpha, T beta, T *A, T* B, T* C, magma_queue_t queue){

    //magma_trans_t magma_transa = magma_trans_const(transa);
    //magma_trans_t magma_transb = magma_trans_const(transb);
    magma_trans_t magma_transa = magma_trans_const('N');
    magma_trans_t magma_transb = magma_trans_const('N');

    magma_tgemm(magma_transa, magma_transb, n, n, n, alpha, A, n, B, n, beta, C, n, queue);

}

// single precision 
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    //printf("in single precision.\n");
    cublasStatus_t cublasError = cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasError;
}

// double precision
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    //printf("in double precision.\n");
    cublasStatus_t cublasError = cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasError;
}

// assuming dense square matrices of size n x n, all scalars assumed to be 1
void tgemm_cuda_dev(int n, T alpha, T beta, T *A, T* B, T* C, cudaStream_t stream)
{
    cublasHandle_t handle = NULL;

    cublasStatus_t cublasError = cublasCreate(&handle);
    if(cublasError != 0){
        printf("cublasError not Zero!\n");
        exit(1);
    }

    cublasError = cublasSetStream(handle, stream);
    if(cublasError != 0){
        printf("cublasError set Stream! cublasError = %d\n", cublasError);
        exit(1);
    }   
    cublasError = cublasTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
    
    if(cublasError != 0){
        printf("cuSolverError dgemm error! cuSolverError = %d\n", cublasError);
        exit(1);
    }   

}

void dpotrf_cuda_dev(char uplo,int n,T *a,int lda, cudaStream_t stream, int *info)
{

    //static cusolverDnHandle_t handle;
    cusolverDnParams_t params;
    cusolverDnHandle_t handle = NULL;

    //int dev_size = 0;
    static size_t dev_size;
    static size_t host_size;
    double* mem_cuda_dev = 0;
    double* mem_cuda_host = 0;

    int* info_cuda = NULL;
    cudaMalloc((void**)&info_cuda, sizeof(int));

    cublasFillMode_t uplo_cuda = CUBLAS_FILL_MODE_LOWER;

    cudaDataType dataTypeA;
    cudaDataType computeType;

    if(sizeof(T) == 8){
        //printf("Template T is double.\n");
        dataTypeA   = CUDA_R_64F;
        computeType = CUDA_R_64F;
    } else if (sizeof(T) == 4){
        //printf("Template T is float.\n");
        dataTypeA   = CUDA_R_32F;
        computeType = CUDA_R_32F;
    } else {
        printf("invalid type T!\n");
        exit(1);
    }

    /*
    if(uplo == 'L'){
    	printf("take lower.\n");
        uplo_cuda = CUBLAS_FILL_MODE_LOWER;
    } else if(uplo =='U') {
        uplo_cuda = CUBLAS_FILL_MODE_UPPER;
    } else {
        printf("invalid potrf argmument uplo: %c\n", uplo);
        exit(1);
    }
    */

	cusolverStatus_t cuSolverError = cusolverDnCreate(&handle);
    if(cuSolverError != 0){
        printf("cuSolverError not Zero!\n");
        exit(1);
    }

    cuSolverError = cusolverDnCreateParams(&params);
    if(cuSolverError != 0){
        printf("cuSolverError not Zero after params!\n");
        exit(1);
    }

	//cuSolverError = cusolverDnDpotrf_bufferSize(handle, uplo_cuda, n, a, lda, &dev_size);
    cuSolverError = cusolverDnXpotrf_bufferSize(
            handle, params, uplo_cuda, n, dataTypeA, a, lda, computeType, &dev_size, &host_size);
   if(cuSolverError != 0){
        printf("cuSolverError buffer size allocation! cuSolverError = %d\n", cuSolverError);
        exit(1);
    }

    cudaMalloc((void**)&mem_cuda_dev, dev_size * sizeof(double));
    cudaMallocHost((void**)&mem_cuda_host, host_size * sizeof(double));

    //printf("n = %d, lda = %d, uplo = %c\n", n, lda, uplo_cuda);

    cuSolverError = cusolverDnSetStream(handle, stream);
    if(cuSolverError != 0){
        printf("cuSolverError set Stream! cuSolverError = %d\n", cuSolverError);
        exit(1);
    }    

    cuSolverError = cusolverDnXpotrf(
            handle, params, uplo_cuda, n, dataTypeA, a, lda, computeType, 
            mem_cuda_dev, dev_size, mem_cuda_host, host_size, info_cuda);
    //cuSolverError = cusolverDnDpotrf(handle, uplo_cuda, n, a, lda, mem_cuda_dev, dev_size, info_cuda);


    if(cuSolverError != 0){
        printf("cuSolverError potrf! cuSolverError : %d, lda = %d\n", cuSolverError, lda);
        exit(1);
    }

    int info_host;
    cudaMemcpyAsync(&info_host, info_cuda, sizeof(int), cudaMemcpyDeviceToHost, stream);

    if(info_host != 0){
        printf("cuSolverError potrf info not zero! info = %d, lda = %d\n", info_host, lda);
        exit(1);
    }

}


void get_logDet(T* Q_host, int n, double& logDet){
    // assume dense matrix, k-th diagonal entry: k*(n+1)
    double temp = 0.0;

    for(int i=0; i<n; i++){
        temp += log(Q_host[i*(n+1)]);
    }

    logDet = 2*temp;
}


int main(int argc, char* argv[])
{

	if(argc != 1 + 1){
		std::cout << "wrong number of input parameters. " << std::endl;
        std::cerr << "main : n" << std::endl;
        exit(1);
	}

    std::cout << "reading in example. " << std::endl;
    size_t n = atoi(argv[1]);

    MatrixXd Q(n,n);
    //std::string filename = "../gpu_potrf/Qst_firstBlock_" + to_string(n) + "_" + to_string(n) + ".txt";
	//Q = read_matrix(filename, n, n);

    double t_genMat = - omp_get_wtime();
    Q = MatrixXd::Random(n,n);
    Q = Q*Q.transpose();
    std::cout << "Q :\n" << Q.block(0,0,min(10, (int) n), min(10, (int) n)) << std::endl;

    t_genMat += omp_get_wtime();
    printf("time spent generate matrix : %f\n", t_genMat);

    if(sizeof(T) == 8){
        printf("Template T is double.\n");
    } else if (sizeof(T) == 4){
        printf("Template T is float.\n");
    } else {
        printf("invalid type T!\n");
        exit(1);
    }

    T alpha = 1.0;
    T beta = 1.0;

    T* Id_dev;
    T* Q_dev1;
    T* Q_dev2;

    T* Id_host;
    T* Q_host;
    T* L_host;

    checkCudaErrors(cudaMalloc((void**)&Id_dev, n*n*sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&Q_dev1,n*n*sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&Q_dev2,n*n*sizeof(T)));

    checkCudaErrors(cudaMallocHost((void**)&Id_host, n*n*sizeof(T)));
    checkCudaErrors(cudaMallocHost((void**)&Q_host,n*n*sizeof(T)));
    checkCudaErrors(cudaMallocHost((void**)&L_host,n*n*sizeof(T)));

    for(int i=0; i<n*n; i++){
        Id_host[i] = 0.0;
        int k = i / n;
        if(i == k*(n+1)){
            //printf("i = %d\n", i);
            Id_host[i] = 2.0;
        }
    }

    // use pinned memory ...
    for(int i=0; i<n*n; i++){
        Q_host[i] = (T) Q.data()[i];
    }

    MatrixXd L1(n,n);
    MatrixXd L2(n,n);

    int info1;
    int info2;

    cudaStream_t compute_stream;
    cudaStream_t copy_stream;
    cudaEvent_t  compute_gpu;
    cudaEvent_t  copy_gpu;

    magma_queue_t magma_compute_queue;

    checkCudaErrors(cudaStreamCreate(&compute_stream));
    checkCudaErrors(cudaStreamCreate(&copy_stream));
    checkCudaErrors(cudaEventCreate(&compute_gpu));
    checkCudaErrors(cudaEventCreate(&copy_gpu));

    checkCudaErrors(cudaMemcpy(Id_dev, Id_host, n*n*sizeof(T), cudaMemcpyHostToDevice));

    double t_gpu_calls = - omp_get_wtime();
#ifdef MAGMA
    printf("using MAMGA.\n");
    magma_init();
    
    magma_device_t device;
    magma_getdevice(&device);
    magma_queue_create_from_cuda(device, compute_stream, NULL, NULL, &magma_compute_queue);
    
   checkCudaErrors(cudaMemcpy(Q_dev1, Q_host, n*n*sizeof(T), cudaMemcpyHostToDevice));

	tpotrf_magma_dev('L', n, Q_dev1, n, &info1);
    // TODO: check if this works for single preicision
	checkCudaErrors(cudaMemcpy(L_host, Q_dev1, n*n*sizeof(T), cudaMemcpyDeviceToHost));

	printf("\nL_host: ");
	for(int i=0; i<n*n; i++){
		printf(" %f", L_host[i]);
	}
	printf("\n");
 
    cudaDeviceSynchronize();
    ttrsm_magma_dev('R', 'L', 'T', 'N', n, n, alpha, Q_dev1, n, Id_dev, n, magma_compute_queue);

	checkCudaErrors(cudaMemcpy(Id_host, Id_dev, n*n*sizeof(T), cudaMemcpyDeviceToHost));
	printf("\nQ2 after ttrsm: ");
	for(int i=0; i<n*n; i++){
		printf(" %f", Id_host[i]);
	}
	printf("\n");

    tgemm_magma_dev(n, alpha, beta, Id_dev, Id_dev, Q_dev2, magma_compute_queue);

	printf("\nQ2 after gemm: ");
	for(int i=0; i<n*n; i++){
		printf(" %f", Id_host[i]);
	}
	printf("\n");
	checkCudaErrors(cudaMemcpy(Id_host, Q_dev2, n*n*sizeof(T), cudaMemcpyDeviceToHost));


    double logDet;
    get_logDet(L_host, n, logDet);
    printf("logDet       : %f\n", logDet);

    magma_finalize();
#else
    printf("using CUDA.\n");
    printf("entering loop\n\n");
    for(int i=0; i<10; i++){
        //checkCudaErrors(cudaMemcpyAsync(Q_dev1, Id_host, n*n*sizeof(double), cudaMemcpyHostToDevice, compute_stream));
        checkCudaErrors(cudaMemcpyAsync(Q_dev1, Q_host, n*n*sizeof(T), cudaMemcpyHostToDevice, copy_stream));
       
        // perform computation in compute stream but need to wait for copy_stream
        checkCudaErrors(cudaEventRecord(copy_gpu, compute_stream)); // want compute_stream to wait
        checkCudaErrors(cudaStreamWaitEvent(compute_stream, copy_gpu, 0));	

        dpotrf_cuda_dev('L', n, Q_dev1, n, compute_stream, &info1);
        tgemm_cuda_dev(n, alpha, beta, Id_dev, Q_dev1, Q_dev2, compute_stream);
        
        // perform copying in copy_stream
        checkCudaErrors(cudaEventRecord(compute_gpu, compute_stream)); // want copy_stream to wait
        checkCudaErrors(cudaStreamWaitEvent(copy_stream, compute_gpu, 0));	
        //checkCudaErrors(cudaMemcpyAsync(L_host, Q_dev2, n*n*sizeof(T), cudaMemcpyDeviceToHost, copy_stream));
        checkCudaErrors(cudaMemcpyAsync(L_host, Q_dev1, n*n*sizeof(T), cudaMemcpyDeviceToHost, copy_stream));

        cudaDeviceSynchronize();

        double logDet;
        get_logDet(L_host, n, logDet);
        printf("logDet       : %f\n", logDet);

    }
#endif

    t_gpu_calls += omp_get_wtime();
    printf("time spent cuda/magma call: %f\n", t_gpu_calls);

    Eigen::LLT<MatrixXd> lltOfA(Q); // compute the Cholesky decomposition of A
    MatrixXd L = lltOfA.matrixL();

    printf("logDet Eigen : %f\n", 2*(L.diagonal().array().log().sum()));




    /*for(int i=0; i<n*n; i++){
        L1.data()[i] = L_host[i];
    }*/

	
	printf("after trsm on host: ");
	for(int i=0; i<n*n; i++){
		printf(" %f", Id_host[i]);
	}
	printf("\n");
	

    //printf("after computation.\n");

	//MatrixXd L1_lower = L1.triangularView<Eigen::Lower>();

	//std::cout << "L1 : \n" << L1_lower << std::endl;

	//std::cout << "norm(Q - L*L^T) : " << (Q - L1_lower*L1_lower.transpose()).norm() << std::endl;


    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);

    cudaEventDestroy(compute_gpu);
    cudaEventDestroy(copy_gpu);

	cudaFree(Q_dev1);
	cudaFree(Q_dev2);

    cudaFreeHost(Id_host);
    cudaFreeHost(Q_host);
    cudaFreeHost(L_host);
  

	return 1;

}
