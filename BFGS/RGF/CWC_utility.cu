#include <stdio.h>
#include "Types.H"
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "cuda.h"
#include "magma_v2.h"

#ifndef max_stream
#define max_stream 16
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 16
#endif

static volatile size_t c_memory = 0;

extern "C"
void set_gpu(int dev,char *gpu_string){
     struct cudaDeviceProp dprop;
     cudaSetDevice(dev);
     cudaGetDeviceProperties(&dprop, dev);
     strcpy(gpu_string,dprop.name);	
}

extern "C"
void cublas_init(void **handle){
     cublasCreate((cublasHandle_t*)handle);
}

extern "C"
void cublas_finalize(void *handle){
     cublasDestroy((cublasHandle_t)handle);
}

extern "C"
void cusparse_init(void **handle){
     cusparseCreate((cusparseHandle_t*)handle);
}

extern "C"
void cusparse_finalize(void *handle){
     cusparseDestroy((cusparseHandle_t)handle);
}

extern "C"
size_t allocate_data_on_device(void **data,size_t size_data){

     cudaError_t mem_error;  

     mem_error = cudaMalloc(data,size_data);

     if(mem_error!=cudaSuccess){
         printf("CPU wants to allocate %e MBytes on the device, but already %e MBytes are in use\n",size_data/1e6,c_memory/1e6);
	 exit(0);
     }else{
         c_memory = c_memory+size_data;
     }

     return c_memory;
}

extern "C"
void deallocate_data_on_device(void *data){
     cudaFree(data);
}

extern "C"
size_t deallocate_data_on_dev(void *data,size_t size_data){

     cudaFree(data);

     c_memory = c_memory-size_data;

     return c_memory;
}

extern "C"
void copy_data_to_device(void *host_data,void *device_data,int N,int M,size_t size_element){
     cublasSetMatrixAsync(N,M,size_element,host_data,N,device_data,N,NULL);
}

extern "C"
void memcpy_to_device(void *host_data,void *device_data,size_t size_element){
     cudaMemcpyAsync(device_data,host_data,size_element,cudaMemcpyHostToDevice,NULL);
}

extern "C"
void copy_data_to_host(void *host_data,void *device_data,int N,int M,size_t size_element){
     cublasGetMatrixAsync(N,M,size_element,device_data,N,host_data,N,NULL);
}

extern "C"
void memcpy_to_host(void *host_data,void *device_data,size_t size_element){
     cudaMemcpyAsync(host_data,device_data,size_element,cudaMemcpyDeviceToHost,NULL);
}

extern "C"
void dgemm_on_dev(void *handle,char transa,char transb,int m,int n,int k,double alpha,\
      		  double *A,int lda,double *B,int ldb,double beta,double *C, int ldc){
     
     cublasOperation_t opA,opB;

     if(transa=='N'){
	opA = CUBLAS_OP_N;
     }
     if(transa=='C'){
	opA = CUBLAS_OP_C;
     }
     if(transa=='T'){
	opA = CUBLAS_OP_T;
     }

     if(transb=='N'){
	opB = CUBLAS_OP_N;
     }
     if(transb=='C'){
	opB = CUBLAS_OP_C;
     }
     if(transb=='T'){
	opB = CUBLAS_OP_T;
     }

     cublasDgemm((cublasHandle_t)handle,opA,opB,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc);
}

extern "C"
void zgemm_on_dev(void *handle,char transa,char transb,int m,int n,int k,CPX alpha,\
                  CPX *A,int lda,CPX *B,int ldb,CPX beta,CPX *C, int ldc){
  
     cublasOperation_t opA,opB;

     if(transa=='N'){
	opA = CUBLAS_OP_N;
     }
     if(transa=='C'){
	opA = CUBLAS_OP_C;
     }
     if(transa=='T'){
	opA = CUBLAS_OP_T;
     }

     if(transb=='N'){
	opB = CUBLAS_OP_N;
     }
     if(transb=='C'){
	opB = CUBLAS_OP_C;
     }
     if(transb=='T'){
	opB = CUBLAS_OP_T;
     }

     cublasZgemm((cublasHandle_t)handle,opA,opB,m,n,k,(cuDoubleComplex*)&alpha,\
                 (cuDoubleComplex*)A,lda,(cuDoubleComplex*)B,ldb,(cuDoubleComplex*)&beta,\
		 (cuDoubleComplex*)C,ldc);
}

extern "C"
void daxpy_on_dev(void *handle,int n,double alpha,double *x,int incx,double *y,int incy){
  
    cublasDaxpy((cublasHandle_t)handle,n,&alpha,x,incx,y,incy);
}

extern "C"
void zaxpy_on_dev(void *handle,int n,CPX alpha,CPX *x,int incx,CPX *y,int incy){
  
    cublasZaxpy((cublasHandle_t)handle,n,(cuDoubleComplex*)&alpha,(cuDoubleComplex*)x,\
                incx,(cuDoubleComplex*)y,incy);
}

extern "C"
void dasum_on_dev(void *handle,int n,double *x,int incx,double *result)
{
    cublasDasum((cublasHandle_t)handle,n,x,incx,result);
}

extern "C"
void zasum_on_dev(void *handle,int n,CPX *x,int incx,double *result)
{
    cublasDzasum((cublasHandle_t)handle,n,(cuDoubleComplex*)x,incx,result);
}

extern "C"
void dsum_on_dev(int n,double *x,int incx,double *result,magma_queue_t queue)
{
    double one = 1.0;
    double *y;
    int incy = 0;
    cudaMalloc(&y, 1*sizeof(double));
    cudaMemcpy(y, &one, 1*sizeof(double), cudaMemcpyHostToDevice);

    *result = magma_ddot(n, x, incx, y, incy, queue);

    cudaFree(y);
}

extern "C"
void zsum_on_dev(int n,CPX *x,int incx,CPX *result,magma_queue_t queue)
{
    magmaDoubleComplex one = {1.0, 0.0};
    magmaDoubleComplex *y;
    int incy = 0;
    cudaMalloc(&y, 1*sizeof(cuDoubleComplex));
    cudaMemcpy(y, &one, 1*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    magmaDoubleComplex magma_result = magma_zdotu(n, (magmaDoubleComplex*)x, incx, y, incy, queue);
    result = reinterpret_cast<CPX*>(&magma_result);

    cudaFree(y);
}

__global__ void d_init_variable_on_dev(double *var,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<N){
	var[idx] = 0.0;
     }	   

     __syncthreads();
}

extern "C"
void d_init_var_on_dev(double *var,int N,cudaStream_t stream){

    uint i_N = N + (BLOCK_DIM-(N%BLOCK_DIM));

    d_init_variable_on_dev<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>(var,N);
}

__global__ void d_init_eye_on_device(double *var,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<N*N){
	var[idx] = 0.0;
	if(!(idx%(N+1))){
	    var[idx] = 1.0;
	}	   
     }

     __syncthreads();
}

extern "C"
void d_init_eye_on_dev(double *var,int N,cudaStream_t stream){

    uint i_N = N*N + (BLOCK_DIM-((N*N)%BLOCK_DIM));

    d_init_eye_on_device<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>(var,N);
}

__global__ void z_init_variable_on_dev(cuDoubleComplex *var,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<N){
	var[idx].x = 0.0;
	var[idx].y = 0.0;
     }	   

     __syncthreads();
}

extern "C"
void z_init_var_on_dev(CPX *var,int N,cudaStream_t stream){

    uint i_N = N + (BLOCK_DIM-(N%BLOCK_DIM));

    z_init_variable_on_dev<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>((cuDoubleComplex*)var,N);
}

__global__ void z_init_eye_on_device(cuDoubleComplex *var,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<N*N){
	var[idx].x = 0.0;
	var[idx].y = 0.0;
	if(!(idx%(N+1))){
	    var[idx].x = 1.0;
	}	   
     }

     __syncthreads();
}

extern "C"
void z_init_eye_on_dev(CPX *var,int N,cudaStream_t stream){

    uint i_N = N*N + (BLOCK_DIM-((N*N)%BLOCK_DIM));

    z_init_eye_on_device<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>((cuDoubleComplex*)var,N);
}

__global__ void correct_diag_on_device(cuDoubleComplex *var,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if((idx<N*N)&&(!(idx%(N+1)))){
         var[idx].y = 0.0;	   
     }

     __syncthreads();
}

extern "C"
void correct_diag_on_dev(CPX *var,int N,cudaStream_t stream){

    uint i_N = N*N + (BLOCK_DIM-((N*N)%BLOCK_DIM));

    correct_diag_on_device<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>((cuDoubleComplex*)var,N);
}

__global__ void change_variable_type_on_dev(double *var1,cuDoubleComplex *var2,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<N){
	var2[idx].x = var1[idx];
	var2[idx].y = 0.0;
     }	   

     __syncthreads();
}

extern "C"
void change_var_type_on_dev(double *var1,CPX *var2,int N,cudaStream_t stream){

    uint i_N = N + (BLOCK_DIM-(N%BLOCK_DIM));

    change_variable_type_on_dev<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>(var1,(cuDoubleComplex*)var2,N);
}

__global__ void change_sign_imaginary_part_on_dev(cuDoubleComplex *var,int N){

     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<N){
	var[idx].y = -var[idx].y;
     }	   

     __syncthreads();
}

extern "C"
void change_sign_imag_on_dev(CPX *var,int N){

    uint i_N = N + (BLOCK_DIM-(N%BLOCK_DIM));

    change_sign_imaginary_part_on_dev<<< i_N/BLOCK_DIM, BLOCK_DIM >>>((cuDoubleComplex*)var,N);
}

__global__ void d_extract_diag(double *D,int *edge_i,int *index_j,double *nnz,\
	   int NR,int imin,int imax,int shift,int findx){

     int j;
     int ind_j;	   
     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<NR){
	  for(j=edge_i[idx+imin]-findx;j<edge_i[idx+imin+1]-findx;j++){
	      ind_j = index_j[j]-findx-shift-imin;
	      if((ind_j>=0)&&(ind_j<NR)){
	          //D[idx+ind_j*NR] = nnz[j].x;
	          D[idx+ind_j*NR] = nnz[j];
	      }
	  }
     }	   

     __syncthreads();
}

extern "C"
void d_extract_diag_on_dev(double *D,int *edge_i,int *index_j,double *nnz,int NR,\
     int imin,int imax,int shift,int findx,cudaStream_t stream){

    uint i_N = NR + (BLOCK_DIM-(NR%BLOCK_DIM));

    d_extract_diag<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>(D,edge_i,index_j,nnz,NR,imin,imax,shift,findx);
}

__global__ void d_extract_not_diag(double *D,int *edge_i,int *index_j,double *nnz,\
	   int NR,int imin,int imax,int jmin,int side,int shift,int findx){

     int j;
     int ind_j;	   
     int limit = 0;
     int idx   = blockIdx.x*blockDim.x + threadIdx.x;

     if(side==-1){
         limit = -(imin+shift-jmin-1);
     }

     if(idx<NR){
	  for(j=edge_i[idx+imin]-findx;j<edge_i[idx+imin+1]-findx;j++){
	      ind_j = index_j[j]-findx-jmin;
	      if(side*ind_j>=limit){
	          //D[idx+ind_j*NR] = nnz[j].x;
	          D[idx+ind_j*NR] = nnz[j];
	      }
	  }
     }	   

     __syncthreads();
}

extern "C"
void d_extract_not_diag_on_dev(double *D,int *edge_i,int *index_j,double *nnz,int NR,\
     int imin,int imax,int jmin,int side,int shift,int findx,cudaStream_t stream){

    uint i_N = NR + (BLOCK_DIM-(NR%BLOCK_DIM));

    d_extract_not_diag<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>(D,edge_i,index_j,nnz,NR,imin,imax,jmin,side,shift,findx);
}

__global__ void z_extract_diag(cuDoubleComplex *D,int *edge_i,int *index_j,cuDoubleComplex *nnz,\
	   int NR,int imin,int imax,int shift,int findx){

     int j;
     int ind_j;	   
     int idx = blockIdx.x*blockDim.x + threadIdx.x;

     if(idx<NR){
	  for(j=edge_i[idx+imin]-findx;j<edge_i[idx+imin+1]-findx;j++){
	      ind_j = index_j[j]-findx-shift-imin;
	      if((ind_j>=0)&&(ind_j<NR)){
	          D[idx+ind_j*NR].x = nnz[j].x;
		  D[idx+ind_j*NR].y = nnz[j].y;
	      }
	  }
     }	   

     __syncthreads();
}

extern "C"
void z_extract_diag_on_dev(CPX *D,int *edge_i,int *index_j,CPX *nnz,int NR,\
     int imin,int imax,int shift,int findx,cudaStream_t stream){

    uint i_N = NR + (BLOCK_DIM-(NR%BLOCK_DIM));

    z_extract_diag<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>((cuDoubleComplex*)D,edge_i,index_j,(cuDoubleComplex*)nnz,NR,imin,imax,shift,findx);
}

__global__ void z_extract_not_diag(cuDoubleComplex *D,int *edge_i,int *index_j,cuDoubleComplex *nnz,\
	   int NR,int imin,int imax,int jmin,int side,int shift,int findx){

     int j;
     int ind_j;	   
     int limit = 0;
     int idx   = blockIdx.x*blockDim.x + threadIdx.x;

     if(side==-1){
         limit = -(imin+shift-jmin-1);
     }

     if(idx<NR){
	  for(j=edge_i[idx+imin]-findx;j<edge_i[idx+imin+1]-findx;j++){
	      ind_j = index_j[j]-findx-jmin;
	      if(side*ind_j>=limit){
	          D[idx+ind_j*NR].x = nnz[j].x;
		  D[idx+ind_j*NR].y = nnz[j].y;
	      }
	  }
     }	   

     __syncthreads();
}

extern "C"
void z_extract_not_diag_on_dev(CPX* *D,int *edge_i,int *index_j,CPX *nnz,int NR,\
     int imin,int imax,int jmin,int side,int shift,int findx,cudaStream_t stream){

    uint i_N = NR + (BLOCK_DIM-(NR%BLOCK_DIM));

    z_extract_not_diag<<< i_N/BLOCK_DIM, BLOCK_DIM, 0, stream >>>((cuDoubleComplex*)D,edge_i,index_j,(cuDoubleComplex*)nnz,NR,imin,imax,jmin,side,shift,findx);
}

extern "C"
void d_copy_csr_to_device(int size,int n_nonzeros,int *hedge_i,int *hindex_j,double *hnnz,\
          		  int *dedge_i,int *dindex_j,double *dnnz){
    
    cudaMemcpyAsync(dedge_i,hedge_i,(size+1)*sizeof(int),cudaMemcpyHostToDevice,NULL);
    cudaMemcpyAsync(dindex_j,hindex_j,n_nonzeros*sizeof(int),cudaMemcpyHostToDevice,NULL);
    cudaMemcpyAsync(dnnz,hnnz,n_nonzeros*sizeof(double),cudaMemcpyHostToDevice,NULL);
}

extern "C"
void z_copy_csr_to_device(int size,int n_nonzeros,int *hedge_i,int *hindex_j,CPX *hnnz,\
          		  int *dedge_i,int *dindex_j,CPX *dnnz){
    
    cudaMemcpyAsync(dedge_i,hedge_i,(size+1)*sizeof(int),cudaMemcpyHostToDevice,NULL);
    cudaMemcpyAsync(dindex_j,hindex_j,n_nonzeros*sizeof(int),cudaMemcpyHostToDevice,NULL);
    cudaMemcpyAsync(dnnz,hnnz,n_nonzeros*sizeof(CPX),cudaMemcpyHostToDevice,NULL);
}

extern "C"
void d_csr_mult_f(void *handle,int m,int n,int k,int n_nonzeros,int *Aedge_i,int *Aindex_j,\
                  double *Annz,double alpha,double *B,double beta,double *C){

    cusparseSpMatDescr_t descra;
    cusparseCreateCsr(&descra,m,k,n_nonzeros,Aedge_i,Aindex_j,Annz,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,\
                      CUSPARSE_INDEX_BASE_ONE,CUDA_R_64F);

    cusparseDnMatDescr_t descrb;
    cusparseCreateDnMat(&descrb,k,n,k,B,CUDA_R_64F,CUSPARSE_ORDER_COL);

    cusparseDnMatDescr_t descrc;
    cusparseCreateDnMat(&descrc,m,n,m,C,CUDA_R_64F,CUSPARSE_ORDER_COL);

    void *dBuffer;
    size_t bufferSize;
    cusparseSpMM_bufferSize((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,\
                 &alpha,descra,descrb,&beta,descrc,CUDA_R_64F,CUSPARSE_MM_ALG_DEFAULT,&bufferSize);
    cudaMalloc(&dBuffer,bufferSize);

    //cusparseCreateMatDescr(&descra);
    //cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
    //cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ONE);

    //cusparseDcsrmm((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,k,n_nonzeros,\
    //               &alpha,descra,Annz,Aedge_i,Aindex_j,B,k,&beta,C,m);
    cusparseSpMM((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,\
                 &alpha,descra,descrb,&beta,descrc,CUDA_R_64F,CUSPARSE_MM_ALG_DEFAULT,dBuffer);

    cusparseDnMatGetValues(descrc,(void**)&C);

    cusparseDestroySpMat(descra);
    cusparseDestroyDnMat(descrb);
    cusparseDestroyDnMat(descrc);
    cudaFree(dBuffer);
}

extern "C"
void z_csr_mult_f(void *handle,int m,int n,int k,int n_nonzeros,int *Aedge_i,int *Aindex_j,\
                  CPX *Annz,CPX alpha,CPX *B,CPX beta,CPX *C){

    cusparseSpMatDescr_t descra;
    cusparseCreateCsr(&descra,m,k,n_nonzeros,Aedge_i,Aindex_j,Annz,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,\
                      CUSPARSE_INDEX_BASE_ONE,CUDA_C_64F);

    cusparseDnMatDescr_t descrb;
    cusparseCreateDnMat(&descrb,k,n,k,B,CUDA_C_64F,CUSPARSE_ORDER_COL);

    cusparseDnMatDescr_t descrc;
    cusparseCreateDnMat(&descrc,m,n,m,C,CUDA_C_64F,CUSPARSE_ORDER_COL);

    void *dBuffer;
    size_t bufferSize;
    cusparseSpMM_bufferSize((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,\
                 &alpha,descra,descrb,&beta,descrc,CUDA_C_64F,CUSPARSE_MM_ALG_DEFAULT,&bufferSize);
    cudaMalloc(&dBuffer,bufferSize);

    //cusparseCreateMatDescr(&descra);
    //cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
    //cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ONE);

    //cusparseZcsrmm((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,k,n_nonzeros,\
    //               (cuDoubleComplex*)&alpha,descra,(cuDoubleComplex*)Annz,Aedge_i,Aindex_j,\
		//   (cuDoubleComplex*)B,k,(cuDoubleComplex*)&beta,(cuDoubleComplex*)C,m);
    cusparseSpMM((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,\
                 &alpha,descra,descrb,&beta,descrc,CUDA_C_64F,CUSPARSE_MM_ALG_DEFAULT,dBuffer);

    cusparseDnMatGetValues(descrc,(void**)&C);

    cusparseDestroySpMat(descra);
    cusparseDestroyDnMat(descrb);
    cusparseDestroyDnMat(descrc);
    cudaFree(dBuffer);
}

extern "C"
void z_csr_mult_fo(void *handle,int m,int n,int k,int n_nonzeros,int *Aedge_i,int *Aindex_j,\
                   CPX *Annz,CPX alpha,CPX *B,CPX beta,CPX *C){

    cusparseSpMatDescr_t descra;
    cusparseCreateCsr(&descra,m,k,n_nonzeros,Aedge_i,Aindex_j,Annz,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,\
                      CUSPARSE_INDEX_BASE_ZERO,CUDA_C_64F);

    cusparseDnMatDescr_t descrb;
    cusparseCreateDnMat(&descrb,k,n,k,B,CUDA_C_64F,CUSPARSE_ORDER_COL);

    cusparseDnMatDescr_t descrc;
    cusparseCreateDnMat(&descrc,m,n,m,C,CUDA_C_64F,CUSPARSE_ORDER_COL);

    void *dBuffer;
    size_t bufferSize;
    cusparseSpMM_bufferSize((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,\
                 &alpha,descra,descrb,&beta,descrc,CUDA_C_64F,CUSPARSE_MM_ALG_DEFAULT,&bufferSize);
    cudaMalloc(&dBuffer,bufferSize);

    //cusparseCreateMatDescr(&descra);
    //cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
    //cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);

    //cusparseZcsrmm((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,k,n_nonzeros,\
    //               (cuDoubleComplex*)&alpha,descra,(cuDoubleComplex*)Annz,Aedge_i,Aindex_j,\
		//   (cuDoubleComplex*)B,k,(cuDoubleComplex*)&beta,(cuDoubleComplex*)C,m);
    cusparseSpMM((cusparseHandle_t)handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,\
                 &alpha,descra,descrb,&beta,descrc,CUDA_C_64F,CUSPARSE_MM_ALG_DEFAULT,dBuffer);

    cusparseDnMatGetValues(descrc,(void**)&C);

    cusparseDestroySpMat(descra);
    cusparseDestroyDnMat(descrb);
    cusparseDestroyDnMat(descrc);
    cudaFree(dBuffer);
}

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void d_transpose(double *odata, double *idata, int width, int height)
{
	__shared__ double block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

extern "C"
void d_transpose_matrix(double *odata,double *idata,int size_x,int size_y){

    uint i_size_x = size_x + (BLOCK_DIM-(size_x%BLOCK_DIM));
    uint i_size_y = size_y + (BLOCK_DIM-(size_y%BLOCK_DIM));

    dim3 grid(i_size_x / BLOCK_DIM, i_size_y / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    d_transpose<<< grid, threads >>>(odata, idata, size_x, size_y);
}

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void z_transpose(cuDoubleComplex *odata, cuDoubleComplex *idata, int width, int height)
{
	__shared__ cuDoubleComplex block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out].x = block[threadIdx.x][threadIdx.y].x;
		odata[index_out].y = -block[threadIdx.x][threadIdx.y].y;
	}
}

extern "C"
void z_transpose_matrix(CPX *odata,CPX *idata,int size_x,int size_y){

    uint i_size_x = size_x + (BLOCK_DIM-(size_x%BLOCK_DIM));
    uint i_size_y = size_y + (BLOCK_DIM-(size_y%BLOCK_DIM));

    dim3 grid(i_size_x / BLOCK_DIM, i_size_y / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    z_transpose<<< grid, threads >>>((cuDoubleComplex*)odata,(cuDoubleComplex*)idata,size_x,size_y);
}

__global__ void d_symmetrize(double *matrix, int N)
{

        unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
        unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

        if((xIndex < N) && (yIndex < N) && (yIndex>=xIndex)){
            unsigned int index_1  = yIndex * N + xIndex;
            unsigned int index_2  = xIndex * N + yIndex;
            double val_1    = matrix[index_1];
            double val_2    = matrix[index_2];

            matrix[index_1] = (val_1+val_2)/2.0;
            matrix[index_2] = (val_1+val_2)/2.0;
        }

        __syncthreads();
}

extern "C"
void d_symmetrize_matrix(double *matrix,int N,cudaStream_t stream){

    uint i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    dim3 grid(i_size / BLOCK_DIM, i_size / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    d_symmetrize<<< grid, threads, 0, stream >>>(matrix, N);
}

__global__ void z_symmetrize(cuDoubleComplex *matrix, int N)
{

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if((xIndex < N) && (yIndex < N) && (yIndex>=xIndex)){
	    unsigned int index_1  = yIndex * N + xIndex;
	    unsigned int index_2  = xIndex * N + yIndex;
	    cuDoubleComplex val_1 = matrix[index_1];
	    cuDoubleComplex val_2 = matrix[index_2];
	    
	    matrix[index_1].x     = (val_1.x+val_2.x)/2.0;
	    matrix[index_1].y     = (val_1.y-val_2.y)/2.0;
	    matrix[index_2].x     = (val_1.x+val_2.x)/2.0;
	    matrix[index_2].y     = -(val_1.y-val_2.y)/2.0;
	}

	__syncthreads();
}

extern "C"
void z_symmetrize_matrix(CPX *matrix,int N,cudaStream_t stream){

    uint i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    dim3 grid(i_size / BLOCK_DIM, i_size / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    z_symmetrize<<< grid, threads, 0, stream >>>((cuDoubleComplex*)matrix, N);
}

__global__ void z_symmetrize_2(cuDoubleComplex *matrix, int N)
{

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if((xIndex < N) && (yIndex < N) && (yIndex>=xIndex)){
	    unsigned int index_1  = yIndex * N + xIndex;
	    unsigned int index_2  = xIndex * N + yIndex;
	    cuDoubleComplex val_1 = matrix[index_1];

	    if(yIndex==xIndex){
	        matrix[index_1].x = 0.0;
		matrix[index_1].y = val_1.y;
	    }else{
	        matrix[index_2].x = -val_1.x;
		matrix[index_2].y = val_1.y;
	    }
	}

	__syncthreads();
}

extern "C"
void z_symmetrize_matrix_2(CPX *matrix,int N,cudaStream_t stream){

    uint i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    dim3 grid(i_size / BLOCK_DIM, i_size / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    z_symmetrize_2<<< grid, threads, 0, stream >>>((cuDoubleComplex*)matrix, N);
}

__global__ void d_tril(double *A, int lda, int N)
{
   unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
   unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

   if((xIndex < N) && (yIndex < N) && (xIndex > yIndex))
   {
      unsigned int i  = xIndex * lda + yIndex;
      A[i] = 0.0;
   }

   __syncthreads();
}

extern "C"
void d_tril_on_dev(double *A, int lda, int N)
{
    uint i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    dim3 grid(i_size / BLOCK_DIM, i_size / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    d_tril<<<grid, threads>>>(A, lda, N);
}

__global__ void z_tril(cuDoubleComplex *A, int lda, int N)
{
   unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
   unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

   if((xIndex < N) && (yIndex < N) && (xIndex > yIndex))
   {
      unsigned int i  = xIndex * lda + yIndex;
      A[i].x = 0.0;
      A[i].y = 0.0;
   }

   __syncthreads();
}

extern "C"
void z_tril_on_dev(CPX *A, int lda, int N)
{
    uint i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    dim3 grid(i_size / BLOCK_DIM, i_size / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    z_tril<<<grid, threads>>>((cuDoubleComplex*)A, lda, N);
}

__global__ void d_indexed_copy(double *src, double *dst, size_t *index, size_t N)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < N)
   {
      dst[idx] = src[index[idx]];
   }

   __syncthreads();
}

extern "C"
void d_indexed_copy_on_dev(double *src, double *dst, size_t *index, size_t N)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    d_indexed_copy<<<i_size/BLOCK_DIM, BLOCK_DIM>>>(src, dst, index, N);
}

__global__ void z_indexed_copy(cuDoubleComplex *src, cuDoubleComplex *dst, size_t *index, size_t N)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < N)
   {
      dst[idx] = src[index[idx]];
   }

   __syncthreads();
}

extern "C"
void z_indexed_copy_on_dev(CPX *src, CPX *dst, size_t *index, size_t N)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    z_indexed_copy<<<i_size/BLOCK_DIM, BLOCK_DIM>>>((cuDoubleComplex*)src, (cuDoubleComplex*)dst, index, N);
}

__global__ void d_indexed_copy_offset(double *src, double *dst, size_t *index, size_t N, size_t offset)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < N)
   {
      dst[idx] = src[index[idx]-offset];
   }

   __syncthreads();
}

extern "C"
void d_indexed_copy_offset_on_dev(double *src, double *dst, size_t *index, size_t N, size_t offset)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    d_indexed_copy_offset<<<i_size/BLOCK_DIM, BLOCK_DIM>>>(src, dst, index, N, offset);
}

__global__ void z_indexed_copy_offset(cuDoubleComplex *src, cuDoubleComplex *dst, size_t *index, size_t N, size_t offset)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < N)
   {
      dst[idx] = src[index[idx]-offset];
   }

   __syncthreads();
}

extern "C"
void z_indexed_copy_offset_on_dev(CPX *src, CPX *dst, size_t *index, size_t N, size_t offset)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    z_indexed_copy_offset<<<i_size/BLOCK_DIM, BLOCK_DIM>>>((cuDoubleComplex*)src, (cuDoubleComplex*)dst, index, N, offset);
}

__global__ void d_log(double *x, size_t N)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < N)
   {
      x[idx] = log(x[idx]);
   }

   __syncthreads();
}

extern "C"
void d_log_on_dev(double *x, size_t N)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    d_log<<<i_size/BLOCK_DIM, BLOCK_DIM>>>(x, N);
}

__global__ void z_log(cuDoubleComplex *x, size_t N)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < N)
   {
      x[idx].x = log(x[idx].x);
      x[idx].y = 0.0;
   }

   __syncthreads();
}

extern "C"
void z_log_on_dev(CPX *x, size_t N)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    z_log<<<i_size/BLOCK_DIM, BLOCK_DIM>>>((cuDoubleComplex*)x, N);
}

__global__
void d_fill(double* x, const double value, size_t n)
{
    auto i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < n) {
        x[i] = value;
    }
}

extern "C"
void d_fill_on_dev(double *x, const double value, size_t N)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    d_fill<<<i_size/BLOCK_DIM, BLOCK_DIM>>>(x, value, N);
}

__global__
void z_fill(cuDoubleComplex* x, const cuDoubleComplex value, size_t n)
{
    auto i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < n) {
        x[i] = value;
    }
}

extern "C"
void z_fill_on_dev(CPX *x, const CPX value, size_t N)
{
    size_t i_size = N + (BLOCK_DIM-(N%BLOCK_DIM));

    z_fill<<<i_size/BLOCK_DIM, BLOCK_DIM>>>((cuDoubleComplex*)x, *reinterpret_cast<const cuDoubleComplex*>(&value), N);
}

__device__
inline size_t getPos(size_t r, size_t c, size_t ns, size_t nt, size_t nd)
{
   size_t ib = c / ns;

   // c in block2 : c in block1 or dense block
   size_t block2_columns = (ib < nt-1) ? c : (nt-1) * ns;
   size_t block1_columns;
   if (ib < nt-1) // c in block2
      block1_columns = 0;
   else if (ib == nt-1) // c in block1
      block1_columns = c % ns;
   else // c in dense block
      block1_columns = ns;
   // c in dense block : c not in dense block
   size_t dense_block_columns = (c > ns*nt) ? c - ns*nt : 0;

   size_t column_pos;
   if (r < ns*nt) // not in dense row
      column_pos = r - ib*ns;
   else // in dense row
   {
      if (ib < nt-1) // 2 blocks
         column_pos = r - (nt-2)*ns;
      else if (ib == nt-1) // 1 block
         column_pos = r - (nt-1)*ns;
      else // dense block
         column_pos = r - nt*ns;
   }

   return block2_columns*(2*ns+nd) + block1_columns*(ns+nd) + dense_block_columns*nd + column_pos;
}

__global__ void d_init_block_matrix(double *M, size_t *ia, size_t *ja, double *a, size_t nnz, size_t ns, size_t nt, size_t nd)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < nnz)
   {
      size_t c = 0;
      while (ia[c+1] < idx+1)
         c++;
      size_t r = ja[idx];

      size_t i = getPos(r, c, ns, nt, nd);

      M[i] = a[idx];
   }

   __syncthreads();
}

extern "C"
void d_init_block_matrix_on_dev(double *M, size_t *ia, size_t *ja, double *a, size_t nnz, size_t ns, size_t nt, size_t nd)
{
    size_t i_size = nnz + (BLOCK_DIM-(nnz%BLOCK_DIM));

    d_init_block_matrix<<<i_size/BLOCK_DIM, BLOCK_DIM>>>(M, ia, ja, a, nnz, ns, nt, nd);
}

__global__ void z_init_block_matrix(cuDoubleComplex *M, size_t *ia, size_t *ja, cuDoubleComplex *a, size_t nnz, size_t ns, size_t nt, size_t nd)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   if (idx < nnz)
   {
      size_t c = 0;
      while (ia[c+1] < idx+1)
         c++;
      size_t r = ja[idx];

      size_t i = getPos(r, c, ns, nt, nd);

      M[i] = a[idx];
   }

   __syncthreads();
}

extern "C"
void z_init_block_matrix_on_dev(CPX *M, size_t *ia, size_t *ja, CPX *a, size_t nnz, size_t ns, size_t nt, size_t nd)
{
    size_t i_size = nnz + (BLOCK_DIM-(nnz%BLOCK_DIM));

    z_init_block_matrix<<<i_size/BLOCK_DIM, BLOCK_DIM>>>((cuDoubleComplex*)M, ia, ja, (cuDoubleComplex*)a, nnz, ns, nt, nd);
}

__global__ void d_init_supernode(double *M, size_t *ia, size_t *ja, double *a, size_t supernode_fc, size_t supernode_lc, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   size_t offset = getPos(supernode_fc, supernode_fc, ns, nt, nd);

   if (idx < supernode_nnz)
   {
      size_t c = 0;
      while (ia[c+1] < idx+supernode_offset+1)
         c++;
      c += supernode_fc;
      size_t r = ja[idx];

      size_t i = getPos(r, c, ns, nt, nd) - offset;

      M[i] = a[idx];
   }

   __syncthreads();
}

extern "C"
void d_init_supernode_on_dev(double *M, size_t *ia, size_t *ja, double *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd)
{
    size_t i_size = supernode_nnz + (BLOCK_DIM-(supernode_nnz%BLOCK_DIM));

    size_t supernode_fc = supernode * ns;
    size_t supernode_lc = supernode < nt ? (supernode+1) * ns : ns * nt + nd;

    d_init_supernode<<<i_size/BLOCK_DIM, BLOCK_DIM>>>(M, ia, ja, a, supernode_fc, supernode_lc, supernode_nnz, supernode_offset, ns, nt, nd);
}

__global__ void z_init_supernode(cuDoubleComplex *M, size_t *ia, size_t *ja, cuDoubleComplex *a, size_t supernode_fc, size_t supernode_lc, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd)
{
   size_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;

   size_t offset = getPos(supernode_fc, supernode_fc, ns, nt, nd);

   if (idx < supernode_nnz)
   {
      size_t c = 0;
      while (ia[c+1] < idx+supernode_offset+1)
         c++;
      c += supernode_fc;
      size_t r = ja[idx];

      size_t i = getPos(r, c, ns, nt, nd) - offset;

      M[i] = a[idx];
   }

   __syncthreads();
}

extern "C"
void z_init_supernode_on_dev(CPX *M, size_t *ia, size_t *ja, CPX *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd)
{
    size_t i_size = supernode_nnz + (BLOCK_DIM-(supernode_nnz%BLOCK_DIM));

    size_t supernode_fc = supernode * ns;
    size_t supernode_lc = supernode < nt ? (supernode+1) * ns : ns * nt + nd;

    z_init_supernode<<<i_size/BLOCK_DIM, BLOCK_DIM>>>((cuDoubleComplex*)M, ia, ja, (cuDoubleComplex*)a, supernode_fc, supernode_lc, supernode_nnz, supernode_offset, ns, nt, nd);
}
