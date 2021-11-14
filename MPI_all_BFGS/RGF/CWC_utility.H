#include <cuda_runtime.h>
#include <stdio.h>
#include "Types.H"
#include "magma_v2.h"

extern "C" {

  void set_gpu(int,char*);
  void cublas_init(void**);
  void cublas_finalize(void*);
  void cusparse_init(void**);
  void cusparse_finalize(void*);
  size_t allocate_data_on_device(void**,size_t);
  void deallocate_data_on_device(void*);
  size_t deallocate_data_on_dev(void*,size_t);
  void copy_data_to_device(void*,void*,int,int,size_t);
  void memcpy_to_device(void*,void*,size_t);
  void copy_data_to_host(void*,void*,int,int,size_t);
  void memcpy_to_host(void*,void*,size_t);
  void dgemm_on_dev(void*,char,char,int,int,int,double,double*,\
		    int,double*,int,double,double*,int);
  void zgemm_on_dev(void*,char,char,int,int,int,CPX,CPX*,\
		    int,CPX*,int,CPX,CPX*,int);
  void daxpy_on_dev(void*,int,double,double*,int,double*,int);
  void zaxpy_on_dev(void*,int,CPX,CPX*,int,CPX*,int);
  void dasum_on_dev(void *handle,int n,double *x,int incx,double *result);
  void zasum_on_dev(void *handle,int n,CPX *x,int incx,double *result);
  void dsum_on_dev(int n,double *x,int incx,double *result,magma_queue_t queue);
  void zsum_on_dev(int n,CPX *x,int incx,CPX *result,magma_queue_t queue);
  void d_copy_csr_to_device(int,int,int*,int*,double*,int*,int*,double*);
  void z_copy_csr_to_device(int,int,int*,int*,CPX*,int*,int*,CPX*);
  void d_csr_mult_f(void*,int,int,int,int,int*,int*,double*,double,\
		    double*,double,double*);
  void z_csr_mult_f(void*,int,int,int,int,int*,int*,CPX*,CPX,\
		    CPX*,CPX,CPX*);
  void z_csr_mult_fo(void*,int,int,int,int,int*,int*,CPX*,CPX,\
		     CPX*,CPX,CPX*);
  void d_transpose_matrix(double*,double*,int,int);
  void z_transpose_matrix(CPX*,CPX*,int,int);
  void d_init_var_on_dev(double*,int,cudaStream_t);
  void d_init_eye_on_dev(double*,int,cudaStream_t);
  void z_init_var_on_dev(CPX*,int,cudaStream_t);
  void z_init_eye_on_dev(CPX*,int,cudaStream_t);
  void correct_diag_on_dev(CPX*,int,cudaStream_t);
  void change_var_type_on_dev(double*,CPX*,int,cudaStream_t);
  void change_sign_imag_on_dev(CPX*,int);
  void d_extract_diag_on_dev(double*,int*,int*,double*,int,int,int,int,int,\
			     cudaStream_t);
  void d_extract_not_diag_on_dev(double*,int*,int*,double*,int,int,int,int,\
				 int,int,int,cudaStream_t); 
  void d_symmetrize_matrix(double*,int,cudaStream_t);
  void z_extract_diag_on_dev(CPX*,int*,int*,CPX*,int,int,int,int,int,\
			     cudaStream_t);
  void z_extract_not_diag_on_dev(CPX*,int*,int*,CPX*,int,int,int,int,\
				 int,int,int,cudaStream_t); 
  void z_symmetrize_matrix(CPX*,int,cudaStream_t);
  void z_symmetrize_matrix_2(CPX*,int,cudaStream_t);
  void d_tril_on_dev(double *A, int lda, int N);
  void z_tril_on_dev(CPX *A, int lda, int N);
  void d_indexed_copy_on_dev(double *src, double *dst, size_t *index, size_t N);
  void z_indexed_copy_on_dev(CPX *src, CPX *dst, size_t *index, size_t N);
  void d_indexed_copy_offset_on_dev(double *src, double *dst, size_t *index, size_t N, size_t offset);
  void z_indexed_copy_offset_on_dev(CPX *src, CPX *dst, size_t *index, size_t N, size_t offset);
  void d_log_on_dev(double *x, size_t N);
  void z_log_on_dev(CPX *x, size_t N);
  void d_fill_on_dev(double *x, const double value, size_t N);
  void z_fill_on_dev(CPX *x, const CPX value, size_t N);
  void d_init_block_matrix_on_dev(double *M, size_t *ia, size_t *ja, double *a, size_t nnz, size_t ns, size_t nt, size_t nd);
  void z_init_block_matrix_on_dev(CPX *M, size_t *ia, size_t *ja, CPX *a, size_t nnz, size_t ns, size_t nt, size_t nd);
  void d_init_supernode_on_dev(double *M, size_t *ia, size_t *ja, double *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd);
  void z_init_supernode_on_dev(CPX *M, size_t *ia, size_t *ja, CPX *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd);
}
