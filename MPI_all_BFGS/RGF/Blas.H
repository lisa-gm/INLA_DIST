#ifndef __BLAS
#define __BLAS

#include <iostream>
#include "Types.H"
#include "CWC_utility.H"
#include "magma_v2.h"

extern "C" {
//Blas
    void fortran_name(dcopy,DCOPY)(int *n,double *dx,int *incx,double *dy,int *incy);
    void fortran_name(daxpy,DAXPY)(int *n,double *da,double *dx,int *incx,double *dy,\
                                      int *incy);
    double fortran_name(dnrm2,DNRM2)(int *n,double *dx,int *incx);
    double fortran_name(dznrm2,DZNRM2)(int *n,CPX *dx,int *incx);
    void fortran_name(dscal,DSCAL)(int *n,double *da,double *dx,int *incx);
    void fortran_name(zscal,ZSCAL)(int *n,CPX *da,CPX *dx,int *incx);
    void fortran_name(dgemm,DGEMM)(char *transa, char *transb, int *m, int *n, int *k, \
                                   double *alpha, double *a, int *lda, double *b, int *ldb, \
                                   double *beta, double *c, int *ldc);
    void fortran_name(dsymm,DSYMM)(char *side, char *uplo, int *m, int *n, double *alpha, \
                                   double *a, int *lda, double *b, int *ldb, double *beta, \
                                   double *c, int *ldc);
    void fortran_name(dtrsm,DTRSM)(char *side, char *uplo, char *transa, char *diag, \
                                   int *m, int *n, \
                                   double *alpha, double *a, int *lda, double *b, int *ldb);
    void fortran_name(zgemm,ZGEMM)(char *transa, char *transb, int *m, int *n, int *k, \
                                   CPX *alpha, CPX *a, int *lda, CPX *b, int *ldb, \
                                   CPX *beta, CPX *c, int *ldc);
    void fortran_name(zsymm,ZSYMM)(char *side, char *uplo, int *m, int *n, CPX *alpha, \
                                   CPX *a, int *lda, CPX *b, int *ldb, CPX *beta, \
                                   CPX *c, int *ldc);
    void fortran_name(ztrsm,ZTRSM)(char *side, char *uplo, char *transa, char *diag, \
                                   int *m, int *n, \
                                   CPX *alpha, CPX *a, int *lda, CPX *b, int *ldb);
    void fortran_name(zhemm,ZHEMM)(char *side, char *uplo, int *m, int *n, CPX *alpha, \
                                   CPX *a, int *lda, CPX *b, int *ldb, CPX *beta, \
                                   CPX *c, int *ldc);
    void fortran_name(dgemv,DGEMV)(char *trans, int *m, int *n, double *alpha, double *a, \
                                   int *lda, double *x, int *incx, double *beta, double *y, \
                                   int *incy);
    
    void fortran_name(zgemv,ZGEMV)(char *trans, int *m, int *n, CPX *alpha, CPX *a, \
                                   int *lda, CPX *x, int *incx, CPX *beta, CPX *y, \
                                   int *incy);

    double fortran_name(ddot,DDOT)(int *n,double *x,int *incx,double *y,int *incy);
    CPX fortran_name(zdotc,ZDOTC)(int *n,CPX *x,int *incx,CPX *y,int *incy);
    void fortran_name(zcopy,ZCOPY)(int *n,CPX *dx,int *incx,CPX *dy,int *incy);
    void fortran_name(zaxpy,ZAXPY)(int *n, CPX *alpha, CPX *x, int *incx, CPX *y, int *incy);
    double fortran_name(dasum,DASUM)(int *n,double *dx,int *incx);
    double fortran_name(dzasum,DZASUM)(int *n,CPX *dx,int *incx);

//Lapack
    void fortran_name(dgetrf,DGETRF)(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
    void fortran_name(dgetrs,DGETRS)(char *trans, int *n, int *nrhs, double *a, int *lda, \
                                     int *ipiv, double *b, int *ldb, int *info);
    void fortran_name(zgetrf,ZGETRF)(int *m, int *n, CPX *a, int *lda, int *ipiv, int *info);
    void fortran_name(zgetrs,ZGETRS)(char *trans, int *n, int *nrhs, CPX *a, int *lda, \
                                     int *ipiv, CPX *b, int *ldb, int *info);
    void fortran_name(zgetri,ZGETRI)(int *n,CPX *a,int *lda,int *ipiv,CPX *work,int *lwork,\
                                     int *info);
    void fortran_name(dgeev,DGEEV)(char *jobvl, char *jobvr, int *n, double *a, int *lda, \
                                   double *wr, double *wi, double *vl, int *ldvl, double *vr, \
                                   int *ldvr, double *work, int *lwork, int *info);
    void fortran_name(dsyev,DSYEV)(char *JOBZ,char *UPLO,int *N,double *A,int *LDA,double *W,\
				   double *WORK,int *LWORK,int *INFO);
    void fortran_name(dggev,DGGEV)(char* jobvl, char *jobvr, int *n, double *a, int *lda, \
                                   double *b, int *ldb, double *alphar, double *alphai, \
                                   double *beta, double *vl, int *ldvl, double *vr, int *ldvr, \
                                   double *work, int *lwork, int *info);
    void fortran_name(zggev,ZGGEV)(char* jobvl, char *jobvr, int *n, CPX *a, int *lda, \
                                   CPX *b, int *ldb, CPX *alpha, CPX *beta, CPX *vl, int *ldvl,\
				   CPX *vr, int *ldvr, CPX *work, int *lwork, double *rwork, \
				   int *info);
    void fortran_name(zgeev,ZGEEV)(char *jobvl, char *jobvr, int *n, CPX *a, int *lda, \
                                   CPX *w, CPX *vl, int *ldvl, CPX *vr, int *ldvr, CPX *work, \
                                   int *lwork, double *rwork, int *info);
    void fortran_name(zheev,ZHEEV)(char *jobvl, char *uplo, int *n, CPX *a, int *lda, \
                                   double *w, CPX *work, int *lwork, double *rwork, int *info);
    void fortran_name(dgetri,DGETRI)(int *n,double *a,int *lda,int *ipiv,double *work,int *lwork,\
                                     int *info);
    void fortran_name(dsytri,DSYTRI)(char *uplo,int *n,double *a,int *lda,int *ipiv,double *work,\
                                     int *info);
    void fortran_name(zhetrf,ZHETRF)(char *uplo,int *n,CPX *a,int *lda,int *ipiv,CPX *work,\
                                     int *lwork,int *info);
    void fortran_name(zhetri,ZHETRI)(char *uplo,int *n,CPX *a,int *lda,int *ipiv,CPX *work,\
                                     int *info);
    void fortran_name(zhetrs,ZHETRS)(char *uplo,int *n,int *nrhs,CPX *a,int *lda,int *ipiv,\
                                     CPX *b,int *ldb,int *info);
    void fortran_name(dsysv,DSYSV)(char *uplo,int *n,int *nrhs,double *a,int *lda,int *ipiv,\
                                   double *b,int *ldb,double *work,int *lwork,int *info);
    void fortran_name(dsytrf,DSYTRF)(char *uplo,int *n,double *a,int *lda,int *ipiv,double *work,\
                                     int *lwork,int *info);
    void fortran_name(dsytrs,DSYTRS)(char *uplo,int *n,int *nrhs,double *a,int *lda,int *ipiv,\
                                     double *b,int *ldb,int *info);
    void fortran_name(dstebz,DSTEBZ)(char *range,char *order,int *iter,double *vl,double *vu,int *il,int *iu,\
                                     double *abstol,double *diag,double *offd,int *neval,int *nsplit,\
                                     double *eval,int *iblock,int *isplit,double *work,int *iwork,int *info);
    void fortran_name(zlarnv,ZLARNV)(int*,int*,int*,CPX*);
    void fortran_name(dgesdd,DGESDD)(char *jobz,int *m,int *n,double *a,int *lda,double *s,double *u,int *ldu, \
                                     double *vt,int *ldvt,double *work,int *lwork,int *iwork,int *info);
    void fortran_name(zgesdd,ZGESDD)(char *jobz,int *m,int *n,CPX *a,int *lda,double *s,CPX *u,int *ldu,\
                                     CPX *vt,int *ldvt,CPX *work,int *lwork,double *rwork,int *iwork,int *info);

}

/*SAB********************************************************************************************/

inline double c_dzasum(int n,CPX *dx,int incx)
{
    return fortran_name(dzasum,DZASUM)(&n,dx,&incx);
}

/*My Blas*******************************************************************************************/

inline void c_icopy(int n,int *dx,int incx,int *dy,int incy)
{
    int i;
    
    for(i=0;i<n;i++) dy[i*incy] = dx[i*incx];
}

/*Blas*******************************************************************************************/
    
inline void c_dcopy(int n,double *dx,int incx,double *dy,int incy)
{
    fortran_name(dcopy,DCOPY)(&n,dx,&incx,dy,&incy);
}

/************************************************************************************************/

inline void c_daxpy(int n,double da,double *dx,int incx,double *dy,int incy)
{
    fortran_name(daxpy,DAXPY)(&n,&da,dx,&incx,dy,&incy);
}

/************************************************************************************************/

inline double c_dnrm2(int n,double* dx,int incx)
{
    return fortran_name(dnrm2,DNRM2)(&n,dx,&incx);
}

/************************************************************************************************/

inline double c_dznrm2(int n,CPX* dx,int incx)
{
    return fortran_name(dznrm2,DZNRM2)(&n,dx,&incx);
}

/************************************************************************************************/

inline void c_dscal(int n,double da,double *dx,int incx)
{
    fortran_name(dscal,DSCAL)(&n,&da,dx,&incx);
}

/************************************************************************************************/

inline void c_zscal(int n,CPX da,CPX *dx,int incx)
{
    fortran_name(zscal,ZSCAL)(&n,&da,dx,&incx);
}

/************************************************************************************************/

inline void c_dgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, \
                    int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    fortran_name(dgemm,DGEMM)(&transa,&transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
}

/************************************************************************************************/

inline void c_dsymm(char side,char uplo, int m, int n, double alpha, double *a, int lda, \
                    double *b, int ldb, double beta, double *c, int ldc)
{
    fortran_name(dsymm,DSYMM)(&side,&uplo,&m,&n,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
}

/************************************************************************************************/

inline void c_dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb)
{
    fortran_name(dtrsm,DTRSM)(&side,&uplo,&transa,&diag,&m,&n,&alpha,a,&lda,b,&ldb);
}

/************************************************************************************************/

inline void c_zgemm(char transa, char transb, int m, int n, int k, CPX alpha, CPX *a, \
                    int lda, CPX *b, int ldb, CPX beta, CPX *c, int ldc)
{
    fortran_name(zgemm,ZGEMM)(&transa,&transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
}

/************************************************************************************************/

inline void c_zsymm(char side, char uplo, int m, int n, CPX alpha, CPX *a, int lda, \
                    CPX *b, int ldb, CPX beta, CPX *c, int ldc)
{
    fortran_name(zsymm,ZSYMM)(&side,&uplo,&m,&n,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
}

/************************************************************************************************/

inline void c_ztrsm(char side, char uplo, char transa, char diag, int m, int n, CPX alpha, CPX *a, int lda, CPX *b, int ldb)
{
    fortran_name(ztrsm,ZTRSM)(&side,&uplo,&transa,&diag,&m,&n,&alpha,a,&lda,b,&ldb);
}

/************************************************************************************************/

inline void c_zhemm(char side, char uplo, int m, int n, CPX alpha, CPX *a, int lda, \
                    CPX *b, int ldb, CPX beta, CPX *c, int ldc)
{
    fortran_name(zhemm,ZHEMM)(&side,&uplo,&m,&n,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
}

/************************************************************************************************/

inline void c_dgemv(char transa, int m, int n, double alpha, double *a, int lda, double *x, \
                    int incx, double beta, double *y, int incy)
{
    fortran_name(dgemv,DGEMV)(&transa,&m,&n,&alpha,a,&lda,x,&incx,&beta,y,&incy);
}

/************************************************************************************************/

inline void c_zgemv(char transa, int m, int n, CPX alpha, CPX *a, int lda, CPX *x, \
                    int incx, CPX beta, CPX *y, int incy)
{
    fortran_name(zgemv,ZGEMV)(&transa,&m,&n,&alpha,a,&lda,x,&incx,&beta,y,&incy);

}

/***********************************************************************************************/
inline double c_ddot(int n, double *x, int incx, double *y, int incy)
{
    return fortran_name(ddot,DDOT)(&n,x,&incx,y,&incy);
}

/************************************************************************************************/

inline CPX c_zdotc(int n, CPX *x, int incx, CPX *y, int incy)
{

    //return fortran_name(zdotc,ZDOTC)(&n,x,&incx,y,&incy);

    double real,imag;

    real = c_ddot(n,(double*)x,2*incx,(double*)y,2*incy)+\
      c_ddot(n,(double*)&x[0]+1,2*incx,(double*)&y[0]+1,2*incy);
    imag = -c_ddot(n,(double*)&x[0]+1,2*incx,(double*)y,2*incy)+\
      c_ddot(n,(double*)x,2*incx,(double*)&y[0]+1,2*incy);

    return CPX(real,imag);

}

/************************************************************************************************/

inline void c_zcopy(int n,CPX *dx,int incx,CPX *dy,int incy)
{
    fortran_name(zcopy,ZCOPY)(&n,dx,&incx,dy,&incy);
}

/************************************************************************************************/

inline void c_zaxpy(int n, CPX alpha, CPX *x, int incx, CPX *y, int incy)
{
    fortran_name(zaxpy,ZAXPY)(&n,&alpha,x,&incx,y,&incy);
}

/************************************************************************************************/

inline double c_dasum(int n,double *dx,int incx)
{
    return fortran_name(dasum,DASUM)(&n,dx,&incx);
}

/************************************************************************************************/

template <typename T,typename W>
inline void c_tcopy(int n,T *dx,int incx,W *dy,int incy);

template<>
inline void c_tcopy(int n,double *dx,int incx,double *dy,int incy)
{
    c_dcopy(n,dx,incx,dy,incy);
}

template<>
inline void c_tcopy(int n,CPX *dx,int incx,CPX *dy,int incy)
{
    c_zcopy(n,dx,incx,dy,incy);
}

template<>
inline void c_tcopy(int n,double *dx,int incx,CPX *dy,int incy)
{
    c_dcopy(n,dx,incx,(double*)dy,2*incy);
}

/************************************************************************************************/

template <typename T,typename W>
inline void c_taxpy(int n, T alpha, T *x, int incx, W *y, int incy);

template <>
inline void c_taxpy(int n, double alpha, double *x, int incx, double *y, int incy)
{
    c_daxpy(n,alpha,x,incx,y,incy);
}

template <>
inline void c_taxpy(int n, CPX alpha, CPX *x, int incx, CPX *y, int incy)
{
    c_zaxpy(n,alpha,x,incx,y,incy);
}

template <>
inline void c_taxpy(int n, double alpha, double *x, int incx, CPX *y, int incy)
{
    c_daxpy(n,alpha,x,incx,(double*)y,2*incy);
}

/************************************************************************************************/

template <typename T>
inline double c_dtnrm2(int n,T *x,int incx);

template <>
inline double c_dtnrm2(int n,double *x,int incx)
{
    return c_dnrm2(n,x,incx);
}

template <>
inline double c_dtnrm2(int n,CPX *x,int incx)
{
    return c_dznrm2(n,x,incx);
}

/************************************************************************************************/

template <typename T>
inline void c_tscal(int n,T da,T *dx,int incx);

template <>
inline void c_tscal(int n,double da,double *dx,int incx)
{
    c_dscal(n,da,dx,incx);
}

template <>
inline void c_tscal(int n,CPX da,CPX *dx,int incx)
{
    c_zscal(n,da,dx,incx);
}

/************************************************************************************************/

template <typename T>
inline void c_tgemm(char transa, char transb, int m, int n, int k, T alpha, T *a, \
                    int lda, T *b, int ldb, T beta, T *c, int ldc);

template <>
inline void c_tgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, \
                    int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    c_dgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

template <>
inline void c_tgemm(char transa, char transb, int m, int n, int k, CPX alpha, CPX *a, \
                    int lda, CPX *b, int ldb, CPX beta, CPX *c, int ldc)
{
    c_zgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

/************************************************************************************************/

template <typename T>
inline void c_tsymm(char side, char uplo, int m, int n, T alpha, T *a, \
                    int lda, T *b, int ldb, T beta, T *c, int ldc);

template <>
inline void c_tsymm(char side, char uplo, int m, int n, double alpha, double *a, \
                    int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    c_dsymm(side,uplo,m,n,alpha,a,lda,b,ldb,beta,c,ldc);
}

template <>
inline void c_tsymm(char side, char uplo, int m, int n, CPX alpha, CPX *a, \
                    int lda, CPX *b, int ldb, CPX beta, CPX *c, int ldc)
{
    c_zsymm(side,uplo,m,n,alpha,a,lda,b,ldb,beta,c,ldc);
}

/************************************************************************************************/

template <typename T>
inline void c_ttrsm(char side, char uplo, char transa, char diag, int m, int n, \
                    T alpha, T *a, int lda, T *b, int ldb);

template <>
inline void c_ttrsm(char side, char uplo, char transa, char diag, int m, int n, \
                    double alpha, double *a, int lda, double *b, int ldb)
{
    c_dtrsm(side,uplo,transa,diag,m,n,alpha,a,lda,b,ldb);
}

template <>
inline void c_ttrsm(char side, char uplo, char transa, char diag, int m, int n, \
                    CPX alpha, CPX *a, int lda, CPX *b, int ldb)
{
    c_ztrsm(side,uplo,transa,diag,m,n,alpha,a,lda,b,ldb);
}

/************************************************************************************************/

template <typename T>
inline void tgemm_dev(char transa, char transb, int m, int n, int k, T alpha, T *a,\
		      int lda, T *b, int ldb, T beta, T *c, int ldc, magma_queue_t queue);

template <>
inline void tgemm_dev(char transa, char transb, int m, int n, int k, double alpha,\
		      double *a, int lda, double *b, int ldb, double beta, double *c, int ldc, magma_queue_t queue)
{
    magma_trans_t magma_transa = magma_trans_const(transa);
    magma_trans_t magma_transb = magma_trans_const(transb);

    //dgemm_on_dev(handle,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    magma_dgemm(magma_transa,magma_transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,queue);
}

template <>
inline void tgemm_dev(char transa, char transb, int m, int n, int k, CPX alpha,\
		      CPX *a, int lda, CPX *b, int ldb, CPX beta, CPX *c, int ldc, magma_queue_t queue)
{
    magma_trans_t magma_transa = magma_trans_const(transa);
    magma_trans_t magma_transb = magma_trans_const(transb);

    //zgemm_on_dev(handle,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    magma_zgemm(magma_transa,magma_transb,m,n,k,*reinterpret_cast<magmaDoubleComplex*>(&alpha),(magmaDoubleComplex_ptr)a,lda,(magmaDoubleComplex_ptr)b,ldb,*reinterpret_cast<magmaDoubleComplex*>(&beta),(magmaDoubleComplex_ptr)c,ldc,queue);
}

/************************************************************************************************/

template <typename T>
inline void taxpy_dev(void *handle,int n,T alpha,T *x,int incx,T *y,int incy);

template <>
inline void taxpy_dev(void *handle,int n,double alpha,double *x,int incx,double *y,int incy)
{
    daxpy_on_dev(handle,n,alpha,x,incx,y,incy);
}

template <>
inline void taxpy_dev(void *handle,int n,CPX alpha,CPX *x,int incx,CPX *y,int incy)
{
    zaxpy_on_dev(handle,n,alpha,x,incx,y,incy);
}

/************************************************************************************************/

template <typename T>
inline void tasum_dev(void *handle, int n, T *x, int incx, T *result);

template <>
inline void tasum_dev(void *handle, int n, double *x, int incx, double *result)
{
    dasum_on_dev(handle, n, x, incx, result);
}

template <>
inline void tasum_dev(void *handle, int n, CPX *x, int incx, CPX *result)
{
    double dRes = 0.0;
    zasum_on_dev(handle, n, x, incx, &dRes);
    *result = CPX(dRes, 0.0);
}

/************************************************************************************************/

template <typename T>
inline void tsum_dev(int n, T *x, int incx, T *result, magma_queue_t queue);

template <>
inline void tsum_dev(int n, double *x, int incx, double *result, magma_queue_t queue)
{
    dsum_on_dev(n, x, incx, result, queue);
}

template <>
inline void tsum_dev(int n, CPX *x, int incx, CPX *result, magma_queue_t queue)
{
    zsum_on_dev(n, x, incx, result, queue);
}

/*magma*****************************************************************************************/

template <typename T>
inline void tgetrf_dev(int m,int n,T *a,int lda,int *ipiv,int *info);

template <>
inline void tgetrf_dev(int m,int n,double *a,int lda,int *ipiv,int *info)
{
    magma_dgetrf_gpu(m,n,a,lda,ipiv,info);
}

template <>
inline void tgetrf_dev(int m,int n,CPX *a,int lda,int *ipiv,int *info)
{
    magma_zgetrf_gpu(m,n,(magmaDoubleComplex_ptr)a,lda,ipiv,info);
}

/************************************************************************************************/

template <typename T>
inline void tgetrs_dev(char transa,int n,int nrhs,T *a,int lda,int *ipiv,T *b,int ldb,int *info);

template <>
inline void tgetrs_dev(char transa,int n,int nrhs,double *a,int lda,int *ipiv,double *b,int ldb,\
		       int *info)
{
    magma_trans_t magma_transa = magma_trans_const(transa);

    magma_dgetrs_gpu(magma_transa,n,nrhs,a,lda,ipiv,b,ldb,info);
}

template <>
inline void tgetrs_dev(char transa,int n,int nrhs,CPX *a,int lda,int *ipiv,CPX *b,int ldb,\
		       int *info)
{
    magma_trans_t magma_transa = magma_trans_const(transa);

    magma_zgetrs_gpu(magma_transa,n,nrhs,(magmaDoubleComplex_ptr)a,lda,ipiv,\
		     (magmaDoubleComplex_ptr)b,ldb,info);
}

/************************************************************************************************/

template <typename T>
inline void tgesv_dev(int n,int nrhs,T *a,int lda,int *ipiv,T *b,int ldb,int type,int *info);

template <>
inline void tgesv_dev(int n,int nrhs,double *a,int lda,int *ipiv,double *b,int ldb,int type,int *info)
{
    if(type){
	magma_dgesv_nopiv_gpu(n,nrhs,a,lda,b,ldb,info);  
    }else{
      /*                                                                                                                   
        magma_dgesv_nopiv_gpu(n,nrhs,a,lda,b,ldb,info);                                                                    
      */
        magma_dsysv_nopiv_gpu(MagmaLower,n,nrhs,a,lda,b,ldb,info);
    }
}

template <>
inline void tgesv_dev(int n,int nrhs,CPX *a,int lda,int *ipiv,CPX *b,int ldb,int type,int *info)
{

    if(type){
	magma_zgesv_nopiv_gpu(n,nrhs,(magmaDoubleComplex_ptr)a,lda,(magmaDoubleComplex_ptr)b,\
			      ldb,info);
    }else{
      /*                                                                                                 
        magma_zgesv_nopiv_gpu(n,nrhs,(magmaDoubleComplex_ptr)a,lda,(magmaDoubleComplex_ptr)b,\
			      ldb,info);                                                                                         
      */
        magma_zhesv_nopiv_gpu(MagmaLower,n,nrhs,(magmaDoubleComplex_ptr)a,lda,(magmaDoubleComplex_ptr)b,\
                              ldb,info);
    }
}

/************************************************************************************************/

template <typename T>
inline void tgetri_dev(int n,T *a,int lda,int *ipiv,T *work,int lwork,int *info);

template <>
inline void tgetri_dev(int n,double *a,int lda,int *ipiv,double *work,int lwork,int *info)
{
    magma_dgetri_gpu(n,a,lda,ipiv,work,lwork,info);
}

template <>
inline void tgetri_dev(int n,CPX *a,int lda,int *ipiv,CPX *work,int lwork,int *info)
{
    magma_zgetri_gpu(n,(magmaDoubleComplex_ptr)a,lda,ipiv,(magmaDoubleComplex_ptr)work,lwork,info);
}

/************************************************************************************************/

template <typename T>
inline void ttrtri_dev(char uplo,char diag,int n,T *a,int lda,int *info);

template <>
inline void ttrtri_dev(char uplo,char diag,int n,double *a,int lda,int *info)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    magma_diag_t magma_diag = magma_diag_const(diag);

    magma_dtrtri_gpu(magma_uplo,magma_diag,n,a,lda,info);
}

template <>
inline void ttrtri_dev(char uplo,char diag,int n,CPX *a,int lda,int *info)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    magma_diag_t magma_diag = magma_diag_const(diag);

    magma_ztrtri_gpu(magma_uplo,magma_diag,n,(magmaDoubleComplex_ptr)a,lda,info);
}

/************************************************************************************************/

inline void zgetrf_nopiv_dev(int m,int n,CPX *a,int lda,int *info)
{
    magma_zgetrf_nopiv_gpu(m,n,(magmaDoubleComplex_ptr)a,lda,info);
}

/************************************************************************************************/

inline void zgetri_dev(int n,CPX *a,int lda,int *ipiv,CPX *work,int lwork,int *info)
{
    magma_zgetri_gpu(n,(magmaDoubleComplex_ptr)a,lda,ipiv,(magmaDoubleComplex_ptr)work,lwork,info);
}

/************************************************************************************************/

inline void zgetrs_nopiv_dev(char transa,int n,int nrhs,CPX *a,int lda,CPX *b,int ldb,int *info)
{
    magma_trans_t magma_transa = magma_trans_const(transa);

    magma_zgetrs_nopiv_gpu(magma_transa,n,nrhs,(magmaDoubleComplex_ptr)a,lda,\
			   (magmaDoubleComplex_ptr)b,ldb,info);
}

/************************************************************************************************/

template <typename T>
inline void tpotrf_dev(char uplo,int n,T *a,int lda,int *info);

template <>
inline void tpotrf_dev(char uplo,int n,double *a,int lda,int *info)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magma_dpotrf_gpu(magma_uplo,n,a,lda,info);
}

template <>
inline void tpotrf_dev(char uplo,int n,CPX *a,int lda,int *info)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magma_zpotrf_gpu(magma_uplo,n,(magmaDoubleComplex_ptr)a,lda,info);
}

/************************************************************************************************/

template <typename T>
inline void ttrsm_dev(char side, char uplo, char trans, char diag, int m, int n, T alpha, T *a, int lda, T *b, int ldb, magma_queue_t queue);

template <>
inline void ttrsm_dev(char side, char uplo, char trans, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb, magma_queue_t queue)
{
    magma_side_t magma_side = magma_side_const(side);
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    magma_trans_t magma_trans = magma_trans_const(trans);
    magma_diag_t magma_diag = magma_diag_const(diag);

    magma_dtrsm(magma_side, magma_uplo, magma_trans, magma_diag, m, n, alpha, a, lda, b, ldb, queue);
}

template <>
inline void ttrsm_dev(char side, char uplo, char trans, char diag, int m, int n, CPX alpha, CPX *a, int lda, CPX *b, int ldb, magma_queue_t queue)
{
    magma_side_t magma_side = magma_side_const(side);
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);
    magma_trans_t magma_trans = magma_trans_const(trans);
    magma_diag_t magma_diag = magma_diag_const(diag);

    magma_ztrsm(magma_side, magma_uplo, magma_trans, magma_diag, m, n, *reinterpret_cast<magmaDoubleComplex*>(&alpha), (magmaDoubleComplex_ptr)a, lda, (magmaDoubleComplex_ptr)b, ldb, queue);
}

/************************************************************************************************/

template <typename T>
inline void tlacpy_dev(char uplo, int m, int n, T *a, int lda, T *b, int ldb, magma_queue_t queue);

template <>
inline void tlacpy_dev(char uplo, int m, int n, double *a, int lda, double *b, int ldb, magma_queue_t queue)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magmablas_dlacpy(magma_uplo, m, n, a, lda, b, ldb, queue);
}

template <>
inline void tlacpy_dev(char uplo, int m, int n, CPX *a, int lda, CPX *b, int ldb, magma_queue_t queue)
{
    magma_uplo_t magma_uplo = magma_uplo_const(uplo);

    magmablas_zlacpy(magma_uplo, m, n, (magmaDoubleComplex_ptr)a, lda, (magmaDoubleComplex_ptr)b, ldb, queue);
}

/************************************************************************************************/

/*Lapack*****************************************************************************************/

inline void c_dgetrf(int m, int n, double *a, int lda, int *ipiv, int *info)
{
    fortran_name(dgetrf,DGETRF)(&m,&n,a,&lda,ipiv,info);
}

/************************************************************************************************/

inline void c_dgetrs(char transa, int n, int nrhs, double *a, int lda, int *ipiv, double *b, \
                   int ldb, int *info)
{
    fortran_name(dgetrs,DGETRS)(&transa,&n,&nrhs,a,&lda,ipiv,b,&ldb,info);
}

/************************************************************************************************/

inline void c_zgetrf(int m, int n, CPX *a, int lda, int *ipiv, int *info)
{
    fortran_name(zgetrf,ZGETRF)(&m,&n,a,&lda,ipiv,info);
}

/************************************************************************************************/

inline void c_zgetrs(char transa, int n, int nrhs, CPX *a, int lda, int *ipiv, CPX *b, \
                   int ldb, int *info)
{
    fortran_name(zgetrs,ZGETRS)(&transa,&n,&nrhs,a,&lda,ipiv,b,&ldb,info);
}

/************************************************************************************************/

inline void c_zgetri(int n,CPX *a,int lda,int *ipiv,CPX *work,int lwork,int *info)
{
    fortran_name(zgetri,ZGETRI)(&n,a,&lda,ipiv,work,&lwork,info);
}

/************************************************************************************************/

inline void c_dgeev(char jobvl, char jobvr, int n, double *a, int lda, double *wr, double *wi, \
                    double *vl, int ldvl, double *vr, int ldvr, double *work, int lwork, \
                    int *info)
{
    fortran_name(dgeev,DGEEV)(&jobvl,&jobvr,&n,a,&lda,wr,wi,vl,&ldvl,vr,&ldvr,work,&lwork,info);
}

/************************************************************************************************/

inline void c_dsyev(char JOBZ,char UPLO,int N,double *A,int LDA,double *W,double *WORK,int LWORK,\
                    int *INFO)
{
  fortran_name(dsyev,DSYEV)(&JOBZ,&UPLO,&N,A,&LDA,W,WORK,&LWORK,INFO);
}

/************************************************************************************************/

inline void c_dggev(char jobvl, char jobvr, int n, double *a, int lda, double *b, int ldb, \
                    double *alphar, double *alphai, double *beta, double *vl, int ldvl, \
                    double *vr, int ldvr, double *work, int lwork, int *info)
{
    fortran_name(dggev,DGGEV)(&jobvl,&jobvr,&n,a,&lda,b,&ldb,alphar,alphai,beta,vl,&ldvl,vr,\
                              &ldvr,work,&lwork,info);
}

/************************************************************************************************/

inline void c_zggev(char jobvl, char jobvr, int n, CPX *a, int lda, CPX *b, int ldb, CPX *alpha,\
                    CPX *beta, CPX *vl, int ldvl, CPX *vr, int ldvr, CPX *work, int lwork, \
		    double *rwork,int *info)
{
    fortran_name(zggev,ZGGEV)(&jobvl,&jobvr,&n,a,&lda,b,&ldb,alpha,beta,vl,&ldvl,vr,&ldvr,\
			      work,&lwork,rwork,info);
}

/************************************************************************************************/

inline void c_zgeev(char jobvl, char jobvr, int n, CPX *a, int lda, CPX *w, CPX *vl, int ldvl, \
                    CPX *vr, int ldvr, CPX *work, int lwork, double *rwork, int *info)
{
    fortran_name(zgeev,ZGEEV)(&jobvl,&jobvr,&n,a,&lda,w,vl,&ldvl,vr,&ldvr,work,&lwork,rwork,info);
}

/************************************************************************************************/

inline void c_zheev(char jobvl, char uplo, int n, CPX *a, int lda, double *w, CPX *work, \
		    int lwork, double *rwork, int *info)
{
    fortran_name(zheev,ZHEEV)(&jobvl,&uplo,&n,a,&lda,w,work,&lwork,rwork,info);
}

/************************************************************************************************/

inline void c_dgetri(int n,double *a,int lda,int *ipiv,double *work,int lwork,int *info)
{
    fortran_name(dgetri,DGETRI)(&n,a,&lda,ipiv,work,&lwork,info);
}

/************************************************************************************************/

inline void c_dsytri(char uplo,int n,double *a,int lda,int *ipiv,double *work,int *info)
{
    fortran_name(dsytri,DSYTRI)(&uplo,&n,a,&lda,ipiv,work,info);
}

/************************************************************************************************/

inline void c_zhetrf(char uplo,int n,CPX *a,int lda,int *ipiv,CPX *work,int lwork,int *info)
{
    fortran_name(zhetrf,ZHETRF)(&uplo,&n,a,&lda,ipiv,work,&lwork,info);
}

/************************************************************************************************/

inline void c_zhetri(char uplo,int n,CPX *a,int lda,int *ipiv,CPX *work,int *info)
{
    fortran_name(zhetri,ZHETRI)(&uplo,&n,a,&lda,ipiv,work,info);
}

/************************************************************************************************/

inline void c_zhetrs(char uplo,int n,int nrhs,CPX *a,int lda,int *ipiv,CPX *b,int ldb,\
                     int *info)
{
    fortran_name(zhetrs,ZHETRS)(&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,info);
}

/************************************************************************************************/

inline void c_dsysv(char uplo,int n,int nrhs,double *a,int lda,int *ipiv,double *b,int ldb,\
                    double *work,int lwork,int *info)
{
    fortran_name(dsysv,DSYSV)(&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,work,&lwork,info);
}

/************************************************************************************************/

inline void c_dsytrf(char uplo,int n,double *a,int lda,int *ipiv,double *work,int lwork,int *info)
{
    fortran_name(dsytrf,DSYTRF)(&uplo,&n,a,&lda,ipiv,work,&lwork,info);
}

/************************************************************************************************/

inline void c_dsytrs(char uplo,int n,int nrhs,double *a,int lda,int *ipiv,double *b,int ldb,\
                     int *info)
{
    fortran_name(dsytrs,DSYTRS)(&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,info);
}

/************************************************************************************************/

inline void c_dstebz(char *range,char *order,int *iter,double *vl,double *vu,int *il,int *iu,\
                     double *abstol,double *diag,double *offd,int *neval,int *nsplit,\
                     double *eval,int *iblock,int *isplit,double *work,int *iwork,int *info)
{
    fortran_name(dstebz,DSTEBZ)(range,order,iter,vl,vu,il,iu,abstol,diag,offd,neval,nsplit,eval,\
	                            iblock,isplit,work,iwork,info);
}

/************************************************************************************************/

inline void c_zlarnv(int *idist,int *iseed,int n,CPX *x)
{
    fortran_name(zlarnv,ZLARNV)(idist,iseed,&n,x);
}

/************************************************************************************************/

inline void c_dgesdd(char jobz,int m,int n,double *a,int lda,double *s,double *u,int ldu,\
                     double *vt,int ldvt,double *work,int lwork,int *iwork,int *info)
{
    fortran_name(dgesdd,DGESDD)(&jobz,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&lwork,iwork,info);
}

/************************************************************************************************/

inline void c_zgesdd(char jobz,int m,int n,CPX *a,int lda,double *s,CPX *u,int ldu,\
                     CPX *vt,int ldvt,CPX *work,int lwork,double *rwork,int *iwork,int *info)
{
    fortran_name(zgesdd,ZGESDD)(&jobz,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&lwork,rwork,iwork,info);
}

/************************************************************************************************/

template <typename T>
inline void copy_csr_to_device(int size,int n_nonzeros,int *hedge_i,int *hindex_j,T *hnnz,\
			       int *dedge_i,int *dindex_j,T *dnnz);

template <>
inline void copy_csr_to_device(int size,int n_nonzeros,int *hedge_i,int *hindex_j,double *hnnz,\
			       int *dedge_i,int *dindex_j,double *dnnz)
{
    d_copy_csr_to_device(size,n_nonzeros,hedge_i,hindex_j,hnnz,dedge_i,dindex_j,dnnz);
}

template <>
inline void copy_csr_to_device(int size,int n_nonzeros,int *hedge_i,int *hindex_j,CPX *hnnz,\
			       int *dedge_i,int *dindex_j,CPX *dnnz)
{
    z_copy_csr_to_device(size,n_nonzeros,hedge_i,hindex_j,hnnz,dedge_i,dindex_j,dnnz);
}

/************************************************************************************************/

template <typename T>
inline void init_var_on_dev(T *var,int N,cudaStream_t stream);

template <>
inline void init_var_on_dev(double *var,int N,cudaStream_t stream){
    d_init_var_on_dev(var,N,stream);
}

template <>
inline void init_var_on_dev(CPX *var,int N,cudaStream_t stream){
    z_init_var_on_dev(var,N,stream);
}

/************************************************************************************************/

template <typename T>
inline void init_eye_on_dev(T *var,int N,cudaStream_t stream);

template <>
inline void init_eye_on_dev(double *var,int N,cudaStream_t stream){
    d_init_eye_on_dev(var,N,stream);
}

template <>
inline void init_eye_on_dev(CPX *var,int N,cudaStream_t stream){
    z_init_eye_on_dev(var,N,stream);
}

/************************************************************************************************/

template <typename T>
inline void csr_mult_f(void *handle,int m,int n,int k,int n_nonzeros,int *Aedge_i,int *Aindex_j,\
		       T *Annz,T alpha,T *B,T beta,T *C);

template <>
inline void csr_mult_f(void *handle,int m,int n,int k,int n_nonzeros,int *Aedge_i,int *Aindex_j,\
		       double *Annz,double alpha,double *B,double beta,double *C)
{
    d_csr_mult_f(handle,m,n,k,n_nonzeros,Aedge_i,Aindex_j,Annz,alpha,B,beta,C);
}

template <>
inline void csr_mult_f(void *handle,int m,int n,int k,int n_nonzeros,int *Aedge_i,int *Aindex_j,\
		       CPX *Annz,CPX alpha,CPX *B,CPX beta,CPX *C)
{
    z_csr_mult_f(handle,m,n,k,n_nonzeros,Aedge_i,Aindex_j,Annz,alpha,B,beta,C);
}

/************************************************************************************************/

template <typename T>
inline void transpose_matrix(T *odata,T *idata,int size_x,int size_y);

template <>
inline void transpose_matrix(double *odata,double *idata,int size_x,int size_y)
{
    d_transpose_matrix(odata,idata,size_x,size_y);
}

template <>
inline void transpose_matrix(CPX *odata,CPX *idata,int size_x,int size_y)
{
    z_transpose_matrix(odata,idata,size_x,size_y);
}

/************************************************************************************************/

template <typename T>
inline void extract_diag_on_dev(T *D,int *edge_i,int *index_j,T *nnz,int NR,\
     int imin,int imax,int shift,int findx,cudaStream_t stream);

template <>
inline void extract_diag_on_dev(double *D,int *edge_i,int *index_j,double *nnz,int NR,\
     int imin,int imax,int shift,int findx,cudaStream_t stream){
    d_extract_diag_on_dev(D,edge_i,index_j,nnz,NR,imin,imax,shift,findx,stream);
}

template <>
inline void extract_diag_on_dev(CPX *D,int *edge_i,int *index_j,CPX *nnz,int NR,\
     int imin,int imax,int shift,int findx,cudaStream_t stream){
    z_extract_diag_on_dev(D,edge_i,index_j,nnz,NR,imin,imax,shift,findx,stream);
}

/************************************************************************************************/

template <typename T>
inline void extract_not_diag_on_dev(T *D,int *edge_i,int *index_j,T *nnz,int NR,\
     int imin,int imax,int jmin,int side,int shift,int findx,cudaStream_t stream);

template <>
inline void extract_not_diag_on_dev(double *D,int *edge_i,int *index_j,double *nnz,int NR,\
     int imin,int imax,int jmin,int side,int shift,int findx,cudaStream_t stream){
    d_extract_not_diag_on_dev(D,edge_i,index_j,nnz,NR,imin,imax,jmin,side,shift,findx,stream);
}

template <>
inline void extract_not_diag_on_dev(CPX *D,int *edge_i,int *index_j,CPX *nnz,int NR,\
     int imin,int imax,int jmin,int side,int shift,int findx,cudaStream_t stream){
    z_extract_not_diag_on_dev(D,edge_i,index_j,nnz,NR,imin,imax,jmin,side,shift,findx,stream);
}

/************************************************************************************************/

template <typename T>
inline void tril_dev(T *A, int lda, int N);

template <>
inline void tril_dev(double *A, int lda, int N)
{
    d_tril_on_dev(A, lda, N);
}

template <>
inline void tril_dev(CPX *A, int lda, int N)
{
    z_tril_on_dev(A, lda, N);
}

/************************************************************************************************/

template <typename T>
inline void indexed_copy_dev(T *src, T *dst, size_t *index, size_t N);

template <>
inline void indexed_copy_dev(double *src, double *dst, size_t *index, size_t N)
{
    d_indexed_copy_on_dev(src, dst, index, N);
}

template <>
inline void indexed_copy_dev(CPX *src, CPX *dst, size_t *index, size_t N)
{
    z_indexed_copy_on_dev(src, dst, index, N);
}

/************************************************************************************************/

template <typename T>
inline void indexed_copy_offset_dev(T *src, T *dst, size_t *index, size_t N, size_t offset);

template <>
inline void indexed_copy_offset_dev(double *src, double *dst, size_t *index, size_t N, size_t offset)
{
    d_indexed_copy_offset_on_dev(src, dst, index, N, offset);
}

template <>
inline void indexed_copy_offset_dev(CPX *src, CPX *dst, size_t *index, size_t N, size_t offset)
{
    z_indexed_copy_offset_on_dev(src, dst, index, N, offset);
}

/************************************************************************************************/

template <typename T>
inline void indexed_copy(T *src, T *dst, size_t *index, size_t N)
{
   #pragma omp parallel for
   for (int i = 0; i < N; i++)
   {
      dst[i] = src[index[i]];
   }
}

/************************************************************************************************/

template <typename T>
inline void indexed_log_sum(T *x, size_t *index, size_t N, T *sum)
{
   #pragma omp parallel for reduction(+:sum[:1])
   for (int i = 0; i < N; i++)
   {
      *sum += log(x[index[i]]);
   }
}

/************************************************************************************************/

template <typename T>
inline void log_dev(T *x, size_t N);

template <>
inline void log_dev(double *x, size_t N)
{
    d_log_on_dev(x, N);
}

template <>
inline void log_dev(CPX *x, size_t N)
{
    z_log_on_dev(x, N);
}

/************************************************************************************************/

template <typename T>
inline void fill_dev(T *x, const T value, size_t N);

template <>
inline void fill_dev(double *x, const double value, size_t N)
{
    d_fill_on_dev(x, value, N);
}

template <>
inline void fill_dev(CPX *x, const CPX value, size_t N)
{
    z_fill_on_dev(x, value, N);
}

/************************************************************************************************/

template <typename T>
inline void init_block_matrix_dev(T *M, size_t *ia, size_t *ja, T *a, size_t nnz, size_t ns, size_t nt, size_t nd);

template <>
inline void init_block_matrix_dev(double *M, size_t *ia, size_t *ja, double *a, size_t nnz, size_t ns, size_t nt, size_t nd)
{
    d_init_block_matrix_on_dev(M, ia, ja, a, nnz, ns, nt, nd);
}

template <>
inline void init_block_matrix_dev(CPX *M, size_t *ia, size_t *ja, CPX *a, size_t nnz, size_t ns, size_t nt, size_t nd)
{
    z_init_block_matrix_on_dev(M, ia, ja, a, nnz, ns, nt, nd);
}

/************************************************************************************************/

template <typename T>
inline void init_supernode_dev(T *M, size_t *ia, size_t *ja, T *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd);

template <>
inline void init_supernode_dev(double *M, size_t *ia, size_t *ja, double *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd)
{
    d_init_supernode_on_dev(M, ia, ja, a, supernode, supernode_nnz, supernode_offset, ns, nt, nd);
}

template <>
inline void init_supernode_dev(CPX *M, size_t *ia, size_t *ja, CPX *a, size_t supernode, size_t supernode_nnz, size_t supernode_offset, size_t ns, size_t nt, size_t nd)
{
    z_init_supernode_on_dev(M, ia, ja, a, supernode, supernode_nnz, supernode_offset, ns, nt, nd);
}

/************************************************************************************************/

#endif




