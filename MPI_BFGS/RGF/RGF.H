#ifndef __RGF
#define __RGF

#include <string.h>

#include "cublas_v2.h"
#include "Types.H"
#include "Utilities.H"
#include "CWC_utility.H"

template <class T>
class RGF{
	
public:

    RGF(size_t*,size_t*,T*,size_t,size_t,size_t);

    ~RGF();
    double factorize();
    double solve(T*,size_t);
    double solve(T*,T*,size_t);
    double RGFdiag(T*);
    T logDet();
    double residualNorm(T*,T*);
    double residualNormNormalized(T*,T*);
    double flop_count_factorise();
				
private:

    size_t *matrix_ia;
    size_t *matrix_ja;
    T      *matrix_a;
    size_t matrix_size;
    size_t matrix_n_nonzeros;
    size_t matrix_ns;
    size_t matrix_nt;
    size_t matrix_nd;
    size_t *Bmin;
    size_t *Bmax;
    size_t NBlock;
    size_t *diag_pos;

    size_t b_size;
    bool MF_dev_allocated = false;
    bool factorization_completed = false;

    magma_queue_t magma_queue;
    cudaStream_t stream_c;
    void *cublas_handle;

    T *MF;
    T *blockR_dev;
    T *blockM_dev;
    T *blockP_dev;
    T *blockDense_dev;
    T *rhs;
    size_t *ia_dev;
    size_t *ja_dev;
    T      *a_dev;
    T *diag_dev;
    size_t *diag_pos_dev;

    inline size_t mf_block_index(size_t, size_t);
    inline size_t mf_block_lda(size_t, size_t);
    inline size_t mf_dense_block_index(size_t);
    inline size_t mf_dense_block_offset(size_t);
    inline size_t mf_dense_block_lda(size_t);

    double FirstStageFactor();
    double SecondStageSolve(size_t);
    double ThirdStageRGF(T*,T*);
    void create_blocks();
    inline void init_supernode(T *M_dev, size_t supernode);
    inline void copy_supernode_to_host(T *M_dev, size_t supernode);
    inline void copy_supernode_to_device(T *M_dev, size_t supernode);
    inline void copy_supernode_diag(T *src, size_t supernode);
    inline void swap_pointers(T **ptr1, T **ptr2);

   T f_one();
   T f_zero();
};

/************************************************************************************************/

template <class T>
RGF<T>::RGF(size_t *ia, size_t *ja, T *a, size_t ns, size_t nt, size_t nd)
{
    matrix_ia = ia;
    matrix_ja = ja;
    matrix_a = a;
    matrix_ns = ns;
    matrix_nt = nt;
    matrix_nd = nd;

    matrix_size = ns*nt + nd;
    matrix_n_nonzeros = 2*(nt-1)*ns*ns + ns*ns + matrix_size*nd;

    if (nd > 0)
       NBlock = nt + 1;
    else
       NBlock = nt;
    Bmin   = new size_t[NBlock];
    Bmax   = new size_t[NBlock];
    for (size_t i = 0; i < nt; i++)
    {
       Bmin[i] = i*ns;
       Bmax[i] = (i+1)*ns;
    }
    if (nd > 0)
    {
       Bmin[nt] = nt*ns;
       Bmax[nt] = nt*ns + nd;
    }

    // blocks with lda = 2*ns + nd
    diag_pos = new size_t[matrix_size];
    size_t IB;
    for (IB = 0; IB < nt-1; IB++)
    {
       for (size_t i = 0; i < ns; i++)
       {
          diag_pos[IB*ns+i] = IB * ns*(2*ns+nd) + i*(2*ns+nd+1);
       }
    }
    // last block with lda = ns + nd
    IB = nt-1;
    for (size_t i = 0; i < ns; i++)
    {
       diag_pos[IB*ns+i] = IB * ns*(2*ns+nd) + i*(ns+nd+1);
    }
    // dense block with lda = ns + nd
    if (nd > 0)
    {
       IB = nt;
       for (size_t i = 0; i < nd; i++)
       {
          diag_pos[nt*ns+i] = (nt-1) * ns*(2*ns+nd) + ns*(ns+nd) + i*(nd+1);
       }
    }

    magma_init();
    magma_device_t device;
    magma_getdevice(&device);
    magma_queue_create_from_cuda(device, NULL, NULL, NULL, &magma_queue);
    stream_c = magma_queue_get_cuda_stream(magma_queue);
    cublas_handle = magma_queue_get_cublas_handle(magma_queue);
}

/************************************************************************************************/

template <class T>
RGF<T>::~RGF()
{
   if (MF_dev_allocated)
   {
      size_t max_supernode_nnz = matrix_nt > 1 ? matrix_ns*(2*matrix_ns+matrix_nd) : matrix_ns*(matrix_ns+matrix_nd);
      size_t dense_supernode_nnz = matrix_nd > 0 ? matrix_nd*matrix_nd : 0;
      deallocate_data_on_dev(blockR_dev,max_supernode_nnz*sizeof(T));
      deallocate_data_on_dev(blockM_dev,max_supernode_nnz*sizeof(T));
      deallocate_data_on_dev(blockP_dev,max_supernode_nnz*sizeof(T));
      deallocate_data_on_dev(blockDense_dev,dense_supernode_nnz*sizeof(T));
      delete[] MF;
   }

   magma_queue_destroy(magma_queue);
   magma_finalize();

   delete[] Bmin;
   delete[] Bmax;
   delete[] diag_pos;
}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_block_index(size_t r, size_t c)
{
   return diag_pos[c*b_size] + (r-c)*b_size;
}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_block_lda(size_t r, size_t c)
{
   //return matrix->index_i[c*b_size];
   // two blocks
   if (c < matrix_nt-1)
      return 2*matrix_ns + matrix_nd;
   // one block
   if (c < matrix_nt)
      return matrix_ns + matrix_nd;
   // dense block
   else
      return matrix_nd;
}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_dense_block_index(size_t i)
{
   if (i < matrix_nt-1)
      return diag_pos[i*b_size] + 2*b_size;
   else if (i == matrix_nt-1)
      return diag_pos[i*b_size] + b_size;
   else
      return diag_pos[i*b_size];
}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_dense_block_offset(size_t i)
{
   if (i < matrix_nt-1)
      return 2*b_size;
   else if (i == matrix_nt-1)
      return b_size;
   else
      return 0;
}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_dense_block_lda(size_t i)
{
   return mf_block_lda(i, i);;
}

/************************************************************************************************/

template <class T>
double RGF<T>::FirstStageFactor()
{
    int info;
    size_t IB;
    size_t NR,NM;
    T ONE      = f_one();

    // count floating point operations
    double flops = 0;

//tpotrf_dev('L', NR, M1_dev ,NR ,&info)
//rj dpotrf MF[0,0]
//rj dtrsm RLTN MF[0,0] MF[1,0]

    IB = 0;
    NR = Bmax[0]-Bmin[0];

    init_supernode(blockR_dev, IB);
    if (matrix_nd > 0)
    {
       init_supernode(blockDense_dev, matrix_nt);
    }

    // CHOLESKY FACTORISATION FIRST DIAGONAL BLOCK
    tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
    //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(0, 0), mf_block_lda(0, 0));
    flops += 0.5 * NR * NR * NR;

    //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
    if (matrix_nt > 1)
    {
       // UPDATE COLUMNS ACCORDING TO CHOLESKY FACTORISATION
       ttrsm_dev('R', 'L', 'T', 'N', NR+matrix_nd, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[NR], mf_block_lda(1, 0), magma_queue);
       flops += 2.0*(NR + matrix_nd) * NR * NR;
    }

    copy_supernode_to_host(blockR_dev, IB);
    
    //rj IB = 1; IB < NBlock; IB++
    for (IB = 1; IB < matrix_nt-1; IB++)
    {
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       swap_pointers(&blockR_dev, &blockM_dev);
       init_supernode(blockR_dev, IB);

       // UPDATE NEXT DIAGONAL BLOCK 
       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       // todo rj: 3-last parameter ZERO in PARDISO
       tgemm_dev('N', 'T', NR, NR, NM, -ONE, &blockM_dev[NM], mf_block_lda(IB, IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, blockR_dev, mf_block_lda(IB, IB), magma_queue);
       flops += 2.0 * NR * NR * NR;

       if (matrix_nd > 0)
       {
          // Update dense rows of super node IB
          tgemm_dev('N', 'T', matrix_nd, NR, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue);
          // NM = NR
          flops += 2.0*matrix_nd * NR * NM;
          
          // update last diagonal block
          tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue);
          flops += 2.0 * matrix_nd * matrix_nd * NM;
       }
       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
       flops += 0.5 * NR * NR * NR;
       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
       ttrsm_dev('R', 'L', 'T', 'N', NR+matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[NR], mf_block_lda(IB+1, IB), magma_queue);
       flops += 2.0 *(NR + matrix_nd) * NR * NR;
       //printf("RJ: ttrsm NR: %d, a: %d, lda: %d, b: %d, ldb: %d\n", NR, mf_block_index(IB-1, IB-1), mf_block_lda(IB-1, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1));

       copy_supernode_to_host(blockR_dev, IB);
    }

    if (matrix_nt > 1)
    {
       IB = matrix_nt-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       swap_pointers(&blockR_dev, &blockM_dev);
       init_supernode(blockR_dev, IB);

       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       // todo rj: 3-last parameter ZERO in PARDISO
       tgemm_dev('N', 'T', NR, NR, NM, -ONE, &blockM_dev[NM], mf_block_lda(IB, IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, blockR_dev, mf_block_lda(IB, IB), magma_queue);
       flops += 2.0* NR * NR * NR;

       if (matrix_nd > 0)
       {
          tgemm_dev('N', 'T', matrix_nd, NR, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue);
          flops += 2.0*matrix_nd * NR * NM;

          tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue);
          flops += 2.0* matrix_nd * matrix_nd * NM; 
       }
       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
       flops += 0.5 * NR * NR * NR;
    
       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
       if (matrix_nd > 0)
       {
          ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue);
          flops += 2.0*matrix_nd * NR * NR;
       
       }
       //printf("RJ: ttrsm NR: %d, a: %d, lda: %d, b: %d, ldb: %d\n", NR, mf_block_index(IB-1, IB-1), mf_block_lda(IB-1, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1));

       copy_supernode_to_host(blockR_dev, IB);
    }

    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       //rj NM is the same for all blocks 0..nt-1
       tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockR_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockR_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue);
       flops += 2.0* matrix_nd * matrix_nd * NM; 
   
       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       tpotrf_dev('L', NR, blockDense_dev, mf_block_lda(IB, IB), &info);
       flops += 0.5 * NR * NR * NR;

       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));

       copy_supernode_to_host(blockDense_dev, IB);
    }

    return flops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::SecondStageSolve(size_t nrhs)
{
    int info;
    size_t IB;
    size_t NR,NM,NP;
    T ONE      = f_one();

    //Forward pass
    NR = Bmax[0]-Bmin[0];

    // counting FLOPS
    double flops = 0;

    //rj dtrsm LLNN MF[0,0] rhs[0]
    // solve for first diagonal block 
    c_ttrsm('L', 'L', 'N', 'N', NR, nrhs, ONE, &MF[mf_block_index(0, 0)], mf_block_lda(0, 0), &rhs[Bmin[0]], matrix_size);
    flops += 2.0 * NR * NR * nrhs;

    //rj IB = 1; IB < NBlock; IB++
    for (IB = 1; IB < matrix_nt; IB++)
    {
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       //rj dgemm NN M[IB,IB-1] rhs[IB-1] rhs[IB]
       c_tgemm('N', 'N', NR, nrhs, NM, -ONE, &MF[mf_block_index(IB, IB-1)], mf_block_lda(IB, IB-1), &rhs[Bmin[IB-1]], matrix_size, ONE, &rhs[Bmin[IB]], matrix_size);
       flops += 2.0 * NR * nrhs * NM;
       //rj dtrsm LLNN MF[IB,IB] rhs[IB]
       c_ttrsm('L', 'L', 'N', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flops += 2.0*NR * NR * nrhs; 
    }

    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       //rj dgemm NN M[IB,IB-1] rhs[IB-1] rhs[IB]
       for (size_t i = 0; i < NBlock-1; i++)
       {
          c_tgemm('N', 'N', NR, nrhs, NM, -ONE, &MF[mf_dense_block_index(i)], mf_dense_block_lda(i), &rhs[Bmin[i]], matrix_size, ONE, &rhs[Bmin[IB]], matrix_size);
          flops += 2.0*NR * nrhs * NM;
       }
       //rj dtrsm LLNN MF[IB,IB] rhs[IB]
       c_ttrsm('L', 'L', 'N', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flops += 2.0*NR * NR * nrhs;
    }

    //Backward pass
    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       //rj dtrsm LLTN MF[NBlock-1,NBlock-1] rhs[NBlock-1]
       c_ttrsm('L', 'L', 'T', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flops += 2.0*NR * NR * nrhs;
       //rj dgemm TN M[IB,IB-1] rhs[IB] rhs[IB-1]
       for (size_t i = 0; i < NBlock-1; i++)
       {
          c_tgemm('T', 'N', NM, nrhs, NR, -ONE, &MF[mf_dense_block_index(i)], mf_dense_block_lda(i), &rhs[Bmin[IB]], matrix_size, ONE, &rhs[Bmin[i]], matrix_size);
          flops += 2.0*NM * nrhs * NR;
       }
    }

    for (IB = matrix_nt-1; IB > 0; IB--)
    {
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       //rj dtrsm LLTN MF[IB,IB] rhs[IB]
       c_ttrsm('L', 'L', 'T', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flops += 2.0* NR * NR * nrhs;
       //rj dgemm TN M[IB,IB-1] rhs[IB] rhs[IB-1]
       c_tgemm('T', 'N', NM, nrhs, NR, -ONE, &MF[mf_block_index(IB, IB-1)], mf_block_lda(IB, IB-1), &rhs[Bmin[IB]], matrix_size, ONE, &rhs[Bmin[IB-1]], matrix_size);
       flops += 2.0 * NM * nrhs * NR; 
    }

    IB = 0;
    NR = Bmax[IB]-Bmin[IB];

    //rj dtrsm LLTN MF[NBlock-1,NBlock-1] rhs[NBlock-1]
    c_ttrsm('L', 'L', 'T', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
    flops += 2.0 * NR * NR * nrhs;

    return flops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::ThirdStageRGF(T *tmp1_dev,T *tmp2_dev)
{
   int info;
   size_t IB;
   size_t NR,NM,NP,ND;
   T ONE      = f_one();
   T ZERO     = f_zero();

   ND = matrix_nd;

   double flops = 0;

   if (matrix_nd > 0)
   {
      //dense block
      IB = NBlock-1;
      NR = Bmax[IB]-Bmin[IB];

      tlacpy_dev('L', NR, NR, blockDense_dev, mf_block_lda(IB, IB), tmp1_dev, NR, magma_queue);
      tril_dev(tmp1_dev, NR, NR);
      ttrtri_dev('L', 'N', NR, tmp1_dev, NR, &info);
      flops += 0.5 * NR * NR * NR;
      tgemm_dev('T', 'N', NR, NR, NR, ONE, tmp1_dev, NR, tmp1_dev, NR, ZERO, blockDense_dev, mf_block_lda(IB, IB), magma_queue);
      flops += 2.0 * NR * NR * NR;

      //last non-dense block
      IB = NBlock-2;
      NR = Bmax[IB]-Bmin[IB];
      NP = Bmax[IB+1]-Bmin[IB+1];

      tril_dev(blockR_dev, mf_block_lda(IB, IB), NR);
      ttrtri_dev('L', 'N', NR, blockR_dev, mf_block_lda(IB, IB), &info);
      flops += 0.5 * NR * NR * NR;
      init_eye_on_dev(tmp1_dev,NR,0);

      tgemm_dev('T', 'N', NR, NP, NP, ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), blockDense_dev, mf_block_lda(IB+1, IB+1), ZERO, tmp2_dev, NR, magma_queue);
      flops += 2.0 * NR * NP * NP;
      tgemm_dev('N', 'N', NR, NR, NP, ONE, tmp2_dev, NR, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), ONE, tmp1_dev, NR, magma_queue);
      flops += 2.0 * NR * NR * NP;
      tgemm_dev('T', 'N', NR, NR, NR, ONE, blockR_dev, mf_block_lda(IB, IB), tmp1_dev, NR, ZERO, tmp2_dev, NR, magma_queue);
      flops += 2.0 * NR * NR * NR;
      tgemm_dev('N', 'N', NR, NR, NR, ONE, tmp2_dev, NR, blockR_dev, mf_block_lda(IB, IB), ZERO, tmp1_dev, NR, magma_queue);
      flops += 2.0 * NR * NR * NR;
      tlacpy_dev('F', NR, NR, tmp1_dev, NR, blockR_dev, mf_block_lda(IB, IB), magma_queue);

      copy_supernode_diag(blockDense_dev, NBlock-1);
   }
   else
   {
      //last non-dense block
      IB = NBlock-1;
      NR = Bmax[IB]-Bmin[IB];

      tlacpy_dev('L', NR, NR, blockR_dev, mf_block_lda(IB, IB), tmp1_dev, NR, magma_queue);
      tril_dev(tmp1_dev, NR, NR);
      ttrtri_dev('L', 'N', NR, tmp1_dev, NR, &info);
      flops += 0.5 * NR * NR * NR;
      tgemm_dev('T', 'N', NR, NR, NR, ONE, tmp1_dev, NR, tmp1_dev, NR, ZERO, blockR_dev, mf_block_lda(IB, IB), magma_queue);
   }

   //second-last non-dense block .. 0-block
   for (int IBi = matrix_nt-2; IBi > -1; IBi--)
   {
      IB = IBi;
      NR = Bmax[IB]-Bmin[IB];
      NP = Bmax[IB+1]-Bmin[IB+1];

      swap_pointers(&blockR_dev, &blockP_dev);
      copy_supernode_diag(blockP_dev, IB+1);
      copy_supernode_to_device(blockR_dev, IB);

      tril_dev(blockR_dev, mf_block_lda(IB, IB), NR);
      ttrtri_dev('L', 'N', NR, blockR_dev, mf_block_lda(IB, IB), &info);
      flops += 0.5 * NR * NR * NR;
      init_eye_on_dev(tmp1_dev,NR,0);

      tgemm_dev('T', 'N', NR, NP, NP, ONE, &blockR_dev[NR], mf_block_lda(IB+1, IB), blockP_dev, mf_block_lda(IB+1, IB+1), ZERO, tmp2_dev, NR, magma_queue);
      flops += 2.0 * NR * NP * NP;
      tgemm_dev('N', 'N', NR, NR, NP, ONE, tmp2_dev, NR, &blockR_dev[NR], mf_block_lda(IB+1, IB), ONE, tmp1_dev, NR, magma_queue);
      flops += 2.0 * NR * NR * NP;
      
      if (matrix_nd > 0)
      {
         tgemm_dev('T', 'N', NR, ND, ND, ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), blockDense_dev, mf_block_lda(NBlock-1, NBlock-1), ZERO, tmp2_dev, NR, magma_queue);
         flops += 2.0 * NR * ND * ND;
         tgemm_dev('N', 'N', NR, NR, ND, ONE, tmp2_dev, NR, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), ONE, tmp1_dev, NR, magma_queue);
         flops += 2.0 * NR * NR * ND;
      }

      tgemm_dev('T', 'N', NR, NR, NR, ONE, blockR_dev, mf_block_lda(IB, IB), tmp1_dev, NR, ZERO, tmp2_dev, NR, magma_queue);
      flops += 2.0 * NR * NR * NR; 
      tgemm_dev('N', 'N', NR, NR, NR, ONE, tmp2_dev, NR, blockR_dev, mf_block_lda(IB, IB), ZERO, tmp1_dev, NR, magma_queue);
      flops += 2.0 * NR * NR * NR; 
      tlacpy_dev('F', NR, NR, tmp1_dev, NR, blockR_dev, mf_block_lda(IB, IB), magma_queue);
   }

   copy_supernode_diag(blockR_dev, 0);

   return flops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::factorize()
{
    create_blocks();

    //Data allocation
    if (!MF_dev_allocated)
    {
       MF = new T[matrix_n_nonzeros];
       size_t max_supernode_nnz = matrix_nt > 1 ? matrix_ns*(2*matrix_ns+matrix_nd) : matrix_ns*(matrix_ns+matrix_nd);
       size_t dense_supernode_nnz = matrix_nd > 0 ? matrix_nd*matrix_nd : 0;
       allocate_data_on_device((void**)&blockR_dev,max_supernode_nnz*sizeof(T));
       allocate_data_on_device((void**)&blockM_dev,max_supernode_nnz*sizeof(T));
       allocate_data_on_device((void**)&blockP_dev,max_supernode_nnz*sizeof(T));
       allocate_data_on_device((void**)&blockDense_dev,dense_supernode_nnz*sizeof(T));
       MF_dev_allocated = true;
    }

    size_t nnz = matrix_ia[matrix_size];
    size_t max_rows = matrix_nt > 1 ? 2*matrix_ns+matrix_nd : matrix_ns+matrix_nd;
    size_t max_cols = max(matrix_ns, matrix_nd);

    //Temp data allocation
    allocate_data_on_device((void**)&ia_dev,(max_cols+1)*sizeof(size_t));
    allocate_data_on_device((void**)&ja_dev,max_rows*max_cols*sizeof(size_t));
    allocate_data_on_device((void**)&a_dev,max_rows*max_cols*sizeof(T));

    //Computation
    double flops = FirstStageFactor();

    //Temp data deallocation
    deallocate_data_on_dev(ia_dev,(max_cols+1)*sizeof(size_t));
    deallocate_data_on_dev(ja_dev,max_rows*max_cols*sizeof(size_t));
    deallocate_data_on_dev(a_dev,max_rows*max_cols*sizeof(T));

    factorization_completed = true;

    return flops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::solve(T *b, size_t nrhs)
{
    double flops = solve(b, b, nrhs);

    return flops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::solve(T *x, T *b, size_t nrhs)
{
    if (!factorization_completed)
       factorize();

    //Data allocation
    rhs = new T[matrix_size*nrhs];

    //Copy data
    memcpy(rhs,b,matrix_size*nrhs*sizeof(T));

    //Computation
    double flops = SecondStageSolve(nrhs);

    //Copy data
    memcpy(x,rhs,matrix_size*nrhs*sizeof(T));

    //Data deallocation
    delete[] rhs;

    return flops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::RGFdiag(T *diag)
{
    if (!factorization_completed)
       factorize();

    factorization_completed = false;

    //Data allocation
    T *tmp1_dev;
    T *tmp2_dev;
    allocate_data_on_device((void**)&diag_dev,matrix_size*sizeof(T));
    allocate_data_on_device((void**)&diag_pos_dev,matrix_size*sizeof(size_t));
    allocate_data_on_device((void**)&tmp1_dev,b_size*b_size*sizeof(T));
    allocate_data_on_device((void**)&tmp2_dev,b_size*b_size*sizeof(T));

    //Copy diag_pos to device
    memcpy_to_device(diag_pos,diag_pos_dev,matrix_size*sizeof(size_t));

    //Computation
    double flops = ThirdStageRGF(tmp1_dev,tmp2_dev);

    //Copy data to host
    memcpy_to_host(diag,diag_dev,matrix_size*sizeof(T));

    //Data deallocation
    deallocate_data_on_dev(diag_dev,matrix_size*sizeof(T));
    deallocate_data_on_dev(diag_pos_dev,matrix_size*sizeof(size_t));
    deallocate_data_on_dev(tmp1_dev,b_size*b_size*sizeof(T));
    deallocate_data_on_dev(tmp2_dev,b_size*b_size*sizeof(T));

    return flops;
}

/************************************************************************************************/

template <class T>
T RGF<T>::logDet()
{
    if (!factorization_completed)
       factorize();

    //Computation
    T det = T(0.0);
    indexed_log_sum(MF, diag_pos, matrix_size, &det);

    return 2.0*det;
}

/************************************************************************************************/

template <class T>
double RGF<T>::residualNorm(T *x, T *b)
{
   T *r = new T[matrix_size];

   memcpy(r, b, matrix_size*sizeof(T));

   for (size_t ic = 0; ic < matrix_size; ic++)
   {
      for (size_t i = matrix_ia[ic]; i < matrix_ia[ic+1]; i++)
      {
         size_t ir = matrix_ja[i];

         r[ir] -= matrix_a[i]*x[ic];
         if (ir != ic)
            r[ic] -= matrix_a[i]*x[ir];
      }
   }

   double res = c_dtnrm2(matrix_size, r, 1);

   delete[] r;

   return res;
}

/************************************************************************************************/

template <class T>
double RGF<T>::residualNormNormalized(T *x, T *b)
{
   return residualNorm(x, b) / c_dtnrm2(matrix_size, b, 1);
}

/************************************************************************************************/

template <class T>
void RGF<T>::create_blocks()
{
    size_t IB;

    b_size = 0;

    for(IB=0;IB<NBlock;IB++){

        if(Bmax[IB]-Bmin[IB]>b_size){
	    b_size = Bmax[IB]-Bmin[IB];
	}
    }
}

/************************************************************************************************/

template <class T>
inline void RGF<T>::init_supernode(T *M_dev, size_t supernode)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t supernode_nnz = matrix_ia[supernode_lc] - matrix_ia[supernode_fc];
   size_t supernode_offset = matrix_ia[supernode_fc];
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;
   size_t supernode_size = rows * cols;

   fill_dev(M_dev, T(0.0), supernode_size);
   memcpy_to_device(&matrix_ia[supernode_fc],ia_dev,(cols+1)*sizeof(size_t));
   memcpy_to_device(&matrix_ja[supernode_offset],ja_dev,supernode_nnz*sizeof(size_t));
   memcpy_to_device(&matrix_a[supernode_offset],a_dev,supernode_nnz*sizeof(T));
   init_supernode_dev(M_dev, ia_dev, ja_dev, a_dev, supernode, supernode_nnz, supernode_offset, matrix_ns, matrix_nt, matrix_nd);
}

/************************************************************************************************/

template <class T>
inline void RGF<T>::copy_supernode_to_host(T *M_dev, size_t supernode)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t ind = mf_block_index(supernode, supernode);
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;

   memcpy_to_host(&MF[ind], M_dev, rows*cols*sizeof(T));
}

/************************************************************************************************/

template <class T>
inline void RGF<T>::copy_supernode_to_device(T *M_dev, size_t supernode)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t ind = mf_block_index(supernode, supernode);
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;

   memcpy_to_device(&MF[ind], M_dev, rows*cols*sizeof(T));
}

/************************************************************************************************/

template <class T>
inline void RGF<T>::copy_supernode_diag(T *src, size_t supernode)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t offset = mf_block_index(supernode, supernode);
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;

   indexed_copy_offset_dev(src, &diag_dev[supernode_fc], &diag_pos_dev[supernode_fc], cols, offset);
}

/************************************************************************************************/

template <class T>
inline void RGF<T>::swap_pointers(T **ptr1, T **ptr2)
{
   T *tmp = *ptr1;
   *ptr1 = *ptr2;
   *ptr2 = tmp;
}

/************************************************************************************************/

template <>
CPX RGF<CPX>::f_one()
{
    return CPX(1.0, 0.0);
}

template <>
double RGF<double>::f_one()
{
    return 1.0;
}

/************************************************************************************************/

template <>
CPX RGF<CPX>::f_zero()
{
    return CPX(0.0, 0.0);
}

template <>
double RGF<double>::f_zero()
{
    return 0.0;
}

/************************************************************************************************/

#endif
