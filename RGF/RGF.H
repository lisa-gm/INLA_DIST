#ifndef __RGF
#define __RGF

#include <string.h>
#include <omp.h>

#include "cublas_v2.h"
#include "Types.H"
#include "Utilities.H"
#include "CWC_utility.H"
#include "Blas.H"  // to inherit #define CUDA_POTRF


// debug ...
#include <iostream>
#include <fstream>
#include <ctime>

//#include "helper_functions.h"

#include "nvToolsExt.h"


//#define PRINT_MSG

template <class T>
class RGF{
	
public:

    RGF(size_t,size_t,size_t);

    ~RGF();
    double factorize(size_t* ia, size_t* ja, T* a);
    double factorize_noCopyHost(size_t* ia, size_t* ja, T* a, double &logDet);
    double solve(size_t*, size_t*, T*, T*,size_t);
    double solve(size_t*, size_t*, T*, T*,T*,size_t);
    double RGFdiag(size_t*, size_t*, T*, T*);
    double RGFinvBlks(size_t*, size_t*, T*, T*);
    double RGFselInv(size_t *ia, size_t *ja, T *a, T *invQ);
    T logDet(size_t*, size_t*, T*);
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

    size_t max_supernode_nnz = 0;

    size_t ind_invBlks_fi;
    size_t mem_alloc_dev = 0;   // keep track of allocated memory on GPU

    //size_t b_size;
    bool MF_allocated = false;
    bool invBlks_allocated = false;
    bool MF_dev_allocated = false;
    bool factorization_completed = false;
    int cpy_indicator; // 0: copy back diagonal, 1: copy back nnzQ, 2: copy back invBlks

    // add copy stream, add compute stream
    magma_queue_t magma_queue_1;
    magma_queue_t magma_queue_2;
    cudaStream_t stream_c;
    cudaStream_t copyStream = NULL;
    cudaStream_t copyStreamDtH = NULL;
    cudaStream_t magma_cudaStream_1 = NULL;
    cudaStream_t magma_cudaStream_2 = NULL;

    cudaEvent_t  initBlock_dev_ev;
    cudaEvent_t  potrf_dev_ev;

    magma_event_t potrf_dev_magma_ev;

    void *cublas_handle;

#ifdef CUDA_POTRF
    int* info_cuda = NULL;
    int *cuda_buffer_flag_potrf;
    int *cuda_buffer_flag_trtri;

    cusolverDnHandle_t *handle;
    cusolverDnParams_t *params;

    size_t *dev_size;
    size_t *host_size;
    double *mem_cuda_dev;
    double *mem_cuda_host;
#endif

    T *MF;
    T *invBlks;
    T *inv_a;
    T *blockR_dev;
    T *blockM_dev;
    T *blockP_dev;
    T *blockDense_dev;
    T *rhs;

    size_t *ia_dev;
    size_t *ja_dev;
    T      *a_dev;

    size_t *inv_ia_dev;
    size_t *inv_ja_dev;
    T      *inv_a_dev;

    T      *diag_dev;
    size_t *diag_pos_dev;

    inline size_t mf_block_index(size_t, size_t);
    inline size_t mf_block_lda(size_t, size_t);
    inline size_t mf_dense_block_index(size_t);
    inline size_t mf_dense_block_offset(size_t);
    inline size_t mf_dense_block_lda(size_t);

    inline size_t invblks_diag_block_index(size_t);
    inline size_t invblks_dense_block_index(size_t);

    double FirstStageFactor();
    double FirstStageFactor_noCopyHost(double &logDet);
    double FirstStageFactor_noCopyHost_testV(double &logDet);
    double SecondStageSolve(size_t);
    double ThirdStageRGF(T*,T*, int);
    //void create_blocks();
    void initialize_MF_host();
    void initialize_invBlks_host();

    // new: compute maximum number of nonzeros over all supernodes
    inline void get_max_supernode_nnz(); 
    
    inline void init_supernode(T *M_dev, size_t supernode, cudaStream_t stream);
    //inline void init_supernode(T *M_dev, size_t supernode);

    inline void copy_supernode_to_host(T *M_dev, size_t supernode, cudaStream_t stream);
    //inline void copy_supernode_to_host(T *M_dev, size_t supernode);

    inline void extract_nnzA(T *M_dev, size_t supernode); // new to extract nnzQ(Qinv)
    inline void copy_supernode_to_host_write(T *M_dev, size_t supernode); // debugging ...

    inline void copy_supernode_to_device(T *M_dev, size_t supernode, cudaStream_t stream);
    //inline void copy_supernode_diag(T *src, size_t supernode, cudaStream_t stream);
    inline void copy_supernode_diag(T *src, size_t supernode);
    inline void swap_pointers(T **ptr1, T **ptr2);

    inline T f_one();
    inline T f_zero();
};

/************************************************************************************************/

template <class T>
RGF<T>::RGF(size_t ns, size_t nt, size_t nd)
{
   // only set up indices etc. no values yet.

#ifdef CUDA_POTRF
   printf("using CUDA POTRF.\n");
#else
   printf("using MAGMA POTRF.\n");
#endif
   
    //matrix_ia = ia;
    //matrix_ja = ja;
    //matrix_a = a;
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

   /*
   printf("diag_pos: ");
    for(int i=0; i<matrix_size; i++){
      printf("%ld ", diag_pos[i]);
    }
   printf("\n");
   */

    gpuErrchk(cudaEventCreate(&initBlock_dev_ev));
    gpuErrchk(cudaEventCreate(&potrf_dev_ev));

    magma_event_create(&potrf_dev_magma_ev);	


    // initialize cuda copyStream
    gpuErrchk(cudaStreamCreate( &copyStream ));
    gpuErrchk(cudaStreamCreate( &copyStreamDtH ));

    magma_init();
    magma_device_t device;
    magma_getdevice(&device);

    // in order to get the magma GEMMs in a non-null stream, the second argument of magma_queue_create needs to be a CUDA stream
    // so make a cuda stream to use for this.
    gpuErrchk(cudaStreamCreate ( &magma_cudaStream_1 ));
    gpuErrchk(cudaStreamCreate ( &magma_cudaStream_2 ));

    //magma_queue_create_from_cuda(device, NULL, NULL, NULL, &magma_queue);
    magma_queue_create_from_cuda(device, magma_cudaStream_1, NULL, NULL, &magma_queue_1);
    magma_queue_create_from_cuda(device, magma_cudaStream_2, NULL, NULL, &magma_queue_2);

    // magma_queue created from specific streams
    //stream_c = magma_queue_get_cuda_stream(magma_queue_1);
    //cublas_handle = magma_queue_get_cublas_handle(magma_queue_1);

#ifdef CUDA_POTRF
    cudaMalloc((void**)&info_cuda, sizeof(int));

    cudaMallocHost((void**)&cuda_buffer_flag_potrf, sizeof(int));
    cuda_buffer_flag_potrf[0] = 0;

    cudaMallocHost((void**)&cuda_buffer_flag_trtri, sizeof(int));
    cuda_buffer_flag_trtri[0] = 0;

    cudaMallocHost((void**)&handle,sizeof(cusolverDnHandle_t));
    cudaMallocHost((void**)&params,sizeof(cusolverDnParams_t));

    cudaMallocHost((void**)&dev_size,sizeof(size_t));
    cudaMallocHost((void**)&host_size,sizeof(size_t));

    cusolverStatus_t cuSolverError = cusolverDnCreate(handle);
    if(cuSolverError != 0){
      printf("cuSolverError not Zero in create handle! Error: %d\n", cuSolverError);
      exit(1);
    }

    cuSolverError = cusolverDnCreateParams(params);
    if(cuSolverError != 0){
      printf("cuSolverError not Zero in create params!\n");
      exit(1);
    }

#endif 

}

/************************************************************************************************/

template <class T>
RGF<T>::~RGF()
{
#ifdef PRINT_MSG
   std::cout << "In RGF destructor. MF_allocated : " << MF_allocated << std::endl;
#endif 

   // in case that only factorize_noCopyHost is called MF not allocated ...
   if (MF_allocated)
   {
      //delete[] MF;
      // if memory PINNED
#ifdef PRINT_MSG
      std::cout << "cudaFreeHost(MF) gets called." << std::endl;
#endif
      cudaFreeHost(MF);
   }

   // if we require entire diagonal blocks
   if(invBlks_allocated){
#ifdef PRINT_MSG
      std::cout << "cudaFreeHost(invBlks) gets called." << std::endl;
#endif
      cudaFreeHost(invBlks);
   }

   if (MF_dev_allocated)
   {
      size_t max_supernode_nnz_dense = matrix_nt > 1 ? matrix_ns*(2*matrix_ns+matrix_nd) : matrix_ns*(matrix_ns+matrix_nd);
      size_t final_supernode_nnz_dense = matrix_nd > 0 ? matrix_nd*matrix_nd : 0;
      deallocate_data_on_dev(blockR_dev,max_supernode_nnz_dense*sizeof(T));
      deallocate_data_on_dev(blockM_dev,max_supernode_nnz_dense*sizeof(T));
      deallocate_data_on_dev(blockP_dev,max_supernode_nnz_dense*sizeof(T));
      deallocate_data_on_dev(blockDense_dev,final_supernode_nnz_dense*sizeof(T));
   }

   magma_queue_destroy(magma_queue_1);
   magma_queue_destroy(magma_queue_2);
   magma_finalize();

   cudaStreamDestroy(copyStream);
   cudaStreamDestroy(copyStreamDtH);
   cudaStreamDestroy(magma_cudaStream_1);
   cudaStreamDestroy(magma_cudaStream_2);

   delete[] Bmin;
   delete[] Bmax;
   delete[] diag_pos;


#ifdef CUDA_POTRF
    cudaFreeHost(info_cuda);
    cudaFreeHost(cuda_buffer_flag_potrf);
    cudaFreeHost(cuda_buffer_flag_trtri);

    cudaFreeHost(host_size);
    cudaFreeHost(dev_size);
    cudaFreeHost(mem_cuda_host);
    cudaFree(mem_cuda_dev);
#endif 

}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_block_index(size_t r, size_t c)
{
   //printf("r = %ld, c = %ld, c*matrix_ns = %ld, (r-c)*matrix_ns = %ld, diag_pos[c*matrix_ns]+(r-c)*matrix_ns = %ld\n", r, c, c*matrix_ns, (r-c)*matrix_ns, diag_pos[c*matrix_ns]+(r-c)*matrix_ns);
   return diag_pos[c*matrix_ns] + (r-c)*matrix_ns;
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
      return diag_pos[i*matrix_ns] + 2*matrix_ns;
   else if (i == matrix_nt-1)
      return diag_pos[i*matrix_ns] + matrix_ns;
   else
      return diag_pos[i*matrix_ns];
}

/************************************************************************************************/

template <class T>
inline size_t RGF<T>::mf_dense_block_offset(size_t i)
{
   if (i < matrix_nt-1)
      return 2*matrix_ns;
   else if (i == matrix_nt-1)
      return matrix_ns;
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
// assuming that invBlks stores :
// Sigma_11, Sigma_n+11, Sigma22, Sigma_n+12, ... , Sigma_nn, Sigma_n+1n, Sigma_n+1n+1
// total size (ns+nd)*ns*nt + nb^2

template <class T>
inline size_t RGF<T>::invblks_diag_block_index(size_t i)
{
   // out-of-bounds check for i which represents time step
   if(i > matrix_nt){
      printf("in invblks_diag_block_index(). block index i = %ld out of bounds!\n", i);
   }
   // (ns+nd)*ns*i
   return (matrix_ns+matrix_nd)*matrix_ns*i;

}


template <class T>
inline size_t RGF<T>::invblks_dense_block_index(size_t i)
{
   // out-of-bounds check for i which represents time step
   if(i > matrix_nt){
      printf("in invblks_diag_block_index(). block index i = %ld out of bounds!\n", i);
   }
   // ns^2*(i+1) + ns*nb*i
   return matrix_ns*matrix_ns*(i+1) + matrix_ns*matrix_nd*i;
}

/************************************************************************************************/


template <class T>
double RGF<T>::FirstStageFactor()
{

#ifdef PRINT_MSG
    std::cout << "In FirstStageFactor(), omp get thread num = " << omp_get_thread_num() << std::endl;
#endif
   
    double t_potrf    = 0;
    double t_dgemm    = 0;
    double t_copy_DtH = 0;
    double t_temp;

    double t_init_supernode = get_time(0.0);

    int info;
    size_t IB;
    size_t NR,NM;
    T ONE      = f_one();

    // count floating point operations
    double flop_count = 0;

//tpotrf_dev('L', NR, M1_dev ,NR ,&info)
//rj dpotrf MF[0,0]
//rj dtrsm RLTN MF[0,0] MF[1,0]

    IB = 0;
    NR = Bmax[0]-Bmin[0];

#ifdef PRINT_MSG
       	std::cout << "IB = " << IB << std::endl;
#endif

    init_supernode(blockR_dev, IB, magma_cudaStream_1);
    if (matrix_nd > 0)
    {
       init_supernode(blockDense_dev, matrix_nt, magma_cudaStream_1);
    }

    // put it in cudaEvent
    //gpuErrchk(cudaEventRecord(copy_gpu, magma_cudaStream_1)); // want compute_stream to wait
    //gpuErrchk(cudaStreamWaitEvent(compute_stream, copy_gpu, 0));

#ifdef PRINT_MSG
    std::cout << "calling first Cholesky factorization now. NR = " << NR << ", mf_block_lda(IB, IB) = " << mf_block_lda(IB, IB) << std::endl;
#endif

    t_init_supernode = get_time(t_init_supernode);
    //printf("time init supernode in factorize: %f\n", t_init_supernode);

    // start timer for counting flops
    double t_fact = get_time(0.0);
	
    t_temp = get_time(0.0);
    // CHOLESKY FACTORISATION FIRST DIAGONAL BLOCK
#ifdef CUDA_POTRF
     // factorization first block -> set buffer flag to zero!
     cuda_buffer_flag_potrf[0] = 0;
     tpotrf_dev_cuda('L', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
   // magma_queue currently not supported
    cudaDeviceSynchronize(); 
    tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
    cudaDeviceSynchronize();
#endif
    //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(0, 0), mf_block_lda(0, 0));
    t_temp = get_time(t_temp);
    t_potrf += t_temp;
    //printf("IB = %d, t_potrf = %f\n", IB, t_temp);
    //flop_count += 1.0/3.0 * NR * NR * NR;
    flop_count += 1.0/3.0 * NR * NR * NR + 0.5 * NR * NR + 1.0/6.0 * NR; 

   // the trsm shouldn't be launched until potrf has finished 
   //cudaDeviceSynchronize();
   gpuErrchk(cudaEventRecord(potrf_dev_ev, magma_cudaStream_1)); 
   gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, potrf_dev_ev, 0));   

    // TODO: if nt = 0 but nd != 0 => problem !! 
    //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
    if (matrix_nt > 1)
    {
       t_temp = get_time(0.0);	    
       // UPDATE COLUMNS ACCORDING TO CHOLESKY FACTORISATION
       // separate the two operations
       //ttrsm_dev('R', 'L', 'T', 'N', NR+matrix_nd, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[NR], mf_block_lda(1, 0), magma_queue_1);
       ttrsm_dev('R', 'L', 'T', 'N', NR, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[NR], mf_block_lda(1, 0), magma_queue_1);

       t_temp = get_time(t_temp);
       t_dgemm += t_temp;
       flop_count += NR * NR * NR;
    }

    if (matrix_nd > 0){
      //printf("mf_dense_block_offset(IB) : %ld\n", mf_dense_block_offset(IB));
      //printf("mf_dense_block_lda(IB) : %ld\n", mf_dense_block_lda(IB));
       ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
       flop_count += matrix_nd * NR * NR;
    }

    t_temp = get_time(0.0);
    cudaDeviceSynchronize();
    // if copy supernode device to host happens while new supernode is being initialized ... memory issues?
    copy_supernode_to_host(blockR_dev, IB, copyStreamDtH);
    //copy_supernode_to_host_write(blockR_dev, IB);

    t_temp = get_time(t_temp);
    t_copy_DtH =+ t_temp;

#ifdef PRINT_MSG
	std::cout << "entering loop block factorization loop now" << std::endl;	
#endif  

    //rj IB = 1; IB < NBlock; IB++
    for (IB = 1; IB < matrix_nt-1; IB++)
    {

#ifdef PRINT_MSG
       	std::cout << "IB = " << IB << std::endl;
#endif

       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       double flop_1iter = 0.0;

       // 
       swap_pointers(&blockR_dev, &blockM_dev);

       //nvtxRangeId_t id_initSN = nvtxRangeStartA("initSuperNode_inLoop");
       init_supernode(blockR_dev, IB, magma_cudaStream_1);
       //init_supernode(blockR_dev, IB);
       //nvtxRangeEnd(id_initSN);

       // UPDATE NEXT DIAGONAL BLOCK 
       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       // todo rj: 3-last parameter ZERO in PARDISO
       // magma_queue_1 is on top of magma_cudaStream_1

       // add cuda_event such that magma_queue_2/magma_cudaStream_2 wait for magma_cudaStream_1 to get here
       // maybe better to use magma version?
       gpuErrchk(cudaEventRecord(initBlock_dev_ev, magma_cudaStream_1)); 
       gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, initBlock_dev_ev, 0));	

       nvtxRangeId_t id_dgemm = nvtxRangeStartA("BigGemm_inLoop");
       t_temp = get_time(0.0);
       tgemm_dev('N', 'T', NR, NR, NM, -ONE, &blockM_dev[NM], mf_block_lda(IB, IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);
       t_temp = get_time(t_temp);
       nvtxRangeEnd(id_dgemm);
       t_dgemm += t_temp;
       
       // dim dgemm: m = , n = , k = 
       flop_count += 2.0 * NR * NR * NM;

       // direct into separate stream -> becomes relevant when many fixed effects ...
       if (matrix_nd > 0)
       {
          // Update dense rows of next super node IB
          id_dgemm = nvtxRangeStartA("DenseLowerGemm_inLoop");
	       t_temp = get_time(0.0);
	       tgemm_dev('N', 'T', matrix_nd, NR, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
          t_dgemm += get_time(t_temp);
          nvtxRangeEnd(id_dgemm);
	       // NM = NR
          flop_count += 2.0*matrix_nd * NR * NM;
          
          // update last diagonal block
          id_dgemm = nvtxRangeStartA("LastBlockLowerGemm_inLoop");
          tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue_2);
          nvtxRangeEnd(id_dgemm);
          flop_count +=  2.0* matrix_nd *matrix_nd *NM;

       }

       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       t_temp = get_time(0.0);
       nvtxRangeId_t id_dpotrf = nvtxRangeStartA("Potrf_inLoop");       

#ifdef CUDA_POTRF
       if(NR != NM){
         //printf("Different block size! Re-initialize cudaBufferSize. NR = %ld, NM = %ld\n", NR, NM);
         cuda_buffer_flag_potrf[0] = 0;
       }
       tpotrf_dev_cuda('L', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
      // how can I get the default stream to wait but make an exception for copyStreamDtH ?
       cudaDeviceSynchronize(); // TO BE REPLACED
       tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
       cudaDeviceSynchronize();
#endif

       nvtxRangeEnd(id_dpotrf);
       t_temp  = get_time(t_temp);
       t_potrf += t_temp;
       //printf("IB = %d, t_potrf = %f\n", IB, t_temp);
       flop_count += 1.0/3.0 * NR * NR * NR + 0.5*NR*NR + 1.0/6.0 * NR;

       //gpuErrchk(cudaEventRecord(potrf_dev_ev, magma_cudaStream_1)); 
       //gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, potrf_dev_ev, 0));   

       magma_event_record(potrf_dev_magma_ev, magma_queue_1);	      
       magma_queue_wait_event(magma_queue_2, potrf_dev_magma_ev);	
       
       // triangular solve, all columns -> potentially split this ... when nd large ... two separate streams.
       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
       nvtxRangeId_t id_ttrsm = nvtxRangeStartA("ttrsm_inLoop");       
       //ttrsm_dev('R', 'L', 'T', 'N', NR+matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[NR], mf_block_lda(IB+1, IB), magma_queue_1);
       ttrsm_dev('R', 'L', 'T', 'N', NR, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[NR], mf_block_lda(IB+1, IB), magma_queue_1);

       if(matrix_nd > 0){
         ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
       }

       nvtxRangeEnd(id_ttrsm);

       flop_count += (NR + matrix_nd) * NR * NR;
       //printf("RJ: ttrsm NR: %d, a: %d, lda: %d, b: %d, ldb: %d\n", NR, mf_block_index(IB-1, IB-1), mf_block_lda(IB-1, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1));

       t_temp = get_time(0.0);

       //nvtxRangeId_t id_Cpy_SN_host = nvtxRangeStartA("copy_SN_toHost_inLoop");     
       cudaDeviceSynchronize();
       copy_supernode_to_host(blockR_dev, IB, copyStreamDtH);  
       //copy_supernode_to_host(blockR_dev, IB);
       //nvtxRangeEnd(id_Cpy_SN_host);             
       t_copy_DtH += get_time(t_temp);
   
    }  // end for loop IB

    if (matrix_nt > 1)
    {
       IB = matrix_nt-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

#ifdef PRINT_MSG
       	std::cout << "IB = " << IB << std::endl;
#endif

       swap_pointers(&blockR_dev, &blockM_dev);
       nvtxRangeId_t id_initSN = nvtxRangeStartA("initLastSuperNode");
       init_supernode(blockR_dev, IB, magma_cudaStream_1);
       //init_supernode(blockR_dev, IB);
       nvtxRangeEnd(id_initSN);

       gpuErrchk(cudaEventRecord(initBlock_dev_ev, magma_cudaStream_1)); 
       gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, initBlock_dev_ev, 0));	

       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       // todo rj: 3-last parameter ZERO in PARDISO
       nvtxRangeId_t id_dgemm = nvtxRangeStartA("LastLargeGemm");
       tgemm_dev('N', 'T', NR, NR, NM, -ONE, &blockM_dev[NM], mf_block_lda(IB, IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);
       nvtxRangeEnd(id_dgemm);             
       flop_count += 2.0* NR * NR * NM;

       if (matrix_nd > 0)
       {
          tgemm_dev('N', 'T', matrix_nd, NR, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
          flop_count += 2.0*matrix_nd * NR * NM;

          tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue_2);
          flop_count += 2.0*matrix_nd *matrix_nd *NM; 
       }

       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
        nvtxRangeId_t id_dpotrf = nvtxRangeStartA("LastPotrf");       
#ifdef CUDA_POTRF
       if(NR != NM){
         //printf("Different block size! Re-initialize cudaBufferSize. NR = %ld, NM = %ld\n", NR, NM);
         cuda_buffer_flag_potrf[0] = 0;
       }
       tpotrf_dev_cuda('L', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
       cudaDeviceSynchronize(); // TO BE REPLACED
       tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
       cudaDeviceSynchronize();
#endif
       nvtxRangeEnd(id_dpotrf);
       flop_count += 1.0/3.0 * NR * NR * NR + 0.5*NR*NR + 1.0/6.0 * NR;

       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
       if (matrix_nd > 0)
       {
          nvtxRangeId_t id_ttrsm = nvtxRangeStartA("LastTtrsm");       
          ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_1);
          nvtxRangeEnd(id_ttrsm);
          flop_count += matrix_nd * NR * NR;
       
       }
       //printf("RJ: ttrsm NR: %d, a: %d, lda: %d, b: %d, ldb: %d\n", NR, mf_block_index(IB-1, IB-1), mf_block_lda(IB-1, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1));
	    t_temp = get_time(0.0);

       cudaDeviceSynchronize();
       nvtxRangeId_t id_Cpy_SN_host = nvtxRangeStartA("copy_LastLarge_SN_toHost");  
       copy_supernode_to_host(blockR_dev, IB, copyStreamDtH);     
       //copy_supernode_to_host_write(blockR_dev, IB);
       //copy_supernode_to_host(blockR_dev, IB);
       nvtxRangeEnd(id_Cpy_SN_host);

       t_copy_DtH += get_time(t_temp);

    }

    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

#ifdef PRINT_MSG
       	std::cout << "IB = " << IB << std::endl;
#endif

       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       //rj NM is the same for all blocks 0..nt-1
       tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockR_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockR_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue_1);
       flop_count += 2.0 * matrix_nd *matrix_nd *NM; 
   
       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       //tpotrf_dev('L', NR, blockDense_dev, mf_block_lda(IB, IB), &info);
#ifdef CUDA_POTRF
       if(NR != NM){
         //printf("Different block size! Re-initialize cudaBufferSize. NR = %ld, NM = %ld\n", NR, NM);
         cuda_buffer_flag_potrf[0] = 0;
       }
       tpotrf_dev_cuda('L', NR, blockDense_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
       tpotrf_dev('L', NR, blockDense_dev, mf_block_lda(IB, IB), &info);
#endif

       flop_count += 1.0/3.0 * NR * NR * NR + 0.5*NR*NR + 1.0/6.0 * NR;
       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));

#ifdef PRINT_MSG
       std::cout << "after complete Cholesky factorization" << std::endl;
#endif
       t_temp = get_time(0.0);

       cudaDeviceSynchronize();
       copy_supernode_to_host(blockDense_dev, IB, copyStreamDtH);
       //copy_supernode_to_host_write(blockDense_dev, IB);
       
       t_copy_DtH += get_time(t_temp);
    
    }

   cudaDeviceSynchronize(); // TO BE REPLACED

   t_fact = get_time(t_fact);
   //nt    : " << flop_count / 1e9 << std::endl;
   //std::cout << "time factorize : " << t_fact << ", sum time copy DtH : " << t_copy_DtH << ", sum t potrf : " << t_potrf << ", sum t dgemm : " << t_dgemm << std::endl;
   double gflops = flop_count / (1e9*t_fact);
   //printf("GFLOPS factorize : %f\n", gflops);

   //std::cout << "time factorization Qxy : " << t_fact << std::endl;
   //std::cout << "gflop count Qxy        : " << flop_count / 1e9 << std::endl;
   return gflops;
}

/************************************************************************************************/
// dummy function to check calls

template <class T>
double RGF<T>::FirstStageFactor_noCopyHost_testV(double &logDet){

    printf("In Factorize noCopyHost test version.\n");

    int info;
    size_t IB;
    size_t NR,NM;
    T ONE      = f_one();

    IB = 0;
    NR = Bmax[0]-Bmin[0];

    nvtxRangeId_t id_initSN = nvtxRangeStartA("initSuperNodeDev");  
    init_supernode(blockR_dev, IB);
    nvtxRangeEnd(id_initSN);    

    T* diag;
    diag = new T[NR];

    // loop where I copy from blockR_dev to blockM_dev & keep factorizing

    for(int i=0; i<100; i++){
#ifdef PRINT_MSG
       std::cout << "Copy to blockM_dev, i = " << i << ",  mf_block_lda(IB, IB) = " << mf_block_lda(IB, IB) << std::endl;
#endif
       nvtxRangeId_t id_cpyDevDev = nvtxRangeStartA("CopySNtoblockMdev");  
       tlacpy_dev('N', 2*NR, NR, blockR_dev, mf_block_lda(IB, IB), blockM_dev,  mf_block_lda(IB, IB), magma_queue_1);
       nvtxRangeEnd(id_cpyDevDev);    

#ifdef PRINT_MSG
       std::cout << "IB = " << IB << std::endl;
#endif

       double t_fact = get_time(0.0);

       // CHOLESKY FACTORISATION FIRST DIAGONAL BLOCK
        nvtxRangeId_t id_dpotrf = nvtxRangeStartA("firstPotrf");  
#ifdef CUDA_POTRF
       tpotrf_dev_cuda('L', NR, blockM_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
       cudaDeviceSynchronize();     
       tpotrf_dev('L', NR, blockM_dev, mf_block_lda(IB, IB), &info);
#endif
       nvtxRangeEnd(id_dpotrf);    

       //copy_supernode_to_host_write(blockR_dev, IB);

       id_cpyDevDev = nvtxRangeStartA("CopyDiagDevDev");  
       copy_supernode_diag(blockM_dev, IB);
       nvtxRangeEnd(id_cpyDevDev);  

      id_cpyDevDev = nvtxRangeStartA("CopyDiagtoHost");  
      cudaMemcpy(diag, diag_dev, NR*sizeof(T), cudaMemcpyDeviceToHost );
      nvtxRangeEnd(id_cpyDevDev);    

      id_cpyDevDev = nvtxRangeStartA("ComLogDetHost");       
      // compute log sum
      log_sum(diag, NR, &logDet);
      logDet = 2*logDet;

      printf("log Det : %f\n", logDet);
      nvtxRangeEnd(id_cpyDevDev);  
    }

    delete[] diag;

    return 1.0;

}




/************************************************************************************************/

template <class T>
double RGF<T>::FirstStageFactor_noCopyHost(double &logDet)
{

   //std::cout << "In FirstStageFactor_noCopyHost(), omp get thread num = " << omp_get_thread_num() << std::endl;

    int info;
    size_t IB;
    size_t NR,NM;
    T ONE      = f_one();

    // count floating point operations
    double flop_count = 0;

    double t_temp;
    double t_dgemm;

//tpotrf_dev('L', NR, M1_dev ,NR ,&info)
//rj dpotrf MF[0,0]
//rj dtrsm RLTN MF[0,0] MF[1,0]

    IB = 0;
    NR = Bmax[0]-Bmin[0];

    init_supernode(blockR_dev, IB, magma_cudaStream_1);
    if (matrix_nd > 0)
    {
       init_supernode(blockDense_dev, matrix_nt, magma_cudaStream_1);
    }

#ifdef PRINT_MSG
    std::cout << "calling first Cholesky factorization now" << std::endl;
    std::cout << "IB = " << IB << std::endl;
#endif

    double t_fact = get_time(0.0);

    // CHOLESKY FACTORISATION FIRST DIAGONAL BLOCK
     nvtxRangeId_t id_dpotrf = nvtxRangeStartA("firstPotrf");       
    //tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);

#ifdef CUDA_POTRF
     // factorization first block -> set buffer flag to zero!
     cuda_buffer_flag_potrf[0] = 0;
     tpotrf_dev_cuda('L', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
    cudaDeviceSynchronize();
    tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
    cudaDeviceSynchronize();

#endif

    nvtxRangeEnd(id_dpotrf);    
    //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(0, 0), mf_block_lda(0, 0));
    flop_count += 1.0/3.0 * NR * NR * NR + 0.5*NR*NR + 1.0/6.0 * NR;

    gpuErrchk(cudaEventRecord(potrf_dev_ev, magma_cudaStream_1)); 
    gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, potrf_dev_ev, 0));   

    // TODO: if nt = 0 but nd != 0 => problem !! 
    //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
    if (matrix_nt > 1)
    {
       t_temp = get_time(0.0);	    
       // UPDATE COLUMNS ACCORDING TO CHOLESKY FACTORISATION
       // separate the two operations
       //ttrsm_dev('R', 'L', 'T', 'N', NR+matrix_nd, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[NR], mf_block_lda(1, 0), magma_queue_1);
       ttrsm_dev('R', 'L', 'T', 'N', NR, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[NR], mf_block_lda(1, 0), magma_queue_1);

       t_temp = get_time(t_temp);
       t_dgemm += t_temp;
       flop_count += NR * NR * NR;
    }

    if (matrix_nd > 0){
      //printf("mf_dense_block_offset(IB) : %ld\n", mf_dense_block_offset(IB));
      //printf("mf_dense_block_lda(IB) : %ld\n", mf_dense_block_lda(IB));
       ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(0, 0), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
       flop_count += matrix_nd * NR * NR;
    }

    cudaDeviceSynchronize();
    // replace all copy supernode to host by computing sum of current diagonal block
    //copy_supernode_to_host(blockR_dev, IB);
    copy_supernode_diag(blockR_dev, IB);

#ifdef PRINT_MSG
   std::cout << "entering loop block factorization loop now" << std::endl; 
#endif  

    //rj IB = 1; IB < NBlock; IB++
    for (IB = 1; IB < matrix_nt-1; IB++)
    {

#ifdef PRINT_MSG
         std::cout << "IB = " << IB << std::endl;
#endif

       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       swap_pointers(&blockR_dev, &blockM_dev);
       init_supernode(blockR_dev, IB, magma_cudaStream_1);

       //cudaDeviceSynchronize();
       gpuErrchk(cudaEventRecord(initBlock_dev_ev, magma_cudaStream_1)); 
       gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, initBlock_dev_ev, 0));	

       // UPDATE NEXT DIAGONAL BLOCK 
       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       // todo rj: 3-last parameter ZERO in PARDISO
       nvtxRangeId_t id_dgemm = nvtxRangeStartA("BigGemm_inLoop");
       tgemm_dev('N', 'T', NR, NR, NM, -ONE, &blockM_dev[NM], mf_block_lda(IB, IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);
       nvtxRangeEnd(id_dgemm);
       flop_count += 2.0 * NR * NR * NR;

       // these dgemms independent from previous one. Put in separate stream, just require supernode to be loaded.
       if (matrix_nd > 0)
       {
          // Update dense rows of super node IB
          tgemm_dev('N', 'T', matrix_nd, NR, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
          // NM = NR
          flop_count += 2.0*matrix_nd * NR * NM; //  matrix_nd *NR *(NM+2) + matrix_nd *NR *NM

          // update last diagonal block
          tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue_2);
          flop_count += 2.0 * matrix_nd * matrix_nd * NM;

       }

       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       nvtxRangeId_t id_dpotrf = nvtxRangeStartA("Potrf_inLoopTest");       
       //tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
#ifdef CUDA_POTRF
       if(NR != NM){
         //printf("Different block size! Re-initialize cudaBufferSize. NR = %ld, NM = %ld\n", NR, NM);
         cuda_buffer_flag_potrf[0] = 0;
       }
       tpotrf_dev_cuda('L', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
       cudaDeviceSynchronize();
       tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
       cudaDeviceSynchronize();
#endif
       nvtxRangeEnd(id_dpotrf);
       flop_count += 1.0/3.0 * NR * NR * NR + 0.5*NR*NR + 1.0/6.0 * NR;

       magma_event_record(potrf_dev_magma_ev, magma_queue_1);	      
       magma_queue_wait_event(magma_queue_2, potrf_dev_magma_ev);	
       
       // triangular solve, all columns -> potentially split this ... when nd large ... two separate streams.
       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
       nvtxRangeId_t id_ttrsm = nvtxRangeStartA("ttrsm_inLoop");       
       //ttrsm_dev('R', 'L', 'T', 'N', NR+matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[NR], mf_block_lda(IB+1, IB), magma_queue_1);
       ttrsm_dev('R', 'L', 'T', 'N', NR, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[NR], mf_block_lda(IB+1, IB), magma_queue_1);

       if(matrix_nd > 0){
         ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
       }

       nvtxRangeEnd(id_ttrsm);       
       //printf("RJ: ttrsm NR: %d, a: %d, lda: %d, b: %d, ldb: %d\n", NR, mf_block_index(IB-1, IB-1), mf_block_lda(IB-1, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1));

       flop_count += (NR + matrix_nd) * NR * NR;

       cudaDeviceSynchronize();
       //copy_supernode_to_host(blockR_dev, IB);
       copy_supernode_diag(blockR_dev, IB);

    } // end loop IB

    if (matrix_nt > 1)
    {
       IB = matrix_nt-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       swap_pointers(&blockR_dev, &blockM_dev);
       init_supernode(blockR_dev, IB, magma_cudaStream_1);

       gpuErrchk(cudaEventRecord(initBlock_dev_ev, magma_cudaStream_1)); 
       gpuErrchk(cudaStreamWaitEvent(magma_cudaStream_2, initBlock_dev_ev, 0));	

       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       // todo rj: 3-last parameter ZERO in PARDISO
       tgemm_dev('N', 'T', NR, NR, NM, -ONE, &blockM_dev[NM], mf_block_lda(IB, IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);
       flop_count += 2.0* NR * NR * NR;

       if (matrix_nd > 0)
       {
          tgemm_dev('N', 'T', matrix_nd, NR, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[NM], mf_block_lda(IB, IB-1), ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_2);
          flop_count += 2.0*matrix_nd * NR * NM;

          tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockM_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue_2);
          flop_count += 2.0* matrix_nd * matrix_nd * NM;
       }

       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       //tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);    
#ifdef CUDA_POTRF
       if(NR != NM){
         //printf("Different block size! Re-initialize cudaBufferSize. NR = %ld, NM = %ld\n", NR, NM);
         cuda_buffer_flag_potrf[0] = 0;
       }
       tpotrf_dev_cuda('L', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
       cudaDeviceSynchronize();
       tpotrf_dev('L', NR, blockR_dev, mf_block_lda(IB, IB), &info);
       cudaDeviceSynchronize();
#endif
       flop_count += 1.0/3.0 * NR * NR * NR + 0.5*NR*NR + 1.0/6.0 * NR;

    
       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dtrsm RLTN MF[IB,IB] MF[IB+1,IB]
       if (matrix_nd > 0)
       {
          ttrsm_dev('R', 'L', 'T', 'N', matrix_nd, NR, ONE, blockR_dev, mf_block_lda(IB, IB), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_1);
          flop_count += matrix_nd * NR * NR;
       
       }
       //printf("RJ: ttrsm NR: %d, a: %d, lda: %d, b: %d, ldb: %d\n", NR, mf_block_index(IB-1, IB-1), mf_block_lda(IB-1, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1));

       //copy_supernode_to_host(blockR_dev, IB);
       copy_supernode_diag(blockR_dev, IB);

    }

    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       //rj dgemm NT M[IB,IB-1] M[IB,IB-1]
       //rj NM is the same for all blocks 0..nt-1
       tgemm_dev('N', 'T', matrix_nd, matrix_nd, NM, -ONE, &blockR_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), &blockR_dev[mf_dense_block_offset(IB-1)], mf_dense_block_lda(IB-1), ONE, blockDense_dev, matrix_nd, magma_queue_1);
       flop_count += 2.0* matrix_nd * matrix_nd * NM;
   
       //printf("RJ: tgemm NR: %d, NM: %d, a: %d, lda: %d, b: %d, ldb: %d, c: %d, ldc: %d\n", NR, NM, mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB-1), mf_block_lda(IB, IB-1), mf_block_index(IB, IB), mf_block_lda(IB, IB));
       //rj dpotrf MF[IB,IB]
       //tpotrf_dev('L', NR, blockDense_dev, mf_block_lda(IB, IB), &info);
#ifdef CUDA_POTRF
       if(NR != NM){
         //printf("Different block size! Re-initialize cudaBufferSize. NR = %ld, NM = %ld\n", NR, NM);
         cuda_buffer_flag_potrf[0] = 0;
       }
       tpotrf_dev_cuda('L', NR, blockDense_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_potrf, 
                       handle, params, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
       cudaDeviceSynchronize();
       tpotrf_dev('L', NR, blockDense_dev, mf_block_lda(IB, IB), &info);
       cudaDeviceSynchronize();
#endif
       flop_count += 1.0/3.0 * matrix_nd * matrix_nd * matrix_nd + 0.5*matrix_nd * matrix_nd + 1.0/6.0 * matrix_nd;

       //printf("RJ: tpotrf NR: %d, a: %d, lda: %d\n\n", NR, mf_block_index(IB, IB), mf_block_lda(IB, IB));
 
#ifdef PRINT_MSG
       std::cout << "after complete Cholesky factorization" << std::endl;
#endif

       //copy_supernode_to_host(blockDense_dev, IB);
       copy_supernode_diag(blockDense_dev, IB);

    }

   t_fact = get_time(t_fact);
   //std::cout << "gflop count    : " << flop_count / 1e9 << std::endl;
   //std::cout << "time factorize : " << t_fact << std::endl;
   double gflops = flop_count / (1e9*t_fact);
   //printf("GFLOPS factorize Qu : %f\n", gflops);

   //std::cout << "time factorization Qu : " << t_fact << std::endl;
   //std::cout << "gflop count Qu        : " << flop_count / 1e9 << std::endl;

   double t_copy = -omp_get_wtime();
   // copy result from the device to the host:
   T* diag;
   diag = new T[matrix_size];
   cudaMemcpy(diag, diag_dev, matrix_size*sizeof(T), cudaMemcpyDeviceToHost );

   // compute log sum
   log_sum(diag, matrix_size, &logDet);
   logDet = 2*logDet;

   t_copy += omp_get_wtime();

#ifdef PRINT_TIMES
   std::cout << "Log Det Copy & Compute time : " << t_copy << std::endl;

#endif
   //printf("\nnew logDet = %f\n", logDet);

    return gflops;
}

/************************************************************************************************/

// After factorisation of A = L*L^T solve (L*L^T) x = b by : forward pass L*y = b, backward pass L^T x = y
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
    double flop_count = 0;

#ifdef PRINT_MSG
    std::cout <<"calling first block solver in second stage factor now" << std::endl;
#endif

    nvtxRangeId_t id_solve = nvtxRangeStartA("solve");

    double t_solve = get_time(0.0);

    //rj dtrsm LLNN MF[0,0] rhs[0]
    // solve for first diagonal block (using solve for tridiagonal matrix), ie. x_0 = D^-0*b_0
    c_ttrsm('L', 'L', 'N', 'N', NR, nrhs, ONE, &MF[mf_block_index(0, 0)], mf_block_lda(0, 0), &rhs[Bmin[0]], matrix_size);
    flop_count += 2.0 * NR * NR * nrhs;

    //rj IB = 1; IB < NBlock; IB++
    for (IB = 1; IB < matrix_nt; IB++)
    {
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       // compute b_i = b_i - E_i-1 * x_i-1
       //rj dgemm NN M[IB,IB-1] rhs[IB-1] rhs[IB]
       c_tgemm('N', 'N', NR, nrhs, NM, -ONE, &MF[mf_block_index(IB, IB-1)], mf_block_lda(IB, IB-1), &rhs[Bmin[IB-1]], matrix_size, ONE, &rhs[Bmin[IB]], matrix_size);
       flop_count += 2.0 * NR * nrhs * NM;
       // solve for x_i = D^-1_i*b_i (which was updated before)
       //rj dtrsm LLNN MF[IB,IB] rhs[IB]
       c_ttrsm('L', 'L', 'N', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flop_count += 2.0*NR * NR * nrhs; 
    }

    // dense rows at the end
    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       // compute b_n = b_n - (F_1*x_1 + F_2*x_2 + ... F_n-1*x_n-1)
       //rj dgemm NN M[IB,IB-1] rhs[IB-1] rhs[IB]
       for (size_t i = 0; i < NBlock-1; i++)
       {    
          c_tgemm('N', 'N', NR, nrhs, NM, -ONE, &MF[mf_dense_block_index(i)], mf_dense_block_lda(i), &rhs[Bmin[i]], matrix_size, ONE, &rhs[Bmin[IB]], matrix_size);
          flop_count += 2.0*NR * nrhs * NM;
       }
       //rj dtrsm LLNN MF[IB,IB] rhs[IB]
       c_ttrsm('L', 'L', 'N', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flop_count += 2.0*NR * NR * nrhs;
    }

    //Backward pass
    if (matrix_nd > 0)
    {
       IB = NBlock-1;
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       // solve D^T_n*x_n = y_n for x_n
       //rj dtrsm LLTN MF[NBlock-1,NBlock-1] rhs[NBlock-1]
       c_ttrsm('L', 'L', 'T', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flop_count += NR * NR * nrhs;
       // now update y everywhere : y_i = y_i - F^T_i * x_n
       //rj dgemm TN M[IB,IB-1] rhs[IB] rhs[IB-1]
       for (size_t i = 0; i < NBlock-1; i++)
       {
          c_tgemm('T', 'N', NM, nrhs, NR, -ONE, &MF[mf_dense_block_index(i)], mf_dense_block_lda(i), &rhs[Bmin[IB]], matrix_size, ONE, &rhs[Bmin[i]], matrix_size);
          flop_count += 2.0*NM * nrhs * NR;
       }
    }

    // loop for tridiagonal block structure (right-hand side y already includes updates from dense rows/columns)
    for (IB = matrix_nt-1; IB > 0; IB--)
    {
       NR  = Bmax[IB]-Bmin[IB];
       NM  = Bmax[IB-1]-Bmin[IB-1];

       // compute D^T_i x_i = y_i
       //rj dtrsm LLTN MF[IB,IB] rhs[IB]
       c_ttrsm('L', 'L', 'T', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);
       flop_count += NR * NR * nrhs;
       // compute y_i = y_i - E_i^T * x_i+1
       //rj dgemm TN M[IB,IB-1] rhs[IB] rhs[IB-1]
       c_tgemm('T', 'N', NM, nrhs, NR, -ONE, &MF[mf_block_index(IB, IB-1)], mf_block_lda(IB, IB-1), &rhs[Bmin[IB]], matrix_size, ONE, &rhs[Bmin[IB-1]], matrix_size);
       flop_count += 2.0 * NM * nrhs * NR; 
    }

    IB = 0;
    NR = Bmax[IB]-Bmin[IB];

   // compute D^T_0 x_0 = y_0
    //rj dtrsm LLTN MF[NBlock-1,NBlock-1] rhs[NBlock-1]
    c_ttrsm('L', 'L', 'T', 'N', NR, nrhs, ONE, &MF[mf_block_index(IB, IB)], mf_block_lda(IB, IB), &rhs[Bmin[IB]], matrix_size);

    flop_count += NR * NR * nrhs;
    t_solve = get_time(t_solve);

    nvtxRangeEnd(id_solve);

    double gflops = flop_count / (1e9*t_solve);
    //printf("GFLOPS solve : %f\n", gflops);


#ifdef PRINT_MSG
    std::cout << "after forward-backword solve in 2nd stage factor" << std::endl;
#endif


    return gflops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::ThirdStageRGF(T *tmp1_dev,T *tmp2_dev, int cpy_indicator)
{
   int info;
   size_t IB;
   size_t NR,NM,NP,ND;
   T ONE      = f_one();
   T ZERO     = f_zero();

   ND = matrix_nd;

   // **************************** better spot? **************************** //
   // need identity matrix of size ns*ns for later
   T *eye_dev;
   mem_alloc_dev = allocate_data_on_device((void**)&eye_dev,matrix_ns*matrix_ns*sizeof(T));  
   init_eye_on_dev(eye_dev,matrix_ns,0);

   // extra buffer allocation
   T *G_LastDense_dev;   // store Sigma_n+1n+1 block
   T *G_dense_i_dev; // store Sigma_n+1i block -> size ns*nb (i.e. b_size*nb)
   T *tmp_AH_dev;    // temporary buffer only for arrowhead part (necessary if want parallelism)
   T *tmp3_dev;  // extra buffer of size nb*ns (ie. nb*b_size)
   T *tmp4_dev;  // extra buffer of size ns*ns

   allocate_data_on_device((void**)&G_LastDense_dev,ND*ND*sizeof(T));   
   allocate_data_on_device((void**)&G_dense_i_dev  ,ND*matrix_ns*sizeof(T));   
   allocate_data_on_device((void**)&tmp_AH_dev     ,ND*matrix_ns*sizeof(T));
   allocate_data_on_device((void**)&tmp3_dev       ,ND*matrix_ns*sizeof(T));
   mem_alloc_dev = allocate_data_on_device((void**)&tmp4_dev       ,matrix_ns*matrix_ns*sizeof(T));  

#ifdef PRINT_MSG
   printf("in 3rd Stage Factor. Allocated GPU memory sel Inv: %ld, cpy ind %d\n", mem_alloc_dev, cpy_indicator);
#endif

#ifdef PRINT_MSG
   // temporary buffer on host, just to check results ...
   T *tmp3; // size ns*nb
   T *tmp4; // size ns*ns

   cudaMallocHost(&tmp3,ND*matrix_ns*sizeof(T));
   cudaMallocHost(&tmp4,matrix_ns*matrix_ns*sizeof(T));
#endif

   //printf("\n*** new version selected inverse... ***\n\n");

   double flop_count = 0;
   nvtxRangeId_t id_inv = nvtxRangeStartA("Inversion");
   double t_inv = get_time(0.0);

   if (matrix_nd > 0)
   {
      //dense block
      IB = NBlock-1;
      NR = Bmax[IB]-Bmin[IB];

      // *************** Gn+1n+1 ************ //
      // copies lower triangular part of blockDense_dev to G_LastDense_dev
      tlacpy_dev('L', NR, NR, blockDense_dev, mf_block_lda(IB, IB), G_LastDense_dev, NR, magma_queue_1);

      cudaDeviceSynchronize();

      // make matrix lower triangular ?!
      // redirect to magma_queue_1
      tril_dev(G_LastDense_dev, NR, NR);

      // solve inv(A)*L = inv(U) for inv(A) by inverting U 
      // G = L^-T * L^-1 => G*L = L^-T
      // tmp1_dev contains lower-triangular matrix, if 2nd argument 'N', maybe means U = L^T?
      // only computes inv(L)??
      // redirect to magma_queue_1

#ifdef CUDA_POTRF
     cuda_buffer_flag_trtri[0] = 0;
     ttrtri_dev_cuda('L', 'N', NR, G_LastDense_dev, NR, info_cuda, cuda_buffer_flag_trtri, 
               handle, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
   // magma_queue currently not supported
    cudaDeviceSynchronize(); 
    ttrtri_dev('L', 'N', NR, G_LastDense_dev, NR, &info);
#endif
      flop_count += 1.0/3.0 * NR * NR * NR + 2.0/3.0 * NR;
      cudaDeviceSynchronize();

      // if tmp1_dev contains L^-1, then
      // C = L^-T*L^-1 -> beta = 0, write into blockDense_dev
      tgemm_dev('T', 'N', NR, NR, NR, ONE, G_LastDense_dev, NR, G_LastDense_dev, NR, ZERO, blockDense_dev, mf_block_lda(IB, IB), magma_queue_1);
      flop_count += 2.0 * NR * NR * NR;
      if(cpy_indicator == 2){
         cudaDeviceSynchronize();
         // instead of full block, just copy elements that are nnz in Q

         // compute indices: total ns^2*nt+nb^2 -> want to write into last nb^s indices
         ind_invBlks_fi = invblks_diag_block_index(IB);
         //memcpy_to_host(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         //memcpy_to_host(invBlks, blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         cudaMemcpy(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T), cudaMemcpyDeviceToHost);

#ifdef PRINT_MSG
         printf("first index final small diagonal block: %ld\n", ind_invBlks_fi);
         printf("final small block : \n");
         for(int i=0; i<ND*ND; i++){
            printf("%f ", invBlks[ind_invBlks_fi+i]);
         }
         printf("\n");
#endif

      }

      //last non-dense block
      IB = NBlock-2;
      NR = Bmax[IB]-Bmin[IB];
      NP = Bmax[IB+1]-Bmin[IB+1];

      //printf("IB = %d, NR = %d, NP = %d\n", IB, NR, NP);

      // *************** Gnn ************ //
      // redirect to magma_queue_1
      tril_dev(blockR_dev, mf_block_lda(IB, IB), NR);
      // solve inv(A)*L = inv(U) for inv(A) by inverting U 
      // G = L^-T * L^-1 => G*L = L^-T
      // redirect to magma_queue_1
#ifdef CUDA_POTRF
     // factorization first block -> set buffer flag to zero!
     if(NR != NP){
         cuda_buffer_flag_trtri[0] = 0;
     }
     ttrtri_dev_cuda('L', 'N', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_trtri, 
                       handle, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
     ttrtri_dev('L', 'N', NR, blockR_dev, mf_block_lda(IB, IB), &info);
#endif

      flop_count +=  1.0/3.0 * NR * NR * NR + 2.0/3.0 * NR;

      cudaDeviceSynchronize();

      // initialize identity matrix temp1_dev of size NR = ns (now)
      // redirect to magma_queue_1 -> potentially later in copy stream, but careful need tmp1_dev before
      init_eye_on_dev(tmp1_dev,NR,magma_cudaStream_1);
      // temp2_dev = L_Fn^T*Gn+1
      tgemm_dev('T', 'N', NR, NP, NP, ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), blockDense_dev, mf_block_lda(IB+1, IB+1), ZERO, tmp2_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NP * NP;
      // temp1_dev = temp2_dev*L_Fn + I 
      tgemm_dev('N', 'N', NR, NR, NP, ONE, tmp2_dev, NR, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), ONE, tmp1_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NR * NP;
      // temp2_dev = blockR_dev^T*temp1_dev <=> temp2_dev = L_Dn^-T*temp1_dev
      tgemm_dev('T', 'N', NR, NR, NR, ONE, blockR_dev, mf_block_lda(IB, IB), tmp1_dev, NR, ZERO, tmp2_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NR * NR;
      // tmp1_dev = temp2_dev*L_Dn^-1 => Gn
      tgemm_dev('N', 'N', NR, NR, NR, ONE, tmp2_dev, NR, blockR_dev, mf_block_lda(IB, IB), ZERO, tmp1_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NR * NR;

      if(cpy_indicator == 2){
         cudaDeviceSynchronize();
         // compute indices: total ns^2*nt+nb^2 -> now at last ns x ns block
         ind_invBlks_fi = invblks_diag_block_index(IB);
         //memcpy_to_host(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         //memcpy_to_host(invBlks, blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         cudaMemcpy(&invBlks[ind_invBlks_fi], tmp1_dev, matrix_ns*matrix_ns*sizeof(T), cudaMemcpyDeviceToHost);
#ifdef PRINT_MSG         
         printf("first index last ns x ns diagonal block: %ld\n", ind_invBlks_fi);
         printf("last ns x ns block : \n");
         for(int i=0; i<NR*NR; i++){
            printf("%f ", invBlks[ind_invBlks_fi+i]);
         }
         printf("\n");
#endif
      }

      // *************** Gn+1n ************ //
      // Sn+1n    = -Sn+1n+1*L_Fn*L_Dn+1^-1
      // tmp3_dev = blockDense_dev*&blockR_dev[mf_dense_block_offset(IB)]*blockR_dev
      // step 1: tmp3 = Sn+1n+1*L_Fn
      //printf("NP = %d, NR = %d, mf_dense_block_lda(IB) = %d\n", NP, NR, mf_dense_block_lda(IB));
      // dgemm(nothing/transpose A, nothing/transpose B, #rows A, #cols B, #cols A, ...)
      tgemm_dev('N', 'N', NP, NR, NP, ONE, blockDense_dev, mf_block_lda(IB+1, IB+1), &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), ZERO, tmp3_dev, NP, magma_queue_1);
      flop_count += 2.0 * NP * NR * NP;

      // step 2: -tmp3_dev*L_Dn^-1 => -tmp3_dev*blockR_dev, dim(tmp3_dev) = (nb, ns), dim(blockR_dev) = (ns, ns)
      // nb, ns, ns
      //printf("mf_block_lda(IB, IB) = %d\n", mf_block_lda(IB, IB));
      tgemm_dev('N', 'N', NP, NR, NR, -ONE, tmp3_dev, NP, blockR_dev, mf_block_lda(IB, IB), ZERO, G_dense_i_dev, NP, magma_queue_1);
      flop_count += 2.0 * NP * NR * NR;

#ifdef PRINT_MSG
      if(matrix_ns*ND < 200){
         cudaMemcpy(tmp3, G_dense_i_dev, matrix_ns*ND*sizeof(T), cudaMemcpyDeviceToHost);
         printf("Sigma_n+1n : \n");
         for(int i=0; i<matrix_ns*ND; i++){
            printf("%f ", tmp3[i]);
         }
         printf("\n"); 
      }
#endif

      cudaDeviceSynchronize();

      // do copying in other stream ...
      if(cpy_indicator == 0){
         // when we only want the diagonal
         copy_supernode_diag(blockDense_dev, NBlock-1);
      } else if(cpy_indicator == 1){
         // want nnzQ entries in CSC format
         extract_nnzA(blockDense_dev, NBlock-1);
      } else if(cpy_indicator == 2){
         // copy over Gn+1n
         ind_invBlks_fi = invblks_dense_block_index(IB);
         //memcpy_to_host(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         //memcpy_to_host(invBlks, blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         cudaMemcpy(&invBlks[ind_invBlks_fi], G_dense_i_dev, matrix_nd*matrix_ns*sizeof(T), cudaMemcpyDeviceToHost);
#ifdef PRINT_MSG
         if(matrix_ns*ND < 200){
            printf("first index last ns x nd diagonal block: %ld\n", ind_invBlks_fi);
            printf("last ns x ns small block : \n");
            for(int i=0; i<matrix_nd*matrix_ns; i++){
               printf("%f ", invBlks[ind_invBlks_fi+i]);
            }
            printf("\n");
         }
#endif
      }
      cudaDeviceSynchronize();
      // *************** Gnn ************ //
      // copy tmp1_dev to blockR_dev
      tlacpy_dev('F', NR, NR, tmp1_dev, NR, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);

#ifdef PRINT_MSG
      if(matrix_ns*matrix_ns < 200){
         cudaMemcpy(tmp4, tmp1_dev, matrix_ns*matrix_ns*sizeof(T), cudaMemcpyDeviceToHost);
         printf("Sigma_nn : \n");
         for(int i=0; i<matrix_ns*matrix_ns; i++){
            printf("%f ", tmp4[i]);
         }
         printf("\n");
      }
#endif

   }
   else
   {
      //last non-dense block
      IB = NBlock-1;
      NR = Bmax[IB]-Bmin[IB];

      // copy lower triangular block of blockR_dev to tmp1_dev
      tlacpy_dev('L', NR, NR, blockR_dev, mf_block_lda(IB, IB), tmp1_dev, NR, magma_queue_1);
      cudaDeviceSynchronize();

      // make make tmp1_dev lower triangular
      tril_dev(tmp1_dev, NR, NR);

      // compute L^-1 = inv(tmp1_dev) which is lower triangular
#ifdef CUDA_POTRF
     // factorization first block -> set buffer flag to zero!
     cuda_buffer_flag_trtri[0] = 0;
     ttrtri_dev_cuda('L', 'N', NR, tmp1_dev, NR, info_cuda, cuda_buffer_flag_trtri, 
                       handle, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else
      // magma_queue currently not supported
      cudaDeviceSynchronize(); 
      ttrtri_dev('L', 'N', NR, tmp1_dev, NR, &info);
#endif

      flop_count += 1.0/3.0 * NR * NR * NR + 2.0/3.0*NR;
      
      // blockR_dev = L^-T*L^-1
      tgemm_dev('T', 'N', NR, NR, NR, ONE, tmp1_dev, NR, tmp1_dev, NR, ZERO, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);
      flop_count += 2.0 * NR * NR * NR;

      if(cpy_indicator == 2){
         cudaDeviceSynchronize();
         // compute indices: total ns^2*nt+nb^2 -> now at last ns x ns block
         ind_invBlks_fi = invblks_diag_block_index(IB);
         //memcpy_to_host(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         //memcpy_to_host(invBlks, blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         cudaMemcpy(&invBlks[ind_invBlks_fi], tmp1_dev, matrix_ns*matrix_ns*sizeof(T), cudaMemcpyDeviceToHost);
#ifdef PRINT_MSG   
         if(NR*NR < 200){
            printf("first index last ns x ns diagonal block case with no fixed effects : %ld\n", ind_invBlks_fi);
            printf("last ns x ns block : \n");
            for(int i=0; i<NR*NR; i++){
               printf("%f ", invBlks[ind_invBlks_fi+i]);
            }
            printf("\n");
         }
#endif
      }

   }
      /*
      T* tmp_host;
      cudaMallocHost(&tmp_host, (NR*NR) * sizeof(T));
      tlacpy_dev('F', NR, NR, blockR_dev, mf_block_lda(IB,IB), tmp4_dev, NR, magma_queue_1);
      cudaDeviceSynchronize();
      gpuErrchk(cudaMemcpy(&tmp_host[0], &tmp4_dev[0], NR*NR*sizeof(double), cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();

      std::string fileName = "invBlock_cpyInd" + to_string(cpy_indicator) + "_IB" + to_string(IB) + ".txt";
      ofstream file(fileName,    ios::out | ::ios::trunc);

      for(int i=0; i<NR*NR; i++){
         file << std::setprecision(15) << tmp_host[i] << endl;
      }

      file.close();
      cudaFreeHost(tmp_host);
      */

      // -> already copy next block to device
      cudaDeviceSynchronize();
      copy_supernode_to_device(blockP_dev, matrix_nt-2, copyStream);

   //second-last non-dense block .. 0-block
   for (int IBi = matrix_nt-2; IBi > -1; IBi--)
   {
      IB = IBi;
      NR = Bmax[IB]-Bmin[IB];
      NP = Bmax[IB+1]-Bmin[IB+1];

      // don't swap pointers before copy_supernode_to_device() isn't done
      cudaDeviceSynchronize();
      // before swap: blockR_dev: Sigma_IBi+1, blockP_dev: CholeskyFactor IBi, then swap
      swap_pointers(&blockR_dev, &blockP_dev);
      cudaDeviceSynchronize();

      // copy diagonal from previous iteration into buffer on device
      // redirect to copy stream
      if(cpy_indicator == 0){
         copy_supernode_diag(blockP_dev, IB+1);
      } else if(cpy_indicator == 1){
         //copy_supernode_diag(blockP_dev, IB+1);
         // write G_dense_i_dev into "correct spot in blockP_dev"
         if(matrix_nd > 0){
            //printf("ND = %ld\n", ND);
            tlacpy_dev('F', ND, NR, G_dense_i_dev, ND, &blockP_dev[mf_dense_block_offset(IB+1)], mf_dense_block_lda(IB+1), magma_queue_1);
         }
         cudaDeviceSynchronize();
         extract_nnzA(blockP_dev, IB+1);
         //exit(1);
      }

      /*
      cudaDeviceSynchronize();
      copy_supernode_to_device(blockR_dev, IB);
      cudaDeviceSynchronize();
      */

      // tril(blockR_dev)
      tril_dev(blockR_dev, mf_block_lda(IB, IB), NR);
      // if L = blockR_dev, then compute L^-1
      nvtxRangeId_t id_trtri = nvtxRangeStartA("TriangularSolve_inLoop");
#ifdef CUDA_POTRF
     // factorization first block -> set buffer flag to zero!
     if(NR != NP){
      cuda_buffer_flag_trtri[0] = 0;
     }
     ttrtri_dev_cuda('L', 'N', NR, blockR_dev, mf_block_lda(IB, IB), info_cuda, cuda_buffer_flag_trtri, 
                       handle, magma_cudaStream_1, dev_size, host_size, mem_cuda_dev, mem_cuda_host);
#else 
    ttrtri_dev('L', 'N', NR, blockR_dev, mf_block_lda(IB, IB), &info);
#endif
      nvtxRangeEnd(id_trtri);
      flop_count += 1.0/3.0 * NR * NR * NR + 2.0/3.0 * NR;

      // initialize eye(NR, NR)
      init_eye_on_dev(tmp1_dev,NR,magma_cudaStream_1);
      //cudaDeviceSynchronize();

      // tmp2_dev = blockR_dev^T*blockP_dev => L_Ei^T*Gi+1
      tgemm_dev('T', 'N', NR, NP, NP, ONE, &blockR_dev[NR], mf_block_lda(IB+1, IB), blockP_dev, mf_block_lda(IB+1, IB+1), ZERO, tmp2_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NP * NP;

      // temp1_dev = tmp2_dev*blockR_dev + I => tmp2_dev*L_Ei + I
      tgemm_dev('N', 'N', NR, NR, NP, ONE, tmp2_dev, NR, &blockR_dev[NR], mf_block_lda(IB+1, IB), ONE, tmp1_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NR * NP;
      //cudaDeviceSynchronize(); //

      // blockP_dev not in use anymore until next iter where pointers are swapped 
      // -> already copy next block to device 
      // use cuda Events
      if(IB > 0){
         //cudaDeviceSynchronize();
         gpuErrchk(cudaEventRecord(potrf_dev_ev, magma_cudaStream_1)); 
         gpuErrchk(cudaStreamWaitEvent(copyStream, potrf_dev_ev, 0));

         copy_supernode_to_device(blockP_dev, IB-1, copyStream);
         //cudaDeviceSynchronize();     
      }

      if (matrix_nd > 0)
      {
         // tmp3_dev = L_Fi^T*Gn+1n+1 // 
         tgemm_dev('T', 'N', NR, ND, ND, ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), blockDense_dev, mf_block_lda(NBlock-1, NBlock-1), ZERO, tmp3_dev, NR, magma_queue_1);
         flop_count += 2.0 * NR * ND * ND;

         // tmp1_dev = tmp3_dev*L_Fi + tmp1_dev
         tgemm_dev('N', 'N', NR, NR, ND, ONE, tmp3_dev, NR, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), ONE, tmp1_dev, NR, magma_queue_1);
         flop_count += 2.0 * NR * NR * ND;

         // store Gn+1i+1 in G_dense_i_dev -> overwrite in every iteration
         // total: Gii = L_Di^-T*(L_Ei^T*Gi+1i+1*L_Ei + L_Fi^T*Gn+1n+1*L_Fi + L_Fi^T*Gn+1i+1*L_Ei + L_Ei^T*Gi+1n+1*L_Fi)L_Di^-1
         // need: L_Fi^T*Gi+1n+1*L_Ei (& its transpose) i.e. L_Ei^T*Gn+1i+1*L_Fi
         // tmp4_dev = L_Fi^T*Gn+1i+1*L_Ei
         // step 1: tmp4_dev = L_Fi^T*Gn+1i+1, dim(tmp2_dev) = (ns,ns)
         tgemm_dev('T', 'N', NP, NR, ND, ONE, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), G_dense_i_dev, ND, ZERO, tmp4_dev, NR, magma_queue_1);
         flop_count += 2.0 * NP * NR * ND;

#ifdef PRINT_MSG
         if(NR*NR < 200){
            cudaMemcpy(tmp4, tmp4_dev,NR*NR*sizeof(T), cudaMemcpyDeviceToHost);
            printf("L_Fi^T*Gn+1i+1: \n");
            for(int i=0; i<NR*NR; i++){
               printf("%f ", tmp4[i]);
            }
            printf("\n");
         } 
#endif

         // tmp2_dev = tmp4_dev*L_Ei
         tgemm_dev('N', 'N', NR, NP, NR, ONE, tmp4_dev, NR, &blockR_dev[NR], mf_block_lda(IB+1, IB), ZERO, tmp2_dev, NR, magma_queue_1);
         flop_count += 2.0 * NR * NP * NR;

#ifdef PRINT_MSG
         if(NR*NR < 200){
            cudaMemcpy(tmp4, tmp4_dev,NR*NR*sizeof(T), cudaMemcpyDeviceToHost);
            printf("L_Fi^T*Gn+1i+1*L_Ei:\n");
            for(int i=0; i<NR*NR; i++){
               printf("%f ", tmp4[i]);
            }
            printf("\n");
         }
#endif

         // copy tmp2 into temp4 
         tlacpy_dev('F', NR, NR, tmp2_dev, NR, tmp4_dev, NR, magma_queue_1);
         // -> then tmp4 = tmp2' + tmp4
         tgemm_dev('T', 'N', NR, NR, NR, ONE, tmp2_dev, NR, eye_dev, NR, ONE, tmp4_dev, NR, magma_queue_1);
         flop_count += 2.0 * NR * NR * NR;

#ifdef PRINT_MSG
         if(NR*NR < 200){
            cudaMemcpy(tmp4, tmp4_dev,NR*NR*sizeof(T), cudaMemcpyDeviceToHost);
            printf("L_Fi^T*Gn+1i+1*L_Ei + t(L_Fi^T*Gn+1i+1*L_Ei): \n");
            for(int i=0; i<NR*NR; i++){
               printf("%f ", tmp4[i]);
            }
            printf("\n");
         }
#endif

         // tmp1_dev = tmp4_dev + tmp1_dev (rest happens outside this if statement)
         tgemm_dev('N', 'N', NR, NR, NR, ONE, tmp4_dev, NR, eye_dev, NR, ONE, tmp1_dev, NR, magma_queue_1);
         flop_count += 2.0 * NR * NR * NR;

#ifdef PRINT_MSG
         if(NR*NR < 200){
            cudaMemcpy(tmp4, tmp1_dev,NR*NR*sizeof(T), cudaMemcpyDeviceToHost);
            printf("tmp4_dev: \n");
            for(int i=0; i<NR*NR; i++){
               printf("%f ", tmp4[i]);
            }
            printf("\n");
         }
#endif
         // compute inv Dense rows
         // Gn+1i = - (Gn+1i+1*L_Ei + Gn+1n+1*L_Fi)*L_Di^-1 , dim(Gn+1i+1*L_Ei) = dim(Gn+1i+1) = (nb, ns)
         // tmp_AH_dev = G_dense_i_dev*(&blockR_dev[NR], mf_block_lda(IB+1, IB))
         tgemm_dev('N', 'N', ND, NP, NR, ONE, G_dense_i_dev, ND, &blockR_dev[NR], mf_block_lda(IB+1, IB), ZERO, tmp_AH_dev, ND, magma_queue_2);
         flop_count += 2.0 * ND * NP * NR;

#ifdef PRINT_MSG
         if(NR*ND < 200){
            cudaMemcpy(tmp3, tmp3_dev,NR*ND*sizeof(T), cudaMemcpyDeviceToHost);
            printf("tmp3_dev: \n");
            for(int i=0; i<ND*NR; i++){
               printf("%f ", tmp3[i]);
            }
            printf("\n");
         }
#endif

         // compute inv Dense rows
         // tmp_AH_dev = Gn+1n+1*L_Fi + tmp_AH_dev 
         tgemm_dev('N', 'N', ND, NR, ND, ONE, blockDense_dev, ND, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), ONE, tmp_AH_dev, ND, magma_queue_2);
         flop_count += 2.0 * ND * NR * ND;

#ifdef PRINT_MSG    
         if(NR*ND < 200){
            cudaMemcpy(tmp3, tmp3_dev,NR*ND*sizeof(T), cudaMemcpyDeviceToHost);
            printf("tmp3_dev: \n");
            for(int i=0; i<ND*NR; i++){
               printf("%f ", tmp3[i]);
            }
            printf("\n");
         }
#endif
         // compute inv Dense rows
         // Gn+1i = tmp_AH_dev*L_Di^-1 => G_dense_i_dev*blockR_dev
         tgemm_dev('N', 'N', ND, NR, NR, -ONE, tmp_AH_dev, ND, blockR_dev, mf_block_lda(IB, IB), ZERO, G_dense_i_dev, ND, magma_queue_1);
         flop_count += 2.0 * ND * NR * NR;

#ifdef PRINT_MSG
         if(NR*ND < 200){
            cudaMemcpy(tmp3, G_dense_i_dev,NR*ND*sizeof(T), cudaMemcpyDeviceToHost);
            printf("tmp3_dev: \n");
            for(int i=0; i<ND*NR; i++){
               printf("%f ", tmp3[i]);
            }
            printf("\n");
         }
#endif

         if(cpy_indicator == 2){
            // copy over Gn+1i
            ind_invBlks_fi = invblks_dense_block_index(IB);
            //memcpy_to_host(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
            //memcpy_to_host(invBlks, blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
            cudaMemcpy(&invBlks[ind_invBlks_fi], G_dense_i_dev, matrix_nd*matrix_ns*sizeof(T), cudaMemcpyDeviceToHost);
#ifdef PRINT_MSG    
         if(matrix_nd*matrix_ns < 200){
            printf("first index of i-th ns x nd diagonal block: %ld\n", ind_invBlks_fi);
            printf(" i-th ns x nd dense block : \n");
            for(int i=0; i<matrix_nd*matrix_ns; i++){
               printf("%f ", invBlks[ind_invBlks_fi+i]);
            }
            printf("\n");
         }
 #endif
         } // end cpy invBlks

      } // end if matrix_nd > 0

      // tmp2_dev = blockR_dev^T*tmp1_dev => tmp2_dev = L^-T*tmp1_dev
      tgemm_dev('T', 'N', NR, NR, NR, ONE, blockR_dev, mf_block_lda(IB, IB), tmp1_dev, NR, ZERO, tmp2_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NR * NR; 

      // tmp1_dev = tmp2_dev*L^-1
      tgemm_dev('N', 'N', NR, NR, NR, ONE, tmp2_dev, NR, blockR_dev, mf_block_lda(IB, IB), ZERO, tmp1_dev, NR, magma_queue_1);
      flop_count += 2.0 * NR * NR * NR; 

#ifdef PRINT_MSG
      if(NR*NR < 200){
         cudaMemcpy(tmp4, tmp1_dev, NR*NR*sizeof(T), cudaMemcpyDeviceToHost);
         printf("Gii: \n");
         for(int i=0; i<NR*NR; i++){
            printf("%f ", tmp4[i]);
         }
         printf("\n");
      }
#endif

      if(cpy_indicator == 2){
         cudaDeviceSynchronize();
         // compute indices: total ns^2*nt+nb^2 -> now at last ns x ns block
         ind_invBlks_fi = invblks_diag_block_index(IB);
         //memcpy_to_host(&invBlks[ind_invBlks_fi], blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         //memcpy_to_host(invBlks, blockDense_dev, matrix_nd*matrix_nd*sizeof(T));
         cudaMemcpy(&invBlks[ind_invBlks_fi], tmp1_dev, matrix_ns*matrix_ns*sizeof(T), cudaMemcpyDeviceToHost);
#ifdef PRINT_MSG     
         if(NR*NR < 200){
            printf("first index last ns x ns diagonal block case with no fixed effects : %ld\n", ind_invBlks_fi);
            printf("i-th ns x ns block : \n");
            for(int i=0; i<NR*NR; i++){
               printf("%f ", invBlks[ind_invBlks_fi+i]);
            }
            printf("\n");
         }
#endif
      }

      // copy tmp1_dev to blockR_dev
      tlacpy_dev('F', NR, NR, tmp1_dev, NR, blockR_dev, mf_block_lda(IB, IB), magma_queue_1);

   } // end loop over nt

#ifdef PRINT_MSG
   if(cpy_indicator == 2){
      printf("array containing all neccessary inv blk entries: \n");
      for(int i=0; i<(matrix_ns+matrix_nd)*matrix_ns*matrix_nt+matrix_nd*matrix_nd; i++){
            printf("%f ", invBlks[i]);
      }
      printf("\n");
   }
#endif

   t_inv = get_time(t_inv);
   double gflops = flop_count / (1e9*t_inv);
   printf("GFLOPS sel. inversion : %f, time selInv : %f\n", gflops, t_inv);

   nvtxRangeEnd(id_inv);

   cudaDeviceSynchronize();

   // copy diagonal of first block into buffer
   if(cpy_indicator == 0){
      copy_supernode_diag(blockR_dev, 0);
   }

   if(cpy_indicator == 1){
      if(ND > 0){
         tlacpy_dev('F', ND, NR, G_dense_i_dev, ND, &blockR_dev[mf_dense_block_offset(IB)], mf_dense_block_lda(IB), magma_queue_1);
      }
      cudaDeviceSynchronize();
      extract_nnzA(blockR_dev, 0);
   }

   // free memory of extra buffers
   deallocate_data_on_dev(G_LastDense_dev, ND*ND*sizeof(T));
   deallocate_data_on_dev(eye_dev       ,matrix_ns*matrix_ns*sizeof(T));
   deallocate_data_on_dev(G_dense_i_dev ,ND*matrix_ns*sizeof(T));
   deallocate_data_on_dev(tmp_AH_dev    ,ND*matrix_ns*sizeof(T));

   deallocate_data_on_dev(tmp3_dev,    ND*matrix_ns*sizeof(T));
   mem_alloc_dev = deallocate_data_on_dev(tmp4_dev,    matrix_ns*matrix_ns*sizeof(T));

#ifdef PRINT_MSG
   printf("Allocated GPU memory after sel Inv: %ld, cpyInd %d\n", mem_alloc_dev, cpy_indicator);

   cudaFreeHost(tmp3);
   cudaFreeHost(tmp4);
#endif

   return gflops;
} // end thirdstageFactor

template <class T>
void RGF<T>::initialize_MF_host(){

#ifdef PRINT_MSG
   std::cout << "in initialize MF host" << std::endl;
#endif   
   
   // cudaMallocHost takes a long time -> only worth it if we reuse it!
   if(!MF_allocated)
   {
      // memory allocation on CPU 
      // PREVIOUSLY unpinned memory:
      //MF = new T[matrix_n_nonzeros];
      //unsigned int dummy_flag = 0;
      //cudaHostRegister(MF, (matrix_n_nonzeros) * sizeof(T), dummy_flag);
      // pin memory:
      cudaMallocHost(&MF, (matrix_n_nonzeros) * sizeof(T));
      MF_allocated = true;
      //printf("In MF_allocated not allocated.\n");
   }

}


template <class T>
void RGF<T>::initialize_invBlks_host(){

#ifdef PRINT_MSG
   std::cout << "in initialize invBlks host" << std::endl;
#endif   
   
   // cudaMallocHost takes a long time -> only worth it if we reuse it!
   if(!invBlks_allocated)
   {
      double t_invBlk_alloc = get_time(0.0);
      // allocate space for Diagonal blocks and dense rows below
      size_t nnz_invBlks = (matrix_ns+matrix_nd)*matrix_ns*matrix_nt + matrix_nd*matrix_nd;
      cudaMallocHost(&invBlks, nnz_invBlks * sizeof(T));
      t_invBlk_alloc = get_time(t_invBlk_alloc);

      invBlks_allocated = true;

//#ifdef PRINT_MSG
      std::cout << "in initialize invBlks host. nnz(invBlks) : " << nnz_invBlks << ", Allocation time : " << t_invBlk_alloc << std::endl;
//#endif  
   }

}

/************************************************************************************************/

template <class T>
double RGF<T>::factorize(size_t *ia, size_t *ja, T *a)
{

#ifdef PRINT_MSG
   std::cout << "entered factorize." << std::endl;
#endif

    //int GPU_rank;
    //cudaGetDevice(&GPU_rank);
    //int numaNodeID = topo_get_numNode(GPU_rank);

    matrix_ia = ia;
    matrix_ja = ja;
    matrix_a = a;

    //create_blocks();

   double t_allocate_MF_host = get_time(0.0);

   initialize_MF_host();
   
#ifdef PRINT_MSG   
   std::cout << "in factorize. MF_allocated : " << MF_allocated << std::endl; 
#endif

#if 0
   // cudaMallocHost takes a long time -> only worth it if we reuse it!
   if(!MF_allocated)
   {
      // memory allocation on CPU 
      // PREVIOUSLY unpinned memory:
      //MF = new T[matrix_n_nonzeros];
      //unsigned int dummy_flag = 0;
      //cudaHostRegister(MF, (matrix_n_nonzeros) * sizeof(T), dummy_flag);
      // pin memory:
      cudaMallocHost(&MF, (matrix_n_nonzeros) * sizeof(T));
      MF_allocated = true;
      //printf("In MF_allocated not allocated.\n");
   }
#endif

   t_allocate_MF_host = get_time(t_allocate_MF_host);
   //std::cout << "time to allocate memory on CPU: " << t_allocate_MF_host << std::endl;

   double t_allocate_MF_dev = get_time(0.0);

    //Data allocation on GPU -> negligible time
    if (!MF_dev_allocated)
    {
       //printf("In MF_dev_allocated not allocated.\n");
       //MF = new T[matrix_n_nonzeros];
       size_t max_supernode_nnz_dense = matrix_nt > 1 ? matrix_ns*(2*matrix_ns+matrix_nd) : matrix_ns*(matrix_ns+matrix_nd);
       size_t final_supernode_nnz_dense = matrix_nd > 0 ? matrix_nd*matrix_nd : 0;
       allocate_data_on_device((void**)&blockR_dev,max_supernode_nnz_dense*sizeof(T));
       allocate_data_on_device((void**)&blockM_dev,max_supernode_nnz_dense*sizeof(T));
       allocate_data_on_device((void**)&blockP_dev,max_supernode_nnz_dense*sizeof(T));
       allocate_data_on_device((void**)&blockDense_dev,final_supernode_nnz_dense*sizeof(T));
       MF_dev_allocated = true;
    }

   t_allocate_MF_dev = get_time(t_allocate_MF_dev);
   //std::cout << "time to allocate memory on GPU: " << t_allocate_MF_dev << std::endl;

    
    size_t nnz = matrix_ia[matrix_size];
    //std::cout << "nnz = " << nnz << std::endl;
    size_t max_rows = matrix_nt > 1 ? 2*matrix_ns+matrix_nd : matrix_ns+matrix_nd;
    size_t max_cols = max(matrix_ns, matrix_nd);

    //Temp data allocation
    allocate_data_on_device((void**)&ia_dev,(max_cols+1)*sizeof(size_t));
    allocate_data_on_device((void**)&ja_dev,max_rows*max_cols*sizeof(size_t));
    mem_alloc_dev = allocate_data_on_device((void**)&a_dev,max_rows*max_cols*sizeof(T));

#ifdef PRINT_MSG
    printf("Allocated GPU memory Factorize: %ld\n", mem_alloc_dev);
#endif

    //Computation
    double gflops = FirstStageFactor();

     /*std::string fileName = "L.txt"; 
      ofstream file(fileName,    ios::out | ::ios::trunc);

     for(int i=0; i<matrix_n_nonzeros; i++){
        file << std::setprecision(15) << MF[i] << endl;
     }

     file.close();
     */
     
    //Temp data deallocation
    deallocate_data_on_dev(ia_dev,(max_cols+1)*sizeof(size_t));
    deallocate_data_on_dev(ja_dev,max_rows*max_cols*sizeof(size_t));
    mem_alloc_dev = deallocate_data_on_dev(a_dev,max_rows*max_cols*sizeof(T));

#ifdef PRINT_MGS
    printf("Allocated GPU memory after Factorize: %ld\n", mem_alloc_dev);
#endif

    factorization_completed = true;

    return gflops;
}


/************************************************************************************************/

template <class T>
double RGF<T>::factorize_noCopyHost(size_t* ia, size_t* ja, T* a, double &logDet)
{

#ifdef PRINT_MSG	
   printf("In factorize no copy host function.\n");
#endif   
   
   matrix_ia = ia;
   matrix_ja = ja;
   matrix_a = a;

   //create_blocks();

    // Data allocation device
    if (!MF_dev_allocated)
    {
       // don't need to initialize MF 
       //MF = new T[matrix_n_nonzeros];
       size_t max_supernode_nnz_dense = matrix_nt > 1 ? matrix_ns*(2*matrix_ns+matrix_nd) : matrix_ns*(matrix_ns+matrix_nd);
       size_t final_supernode_nnz_dense = matrix_nd > 0 ? matrix_nd*matrix_nd : 0;
       allocate_data_on_device((void**)&blockR_dev,max_supernode_nnz_dense*sizeof(T));
       allocate_data_on_device((void**)&blockM_dev,max_supernode_nnz_dense*sizeof(T));
       allocate_data_on_device((void**)&blockP_dev,max_supernode_nnz_dense*sizeof(T));
       allocate_data_on_device((void**)&blockDense_dev,final_supernode_nnz_dense*sizeof(T));
       MF_dev_allocated = true; // otherwise stuff never gets deleted ...
    }

    size_t nnz = matrix_ia[matrix_size];
    size_t max_rows = matrix_nt > 1 ? 2*matrix_ns+matrix_nd : matrix_ns+matrix_nd;
    size_t max_cols = max(matrix_ns, matrix_nd);

    //Temp data allocation
    allocate_data_on_device((void**)&ia_dev,(max_cols+1)*sizeof(size_t));
    allocate_data_on_device((void**)&ja_dev,max_rows*max_cols*sizeof(size_t));
    allocate_data_on_device((void**)&a_dev,max_rows*max_cols*sizeof(T));

    //////// NEW /////////////
   // allocate arrays to store diagonal elements 
   allocate_data_on_device((void**)&diag_dev,matrix_size*sizeof(T));
   mem_alloc_dev = allocate_data_on_device((void**)&diag_pos_dev,matrix_size*sizeof(size_t));

#ifdef PRINT_MSG
   printf("Allocated GPU memory Factorize noCpyHost: %ld\n", mem_alloc_dev);
#endif

   //Copy diag_pos to device
    memcpy_to_device(diag_pos,diag_pos_dev,matrix_size*sizeof(size_t), NULL );

    //Computation
    double gflops = FirstStageFactor_noCopyHost(logDet);
    //double gflops = FirstStageFactor_noCopyHost_testV(logDet);

    //////// NEW /////////////
    //Data deallocation
    deallocate_data_on_dev(diag_dev,matrix_size*sizeof(T));
    deallocate_data_on_dev(diag_pos_dev,matrix_size*sizeof(size_t));

    //Temp data deallocation
    deallocate_data_on_dev(ia_dev,(max_cols+1)*sizeof(size_t));
    deallocate_data_on_dev(ja_dev,max_rows*max_cols*sizeof(size_t));
    mem_alloc_dev = deallocate_data_on_dev(a_dev,max_rows*max_cols*sizeof(T));

#ifdef PRINT_MSG
    printf("Allocated GPU memory after Factorize noCpyHost: %ld\n", mem_alloc_dev);
#endif

    // factor doesn't exist, hence mark as false
    factorization_completed = false;

    return gflops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::solve(size_t *ia, size_t *ja, T *a, T *b, size_t nrhs)
{
    double gflops = solve(b, b, nrhs);

    return gflops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::solve(size_t *ia, size_t *ja, T *a, T *x, T *b, size_t nrhs)
{

   if (!factorization_completed){
       factorize(ia, ja, a);
   }
   
   // check if solve is performed on the right matrix 
   if(!(ia[matrix_size] == matrix_ia[matrix_size])){
      std::cout << "Matrices don't match!!" << std::endl;
      exit(1);
   }

   // check MF diff put in static array first time
   /*
   static int solve_flag = 0;
   static double *MF_initial;
   
   if(solve_flag == 0){
      printf("in solve, solve flag = 0\n");
      MF_initial = new double[matrix_n_nonzeros];
      for(int i=0; i<matrix_n_nonzeros; i++){
         MF_initial[i] = MF[i];
      }
      solve_flag = 1;
   } else {
      double temp = 0;
      for(int i=0; i<matrix_n_nonzeros; i++){
            temp += (MF_initial[i]-MF[i])*(MF_initial[i]-MF[i]);
        }
        printf("norm(MF-MF_initial) = %f\n", temp);
   }
   */

    //Data allocation
    rhs = new T[matrix_size*nrhs];

    //Copy data
    memcpy(rhs,b,matrix_size*nrhs*sizeof(T));

    //Computation
    double gflops = SecondStageSolve(nrhs);

    //Copy data
    memcpy(x,rhs,matrix_size*nrhs*sizeof(T));

    //Data deallocation
    delete[] rhs;

    return gflops;
}

/************************************************************************************************/

template <class T>
double RGF<T>::RGFdiag(size_t *ia, size_t *ja, T *a, T *diag)
{

   if (!factorization_completed)
       factorize(ia, ja, a);

   // check if number of nnz are the same to check if matrix is the same
   if(!(ia[matrix_size] == matrix_ia[matrix_size])){
      std::cout << "Matrices don't match!!" << std::endl;
      exit(1);
   }

    factorization_completed = false;
    cpy_indicator = 0; // => only copy back diagonal

    //Data allocation
    T *tmp1_dev;
    T *tmp2_dev;

#ifdef PRINT_MSG
    printf("\nRGFSelInv: before allocating data on device now. CPY indicator %d\n", cpy_indicator);
#endif

    allocate_data_on_device((void**)&diag_dev,matrix_size*sizeof(T));
    allocate_data_on_device((void**)&diag_pos_dev,matrix_size*sizeof(size_t));
    allocate_data_on_device((void**)&tmp1_dev,matrix_ns*matrix_ns*sizeof(T));
    mem_alloc_dev = allocate_data_on_device((void**)&tmp2_dev,matrix_ns*matrix_ns*sizeof(T));

#ifdef PRINT_MSG
    printf("Allocated GPU memory sel Inv: %ld, cpy ind %d\n", mem_alloc_dev, cpy_indicator);
#endif

    //Copy diag_pos to device
    memcpy_to_device(diag_pos,diag_pos_dev,matrix_size*sizeof(size_t), NULL );

    //Computation
    double gflops = ThirdStageRGF(tmp1_dev,tmp2_dev, cpy_indicator);

    //Copy data to host
    memcpy_to_host(diag,diag_dev,matrix_size*sizeof(T), NULL );

    //Data deallocation
    deallocate_data_on_dev(diag_dev,matrix_size*sizeof(T));
    deallocate_data_on_dev(diag_pos_dev,matrix_size*sizeof(size_t));
    deallocate_data_on_dev(tmp1_dev,matrix_ns*matrix_ns*sizeof(T));
    deallocate_data_on_dev(tmp2_dev,matrix_ns*matrix_ns*sizeof(T));

#ifdef PRINT_MSG
    printf("Allocated GPU memory after sel Inv: %ld, cpy ind %d\n", mem_alloc_dev, cpy_indicator);
#endif

    return gflops;
}

template <class T>
double RGF<T>::RGFselInv(size_t *ia, size_t *ja, T *a, T *invQ)
{  

   if (!factorization_completed)
       factorize(ia, ja, a);

   // check if number of nnz are the same to check if matrix is the same
   if(!(ia[matrix_size] == matrix_ia[matrix_size])){
      std::cout << "Matrices don't match!!" << std::endl;
      exit(1);
   }

   cpy_indicator = 1; // => copy back all entries of nnzQ

    //Data allocation
    T *tmp1_dev;
    T *tmp2_dev;

    size_t nnz = matrix_ia[matrix_size];

    //inv_a = new T[nnz];
    gpuErrchk(cudaMallocHost((void**)&inv_a, nnz*sizeof(T)));

    //std::cout << "nnz = " << nnz << std::endl;
    size_t max_rows = matrix_nt > 1 ? 2*matrix_ns+matrix_nd : matrix_ns+matrix_nd;
    size_t max_cols = max(matrix_ns, matrix_nd);

#ifdef PRINT_MSG
    printf("\nRGFSelInv: allocating data on device now. CPY indicator %d\n", cpy_indicator);
#endif

    //allocate_data_on_device((void**)&diag_dev,matrix_size*sizeof(T));
    //allocate_data_on_device((void**)&diag_pos_dev,matrix_size*sizeof(size_t));
    allocate_data_on_device((void**)&tmp1_dev,matrix_ns*matrix_ns*sizeof(T));
    allocate_data_on_device((void**)&tmp2_dev,matrix_ns*matrix_ns*sizeof(T));

    //Temp data allocation
    // ALLOCATES MORE MEMORY THAN NEEDED -> only need max nnz for all supernodes
    /*allocate_data_on_device((void**)&inv_ia_dev,(max_cols+1)*sizeof(size_t));
    allocate_data_on_device((void**)&inv_ja_dev,max_rows*max_cols*sizeof(size_t));
    allocate_data_on_device((void**)&inv_a_dev,max_rows*max_cols*sizeof(T));*/

    // compute maximum number of nnz over all supernodes (of sparse input matrix A)
    // only works because only the last set of columns is not of size matrix_ns ...
    get_max_supernode_nnz();
    //printf("final max supernode nnz : %ld\n", max_supernode_nnz);

    allocate_data_on_device((void**)&inv_ia_dev,(max_cols+1)*sizeof(size_t));
    allocate_data_on_device((void**)&inv_ja_dev,max_supernode_nnz*sizeof(size_t));
    allocate_data_on_device((void**)&inv_a_dev,max_supernode_nnz*sizeof(T));

    //Computation
    double gflops = ThirdStageRGF(tmp1_dev,tmp2_dev, cpy_indicator);

   factorization_completed = false; // we have overwritten the Cholesky factor

    //CAREFUL: invBlks are on CPU but not as contiguous array, block-wise !!
    //memcpy_to_host(diag,diag_dev,matrix_size*sizeof(T));

    // restructure array to only contain value array of CSC format
    //size_t nnz_invBlks = (matrix_ns+matrix_nd)*matrix_ns*matrix_nt+matrix_nd*matrix_nd;
    //for(size_t i=0; i<nnz_invBlks; i++){
      //invBlks_ext[i] = invBlks[i];
    //}

    for(size_t i=0; i<nnz; i++){
      invQ[i] = inv_a[i];
    }

    //Data deallocation
    //deallocate_data_on_dev(diag_dev,matrix_size*sizeof(T));
    //deallocate_data_on_dev(diag_pos_dev,matrix_size*sizeof(size_t));
    deallocate_data_on_dev(tmp1_dev,matrix_ns*matrix_ns*sizeof(T));
    deallocate_data_on_dev(tmp2_dev,matrix_ns*matrix_ns*sizeof(T));

    //Temp data deallocation
    deallocate_data_on_dev(inv_ia_dev,(max_cols+1)*sizeof(size_t));
    deallocate_data_on_dev(inv_ja_dev,max_rows*max_cols*sizeof(size_t));
    deallocate_data_on_dev(inv_a_dev,max_rows*max_cols*sizeof(T));

    cudaFreeHost(inv_a);

    return gflops;
}

template <class T>
double RGF<T>::RGFinvBlks(size_t *ia, size_t *ja, T *a, T *invBlks_ext)
{

   printf("!!!CAREFUL: RGFinvBlks function abandoned. Not sure if it gives correct results!!!\n");

   if (!factorization_completed)
       factorize(ia, ja, a);

   // check if number of nnz are the same to check if matrix is the same
   if(!(ia[matrix_size] == matrix_ia[matrix_size])){
      std::cout << "Matrices don't match!!" << std::endl;
      exit(1);
   }

   factorization_completed = false;
   cpy_indicator = 2; // => copy back inv diag blocks in block format

    //std::cout << "nnz = " << nnz << std::endl;
    size_t max_rows = matrix_nt > 1 ? 2*matrix_ns+matrix_nd : matrix_ns+matrix_nd;
    size_t max_cols = max(matrix_ns, matrix_nd);

   initialize_invBlks_host();

    //Data allocation
    T *tmp1_dev;
    T *tmp2_dev;

    //allocate_data_on_device((void**)&diag_dev,matrix_size*sizeof(T));
    //allocate_data_on_device((void**)&diag_pos_dev,matrix_size*sizeof(size_t));
    allocate_data_on_device((void**)&tmp1_dev,matrix_ns*matrix_ns*sizeof(T));
    allocate_data_on_device((void**)&tmp2_dev,matrix_ns*matrix_ns*sizeof(T));

    //Copy diag_pos to device
    //memcpy_to_device(diag_pos,diag_pos_dev,matrix_size*sizeof(size_t));

    //Computation
    double gflops = ThirdStageRGF(tmp1_dev,tmp2_dev, cpy_indicator);

    //CAREFUL: invBlks are on CPU but not as contiguous array, block-wise !!
    //memcpy_to_host(diag,diag_dev,matrix_size*sizeof(T));

    // restructure array to only contain value array of CSC format
    size_t nnz_invBlks = (matrix_ns+matrix_nd)*matrix_ns*matrix_nt+matrix_nd*matrix_nd;
    for(size_t i=0; i<nnz_invBlks; i++){
      invBlks_ext[i] = invBlks[i];
    }

    //Data deallocation
    //deallocate_data_on_dev(diag_dev,matrix_size*sizeof(T));
    //deallocate_data_on_dev(diag_pos_dev,matrix_size*sizeof(size_t));
    deallocate_data_on_dev(tmp1_dev,matrix_ns*matrix_ns*sizeof(T));
    deallocate_data_on_dev(tmp2_dev,matrix_ns*matrix_ns*sizeof(T));

    return gflops;
}

/************************************************************************************************/

template <class T>
T RGF<T>::logDet(size_t *ia, size_t *ja, T *a)
{
    if (!factorization_completed)
       factorize(ia, ja, a);

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

/*
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
*/


/************************************************************************************************/

template <class T>
inline void RGF<T>::get_max_supernode_nnz()
{
    // compute maximum number of nnz over all supernodes
    // only works because only the last set of columns is not of size matrix_ns ...
    max_supernode_nnz = 0;
    size_t count_nnz = 0;
    // # supernodes: matrix_nt + 1 
    for(int supernode=0; supernode<matrix_nt+1; supernode++){ 
      size_t supernode_fc = supernode * matrix_ns;
      size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
      size_t supernode_nnz = matrix_ia[supernode_lc] - matrix_ia[supernode_fc];
      count_nnz += supernode_nnz;
      max_supernode_nnz = max(supernode_nnz, max_supernode_nnz);

#ifdef PRINT_MSG
      printf("supernode= %d, supernode_nnz = %ld, max_supernode_nnz= %ld\n", supernode, supernode_nnz, max_supernode_nnz);
#endif
    }

}


/************************************************************************************************/

template <class T>
inline void RGF<T>::init_supernode(T *M_dev, size_t supernode, cudaStream_t stream)
{

    size_t supernode_fc = supernode * matrix_ns;
    size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
    size_t supernode_nnz = matrix_ia[supernode_lc] - matrix_ia[supernode_fc];
    size_t supernode_offset = matrix_ia[supernode_fc];
    size_t rows = mf_block_lda(supernode, supernode);
    size_t cols = supernode_lc - supernode_fc;
    size_t supernode_size = rows * cols;

    cudaMemsetAsync((void*)M_dev, 0, supernode_size*sizeof(T), stream );

    //printf("in init supernode. a[0] = %f\n", matrix_a[supernode_offset]);

    memcpy_to_device( &matrix_ia[supernode_fc],     ia_dev, (cols+1)*sizeof(size_t),       stream );
    memcpy_to_device( &matrix_ja[supernode_offset], ja_dev,  supernode_nnz*sizeof(size_t), stream );
    memcpy_to_device( &matrix_a[supernode_offset],  a_dev,   supernode_nnz*sizeof(T),      stream );
        
    init_supernode_dev(M_dev, ia_dev, ja_dev, a_dev, supernode, supernode_nnz, supernode_offset, matrix_ns, matrix_nt, matrix_nd, stream );

    //double a_host;
    //gpuErrchk(cudaMemcpy(&a_host, &M_dev[0], sizeof(double), cudaMemcpyDeviceToHost)); 
    //printf("in init supernode. IB = %d, a[0] = %f\n", supernode, a_host);

}


/************************************************************************************************/

// M_dev is local, all entries of 1 supernode
// a_dev is local -> all nnz entries of a in that supernode
template <class T>
inline void RGF<T>::extract_nnzA(T *M_dev, size_t supernode)
{
   // only works because only the last set of columns is not of size matrix_ns ...
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t supernode_nnz = matrix_ia[supernode_lc] - matrix_ia[supernode_fc];
   size_t supernode_offset = matrix_ia[supernode_fc];
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;
   size_t supernode_size = rows * cols;

   // check that supernode_nnz is smaller than max_supernode_nnz because length(inv_a_dev/inv_ja_dev) = max_supernode_nnz
   if(supernode_nnz > max_supernode_nnz){
      printf("Memory Allocation problem! supernode_nnz: %ld > max_supernode_nnz: %ld\n", supernode_nnz, max_supernode_nnz);
      exit(1);
   }

   gpuErrchk(cudaMemcpy(inv_ia_dev, &matrix_ia[supernode_fc], (cols+1)*sizeof(size_t), cudaMemcpyHostToDevice ));
   gpuErrchk(cudaMemcpy(inv_ja_dev, &matrix_ja[supernode_offset], supernode_nnz*sizeof(size_t), cudaMemcpyHostToDevice ));

   //memcpy_to_device(&matrix_ia[supernode_fc],inv_ia_dev,(cols+1)*sizeof(size_t));
   //memcpy_to_device(&matrix_ja[supernode_offset],inv_ja_dev,supernode_nnz*sizeof(size_t));
   //memcpy_to_device(&matrix_a[supernode_offset],a_dev,supernode_nnz*sizeof(T));

   /*
   T* M_host;
   M_host = new T[supernode_size];

   cudaMemcpy(M_host, M_dev, supernode_size*sizeof(T), cudaMemcpyDeviceToHost );
  
   printf("IB = %ld, supernode nnz = %ld, supernode size = %ld, M host : ", supernode, supernode_nnz, supernode_size);
   for(int i=0; i<supernode_size; i++){
      printf(" %f", M_host[i]);
   }
   printf("\n");

   delete[] M_host;

   size_t* inv_ia_host;
   inv_ia_host = new size_t[supernode_nnz];

   //memcpy_to_host(a_dev, a_host, supernode_nnz*sizeof(T));
   cudaMemcpy(inv_ia_host, inv_ia_dev, (cols+1)*sizeof(size_t), cudaMemcpyDeviceToHost );
   //memcpy_to_host(a_dev, &invQ[supernode_offset], supernode_nnz*sizeof(T));

   printf("ia host : ");
   for(int i=0; i<cols+1; i++){
      printf(" %ld", inv_ia_host[i]);
   }
   printf("\n");
   */

   extract_nnzA_dev(inv_a_dev, inv_ia_dev, inv_ja_dev, M_dev, supernode, supernode_nnz, supernode_offset, matrix_ns, matrix_nt, matrix_nd);

   //exit(1);
   //printf("supernode offset : %ld, supernode nnz : %ld\n", supernode_offset, supernode_nnz);

   gpuErrchk(cudaMemcpy(&inv_a[supernode_offset], inv_a_dev, supernode_nnz*sizeof(T), cudaMemcpyDeviceToHost));

   /*
   printf("inv_a : ");
   for(int i=0; i<supernode_nnz; i++){
      //inv_a[supernode_offset+i] = inv_a_host[i];
      printf(" %f", inv_a[supernode_offset+i]);
   }
   printf("\n");
   cudaDeviceSynchronize();   
   */
}

/************************************************************************************************/

template <class T>
inline void RGF<T>::copy_supernode_to_host(T *M_dev, size_t supernode, cudaStream_t stream)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t ind = mf_block_index(supernode, supernode);
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;

#ifdef PRINT_MSG
   printf("In copy supernode to host.\n");
   printf("size rows = %ld, size cols = %ld, size rows*cols = %ld, ind = %ld\n", rows, cols, rows*cols, ind);
#endif

    memcpy_to_host(&MF[ind], M_dev, rows*cols*sizeof(T), stream );

}


/************************************************************************************************/

template <class T>
inline void RGF<T>::copy_supernode_to_host_write(T *M_dev, size_t supernode)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t ind = mf_block_index(supernode, supernode);
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;

#ifdef PRINT_MSG
   printf("In copy supernode to host.\n");
   printf("size rows&cols = %ld, ind = %ld\n", rows*cols, ind);
#endif

   memcpy_to_host(&MF[ind], M_dev, rows*cols*sizeof(T), NULL);

  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,sizeof(buffer),"chunk_of_L_%d-%m-%Y_%H:%M:%S.txt",timeinfo);
  std::string file_name(buffer);

   //std::string file_name = "chunk_of_L.txt";
   ofstream file(file_name,    ios::out | ::ios::trunc); 
   for(int i=0; i<rows*cols; i++){
      //file << std::setprecision(15) << MF[ind+i] << endl;
      file << MF[ind+i] << endl;

   }
   file.close();
   std::cout << "wrote to file : " << file_name << std::endl;

}


/************************************************************************************************/

template <class T>
inline void RGF<T>::copy_supernode_to_device(T *M_dev, size_t supernode, cudaStream_t stream)
{
   size_t supernode_fc = supernode * matrix_ns;
   size_t supernode_lc = supernode < matrix_nt ? (supernode+1) * matrix_ns : matrix_ns * matrix_nt + matrix_nd;
   size_t ind = mf_block_index(supernode, supernode);
   size_t rows = mf_block_lda(supernode, supernode);
   size_t cols = supernode_lc - supernode_fc;

    memcpy_to_device(&MF[ind], M_dev, rows*cols*sizeof(T), stream);

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

// added inline here and in the next 3 functions to avoid compiler issues! 
template <>
inline CPX RGF<CPX>::f_one()
{
    return CPX(1.0, 0.0);
}

template <>
inline double RGF<double>::f_one()
{
    return 1.0;
}

/************************************************************************************************/

template <>
inline CPX RGF<CPX>::f_zero()
{
    return CPX(0.0, 0.0);
}

template <>
inline double RGF<double>::f_zero()
{
    return 0.0;
}

/************************************************************************************************/

#endif

