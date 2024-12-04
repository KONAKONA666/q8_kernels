#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <vector>

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

#include "kernel_traits.h"
#include "utils.h"
#include "reg2reg.h"

#include "sm89_mma.hpp"
#include "mma_sm89_traits.hpp"



namespace flash {

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}


template <bool Is_even_MN=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_qk(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {  
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        } 
    }
};

template <bool Is_even_MN=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_v(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {  
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if constexpr (Is_even_MN){
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else {
                    if (get<1>(identity_MN(0, m, k)) < max_MN){
                        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                    } else {
                        clear(D(_, m, k));
                    }
                //     } else if(get<1>(identity_MN(size<0>(identity_MN)-1, m, k)) >= max_MN) {
                //         size_t i = 0;
                //         while(get<1>(identity_MN(i, m, k)) < max_MN ){
                //             D(i, m, k) = S(i, m, k);
                //             i++;
                //         }
                //         for(;i<size<0>(identity_MN);i++){
                //             clear(D(i, m, _));
                //         }
                //     }
                } 
            }
    } 
};

template <typename ToType, typename Fragment>
inline __device__ auto convert_float32_to_fp8(Fragment const &acc_fp32) {
  Tensor acc_fp8 = make_tensor<ToType>(shape(acc_fp32));
  using convert_type = std::conditional_t<
                            std::is_same_v<ToType, cutlass::float_e5m2_t>,
                            __nv_fp8x2_e5m2,
                            __nv_fp8x2_e4m3
                        >;
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp8x2 = recast<convert_type>(acc_fp8);
    for (int i = 0; i < size(acc_fp32x2); ++i) { 
      acc_fp8x2(i) = convert_type(acc_fp32x2(i)); 
    }
  }
  return acc_fp8;
}


// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float softmax_scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    // constexpr float sm_scale = softmax_scale*1.44269504f;
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.0f : -max(mi) * softmax_scale;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) =  exp2f(tensor(mi, ni) * softmax_scale + max_scaled);        
        }
    }
}


template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};


template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, const float softmax_scale) {
    // float softmax_scale_log2 = 1.44269504f;
    if (Is_first) {
        reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale);
        reduce_sum<true>(scores, scores_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        reduce_max</*zero_init=*/false>(scores, scores_max); 
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) { 
            float scores_max_cur = scores_max(mi); 
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur)*softmax_scale); 
            scores_sum(mi) *= scores_scale;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; } 
        } 
        flash::scale_apply_exp2(scores, scores_max, softmax_scale);
        reduce_sum<false>(scores, scores_sum);         
    }
};


 template <typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                            const int col_idx_offset_,
                                            const int max_seqlen_k) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    
    Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
    
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -1000000.0f; }
            }
        }
    }
};


} 

template<bool Is_Even=true>
__global__ void flash_attention_v2_cutlass_mask_kernel(
    float_e4m3_t* Q_ptr, float_e4m3_t* K_ptr, float_e4m3_t* V_ptr, int* BatchMask,
    float_e4m3_t* O_ptr, 
    size_t BATCH, size_t HEADS, size_t M, size_t N, float softmax_scale
) {

    using namespace cute;


    const int m_block = blockIdx.x;

    const int base_id = blockIdx.y;
    const int tidx = threadIdx.x;

   
    using Element = float_e4m3_t;
    using ElementO = float_e4m3_t;

    using ElementAccum = float;
    using index_t = uint32_t;

    constexpr int kHeadDim = 64;
    constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    constexpr int kNWarps = 4;
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    
    constexpr int M_MMA = kNWarps * 16;

    using TiledMMA = TiledMMA<MMA_Atom<SM89_16x8x32_F32F8F8F32_E4M3_TN>, Layout<Shape<Int<kNWarps>, _1, _1>>, Tile<Int<M_MMA>, _16, _32>>;

    constexpr int kNThreads = kNWarps * 32;

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_16, Int<kBlockKSmem>>, // (16, 64)
                            Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));
    
    extern __shared__ float_e4m3_t smem[];
    
    float_e4m3_t* q_sptr = smem;
    float_e4m3_t* k_sptr = q_sptr + cosize(SmemLayoutQ{});
    float_e4m3_t* v_sptr = k_sptr + cosize(SmemLayoutK{});
    
    int V_N =  cute::ceil_div(N, 16)*16;

    const int bs_head_offset_q = base_id * kHeadDim * M;
    const int bs_head_offset_k = base_id * kHeadDim * N;
    const int bs_head_offset_v = base_id * kHeadDim * V_N;

    const int MASK_N = BatchMask[base_id / (int)HEADS];    
   
    Tensor Q = make_tensor(
        make_gmem_ptr(Q_ptr + bs_head_offset_q),
        make_shape(M, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor K = make_tensor(
        make_gmem_ptr(K_ptr + bs_head_offset_k),
        make_shape(N, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor V = make_tensor(
        make_gmem_ptr(V_ptr + bs_head_offset_v),
        make_shape(Int<kHeadDim>{}, V_N),
        make_stride(V_N, Int<1>{}));
    Tensor O = make_tensor(
        make_gmem_ptr(O_ptr + bs_head_offset_q),
        make_shape(M, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));

    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _0{})); 
    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
    Tensor gV = local_tile(V, make_tile(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _)); 

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    Tensor sQ = make_tensor(make_smem_ptr(q_sptr), SmemLayoutQ{}); //(kBlockM, kHeadDim)
    Tensor sK = make_tensor(make_smem_ptr(k_sptr), SmemLayoutK{}); //(kBlockN, kHeadDim)
    Tensor sV = make_tensor(make_smem_ptr(v_sptr), SmemLayoutV{}); //(kHeadDim, kBlockN)

    constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(float_e4m3_t);
    constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float_e4m3_t>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  
    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, _));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, _));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
   
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K) =  ((4, 2, 2), MMA_M, MMA_N)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K) =  ((4, 2), MMA_N, MMA_K)
    Tensor tOrV  = thr_mma.partition_fragment_B(sV);                           // (MMA,MMA_K,MMA_N) =  ((2, 2), MMA_M, MMA_N)
    
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, float_e4m3_t>;

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ); // tSsQ --> tSrQ

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK); // tSrK --> tSsK


    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsV = smem_thr_copy_V.partition_S(sV);
   
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV))); // (HEAD, BLK_N)

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK = gmem_thr_copy_QKV.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_QKV.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(N, kBlockN); 
    int n_block = n_block_max - 1;

    
    flash::copy_qk<Is_Even>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, M - m_block*kBlockM);
    flash::copy_qk<Is_Even>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKcK, N - n_block * kBlockN);

    cute::cp_async_fence();

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{}); 

    // NOTE: K, V分块的数量: 处理的区间
    
    Tensor scores_max = make_tensor<float>(Shape<Int<2 * size<1>(rAccOut)>>{});
    Tensor scores_sum = make_fragment_like(scores_max); 
    clear(rAccOut); 

    constexpr size_t n_masking_steps = 1;
    #pragma unroll
    for(size_t masking_step = 0; masking_step < n_masking_steps; masking_step++, --n_block){
        auto rAccScore = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(rAccScore); 

        flash::cp_async_wait<0>();
        __syncthreads();

        flash::copy_v<Is_Even>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tVcV, N - n_block * kBlockN);
        cute::cp_async_fence();

        flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        flash::apply_mask(rAccScore, n_block * kBlockN, MASK_N);
        Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));
        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy_qk<true>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block-1), tKsK, tKcK);           
            cute::cp_async_fence();
        }
        flash::softmax_rescale_o<true>(scores, scores_max, scores_sum, rAccOut, softmax_scale);
        Tensor rP = flash::convert_float32_to_fp8<float_e4m3_t>(rAccScore);
        auto reg2reg = ReorgCFp8toAFp8();
        reg2reg(rP);
       
        auto tOrPLayout = Layout<Shape<Shape<_4, _2, _2>, Int<size<1>(rP)>, Int<size<2>(tOrV)>>>{};
        Tensor tOrP = make_tensor(rP.data(), tOrPLayout);
        flash::gemm_A_in_regs(rAccOut, tOrP, tOrV, tOsV, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    #pragma unroll
    for (;n_block>=n_block_min; n_block--) {
        auto rAccScore = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(rAccScore); 

        flash::cp_async_wait<0>();
        __syncthreads();
    
        flash::copy_v<true>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tVcV);
        cute::cp_async_fence();
        
        flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        flash::apply_mask(rAccScore, n_block * kBlockN, MASK_N);
        Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy_qk<true>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKcK);
            cute::cp_async_fence();
        }

        flash::softmax_rescale_o<false>(scores, scores_max, scores_sum, rAccOut, softmax_scale);
        
        Tensor rP = flash::convert_float32_to_fp8<float_e4m3_t>(rAccScore);
        auto reg2reg = ReorgCFp8toAFp8();
        reg2reg(rP);
       
        auto tOrPLayout = Layout<Shape<Shape<_4, _2, _2>, Int<size<1>(rP)>, Int<size<2>(tOrV)>>>{};
        Tensor tOrP = make_tensor(rP.data(), tOrPLayout);
        flash::gemm_A_in_regs(rAccOut, tOrP, tOrV, tOsV, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    SumOp<float> sum_op;
    quad_allreduce_(scores_sum, scores_sum, sum_op);

    Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
    
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }

    Tensor rO = flash::convert_float32_to_fp8<float_e4m3_t>(rAccOut);
    using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint16_t>, float_e4m3_t>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float_e4m3_t>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    Tensor sO = make_tensor((float_e4m3_t*)q_sptr, SmemLayoutO{});   

    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);   

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    
    Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _0{}));

    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);       
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_S(cO);   
    __syncthreads();
    
    flash::copy_qk<Is_Even>(gmem_tiled_copy_O, tOsO, tOgO, tOcO, M - m_block*kBlockM);
}


void flash_attention_cuda_mask(void* Q_ptr, void* K_ptr, void* V_ptr, int* BatchMask,
                   void* O_ptr, 
                   size_t BATCH, size_t M, size_t N, size_t NUM_HEADS, float softmax_scale, cudaStream_t stream){

        int threads = 4*32;
        int M_BLOCK = 64;
        int N_BLOCK = 64;
        
        const int num_m_block = (M + M_BLOCK - 1) / M_BLOCK;
        dim3 grid(num_m_block, BATCH * NUM_HEADS, 1);
        dim3 block(threads);
        if(M % M_BLOCK == 0 && N % N_BLOCK == 0 ){
            auto kernel = &flash_attention_v2_cutlass_mask_kernel<true>;
            int smem_size = int(N_BLOCK*64*2 + M_BLOCK*64);
            if (smem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            }
            kernel<<<grid, block, smem_size, stream>>>((float_e4m3_t*)Q_ptr, (float_e4m3_t*)K_ptr, (float_e4m3_t*)V_ptr, BatchMask, (float_e4m3_t*)O_ptr, BATCH, NUM_HEADS, M, N, softmax_scale);
            // C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            auto kernel = &flash_attention_v2_cutlass_mask_kernel<false>;
            int smem_size = int(N_BLOCK*64*2 + M_BLOCK*64);
            if (smem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            }
            kernel<<<grid, block, smem_size, stream>>>((float_e4m3_t*)Q_ptr, (float_e4m3_t*)K_ptr, (float_e4m3_t*)V_ptr, BatchMask, (float_e4m3_t*)O_ptr, BATCH, NUM_HEADS, M, N, softmax_scale);
            // C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        
}


void flash_attention_cuda_mask(void* Q_ptr, void* K_ptr, void* V_ptr, int* BatchMask,
                   void* O_ptr, 
                   size_t BATCH, size_t M, size_t N, size_t NUM_HEADS, float softmax_scale, cudaStream_t stream);