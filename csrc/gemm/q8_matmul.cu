#include <torch/extension.h>
#include <torch/python.h>
#include "q8_gemm_api.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

#include "static_switch.h"

using namespace cute;

inline __device__ float gelu_approximate(float x){
    constexpr float sqrthalfpi2 = 0.7978845608028653558798921198687637369517172623298693153318516593f;
    constexpr float factor = 0.044715f;
    return 0.5f*x*(1.0f + tanhf(sqrthalfpi2*(x + factor*x*x*x)));
}

inline __device__ float gelu_erf(float x){
    return x*0.5f*(1.0f +erff(x * 0.707106781));
    // return 0.5f*x*(1.0f + tanhf(sqrthalfpi2*(x + factor*x*x*x)));
}



template<typename Fragment>
inline __device__ auto convert_fp32_fp8(Fragment const& fp32_fragment){
    Tensor fp8_fragment = make_tensor<float_e4m3_t>(shape(fp32_fragment));
    Tensor fp32x2_fragment = recast<float2>(fp32_fragment);
    Tensor fp8x2_fragment = recast<__nv_fp8x2_e4m3>(fp8_fragment);
    for(int i = 0; i<size(fp32x2_fragment);i++){
        uint16_t RC_FP8 = __nv_cvt_float2_to_fp8x2(fp32x2_fragment(i), __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        fp8x2_fragment(i) = *reinterpret_cast<__nv_fp8x2_e4m3 *>(&RC_FP8);
    }
    return fp8_fragment;
}


template<bool Is_Even, bool fuse_gelu_activation, int BM, int BN, int BK, bool BATCH_A, bool BATCH_B, int KStages,
        typename TiledMMA, typename G2SCopyA, typename G2SCopyB,
        typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,

        typename SmemLayoutScaleA, typename SmemLayoutScaleB,

        typename LayoutScaleAT, typename LayoutScaleBT,
        typename LayoutScaleAV_S, typename LayoutScaleBV_S,
        typename LayoutScaleAV_D, typename LayoutScaleBV_D,
        typename LayoutScaleAV_DView, typename LayoutScaleBV_DView,
        typename G2SScaleCopyA, typename G2SScaleCopyB, 

        typename AScaleTV, typename I2TVAScale,
        typename BScaleTV, typename I2TVBScale,
        

        typename S2RCopyAtomA, typename S2RCopyAtomB,
        typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void gemm_q8_kernel(const int8_t * Aptr, const int8_t * Bptr, 
                                 const float* A_scales, const float* B_scales,
                                 float_e4m3_t*  Cptr, 
                                 int M,
                                 int N, int K, 
                                 int BATCH){

    extern __shared__ float shm_data[];

    float *Ascale_shm = shm_data;
    float *BScale_shm = shm_data + cosize(SmemLayoutScaleA{});
    int8_t *Ashm = (int8_t*)(shm_data + cosize(SmemLayoutScaleA{}) + cosize(SmemLayoutScaleA{}));
    float_e4m3_t* Cshm = (float_e4m3_t*)(shm_data + cosize(SmemLayoutScaleA{}) + cosize(SmemLayoutScaleA{}));
    int8_t *Bshm = (int8_t*)(Ashm + cosize(SmemLayoutA{}));

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;

    int batch_a_offset = BATCH_A ? iz * M*K : 0;
    int batch_b_offset = BATCH_B ? iz * N*K : 0;
    int batch_c_offset = iz * M*N;

    int batch_ascales_offset = BATCH_A ? iz * M : 0;
    int batch_bscales_offset = BATCH_B ? iz * N : 0;
    
    Tensor A = make_tensor(make_gmem_ptr(Aptr + batch_a_offset), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr + batch_b_offset), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Cptr + batch_c_offset), make_shape(M, N), make_stride(N, Int<1>{}));

    Tensor AScales = make_tensor(make_gmem_ptr(A_scales + batch_ascales_offset), make_shape(_1{}, M), make_stride(M, Int<1>{}));
    Tensor BScales = make_tensor(make_gmem_ptr(B_scales + batch_bscales_offset), make_shape(_1{}, N), make_stride(N, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _)); 
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _)); 
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix)); 

    Tensor gAscales = local_tile(AScales, make_tile(Int<1>{}, Int<BM>{}), make_coord(_, iy));
    Tensor gBscales = local_tile(BScales, make_tile(Int<1>{}, Int<BN>{}), make_coord(_, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); 
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});
      
    auto sAscales = make_tensor(make_smem_ptr(Ascale_shm), SmemLayoutScaleA{});
    auto sBscales = make_tensor(make_smem_ptr(BScale_shm), SmemLayoutScaleB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); 
    auto tCrD = thr_mma.partition_fragment_C(gD);           
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); 
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); 

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); 
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); 

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); 


    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);    
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    G2SScaleCopyA g2s_scale_copy_a;
    G2SScaleCopyB g2s_scale_copy_b;
    
    auto g2s_scale_thr_copy_a = g2s_scale_copy_a.get_slice(idx);
    auto g2s_scale_thr_copy_b = g2s_scale_copy_b.get_slice(idx);

    auto tAScalegAScale = g2s_scale_thr_copy_a.partition_S(gAscales);
    auto tAScalesAScale = g2s_scale_thr_copy_a.partition_D(sAscales);

    auto tBScalegBScale = g2s_scale_thr_copy_b.partition_S(gBscales);
    auto tBScalesBScale = g2s_scale_thr_copy_b.partition_D(sBscales);
    
    auto cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    auto tAcA = g2s_thr_copy_a.partition_S(cA);
    
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    
    int residual = M - iy*BM;

    auto tAscalepAscale = make_tensor<bool>(_1{});
    tAscalepAscale(0) = idx < residual;

#pragma unroll
    for (int istage = 0; istage < KStages - 1; ++istage) {
        if constexpr (Is_Even){
            if(istage == 0){
                cute::copy(g2s_scale_copy_a, tAScalegAScale(_, _, _, _0{}), tAScalesAScale);
                cute::copy(g2s_scale_copy_b, tBScalegBScale(_, _, _, _0{}), tBScalesBScale); 
            }

            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                        tAsA_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                        tBsB_copy(_, _, _, istage));
            cp_async_fence();

            ++itile_to_read;
            ++ismem_write;
        } else {
            if(istage == 0){
                cute::copy_if(g2s_scale_copy_a, tAscalepAscale, tAScalegAScale(_, _, _, _0{}), tAScalesAScale);
                cute::copy(g2s_scale_copy_b, tBScalegBScale(_, _, _, _0{}), tBScalesBScale); 
            }
            for (size_t m = 0; m < size<1>(tAsA_copy); m++)
            {
                for (size_t k = 0; k < size<2>(tAsA_copy); k++)
                {
                    if(get<0>(tAcA(0, m, k)) < residual){
                        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, m, k, istage),
                            tAsA_copy(_, m, k, istage));
                    }
                }  
            }
            
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                        tBsB_copy(_, _, _, istage));
            cp_async_fence();

            ++itile_to_read;
            ++ismem_write;
        }

    }

     cp_async_wait<KStages - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik)); 
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    int ntile = K / BK;

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA); // (MMA, MMA_M, MMA_K)

        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;
            
            if (ik == nk - 1) {
                cp_async_wait<KStages - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % KStages;
            }
            

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), // tAsA: (CPY, CPY_M, CPY_K, kStage)
                    tCrA_view(_, _, ik_next));                            // tCrA_view: (CPY, CPY_M, CPY_K)
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), // tBsB: (CPY, CPY_M, CPY_K, kStage)
                    tCrB_view(_, _, ik_next));                            // tCrB_view: (CPY, CPY_M, CPY_K)

            
            if (ik == 0) {
                if (itile_to_read < ntile) {
                    if constexpr (Is_Even){
                        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                            tAsA_copy(_, _, _, ismem_write));
                    
                    } else {
                        for (size_t m = 0; m < size<1>(tAsA_copy); m++)
                        {
                            for (size_t k = 0; k < size<2>(tAsA_copy); k++)
                            {
                                if(get<0>(tAcA(0, m, k)) < residual){
                                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, m, k, itile_to_read),
                                        tAsA_copy(_, m, k, ismem_write));
                                }
                            }  
                        }
                    }
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                            tBsB_copy(_, _, _, ismem_write));

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % KStages;
                }

                cp_async_fence();
            }
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }  // for ik
    }
    __syncthreads();
   
    auto tCrD_fp32 = make_tensor_like<float>(tCrD);
    for (size_t i = 0; i < size(tCrD_fp32); i++)
    {
        tCrD_fp32(i) = static_cast<float>(tCrD(i));
    }
    
    
    auto tCrAscales_copy = make_tensor<float>(LayoutScaleAV_D{});
    auto tCrBscales_copy = make_tensor<float>(LayoutScaleBV_D{});
    
    auto tCrAscales = make_tensor(tCrAscales_copy.data(), LayoutScaleAV_DView{});
    auto tCrBscales = make_tensor(tCrBscales_copy.data(), LayoutScaleBV_DView{});
  

    auto a_scales_tv = AScaleTV{};
    auto b_scales_tv = BScaleTV{};
    
    

    #pragma unroll
    for (size_t mma_m = 0; mma_m < size<1>(tCrD_fp32); mma_m++)
    {
        tCrAscales(0, mma_m, 0) = sAscales(a_scales_tv(idx, make_coord(0, mma_m)));
        tCrAscales(2, mma_m, 0) = sAscales(a_scales_tv(idx, make_coord(1, mma_m)));
    }
    
    #pragma unroll
    for (size_t mma_n = 0; mma_n < size<2>(tCrD_fp32); mma_n++)
    {
        tCrBscales(0, mma_n, 0) = sBscales(b_scales_tv(idx, make_coord(0, mma_n)));
        tCrBscales(1, mma_n, 0) = sBscales(b_scales_tv(idx, make_coord(1, mma_n)));
    }
    

    for (size_t mma_m = 0; mma_m < size<1>(tCrD_fp32); mma_m++)
    {
        for (size_t mma_n = 0; mma_n < size<2>(tCrD_fp32); mma_n++)
        {   
            if constexpr (fuse_gelu_activation){
                tCrD_fp32(_0{}, mma_m, mma_n) = gelu_approximate(tCrD_fp32(_0{}, mma_m, mma_n) * (tCrAscales(_0{}, mma_m, _0{}) * tCrBscales(_0{}, mma_n, _0{})));
                tCrD_fp32(_1{}, mma_m, mma_n) = gelu_approximate(tCrD_fp32(_1{}, mma_m, mma_n) * (tCrAscales(_1{}, mma_m, _0{}) * tCrBscales(_1{}, mma_n, _0{})));
                tCrD_fp32(_2{}, mma_m, mma_n) = gelu_approximate(tCrD_fp32(_2{}, mma_m, mma_n) * (tCrAscales(_2{}, mma_m, _0{}) * tCrBscales(_2{}, mma_n, _0{})));
                tCrD_fp32(_3{}, mma_m, mma_n) = gelu_approximate(tCrD_fp32(_3{}, mma_m, mma_n) * (tCrAscales(_3{}, mma_m, _0{}) * tCrBscales(_3{}, mma_n, _0{})));
            } else {
                tCrD_fp32(_0{}, mma_m, mma_n) = tCrD_fp32(_0{}, mma_m, mma_n) * (tCrAscales(_0{}, mma_m, _0{}) * tCrBscales(_0{}, mma_n, _0{}));
                tCrD_fp32(_1{}, mma_m, mma_n) = tCrD_fp32(_1{}, mma_m, mma_n) * (tCrAscales(_1{}, mma_m, _0{}) * tCrBscales(_1{}, mma_n, _0{}));
                tCrD_fp32(_2{}, mma_m, mma_n) = tCrD_fp32(_2{}, mma_m, mma_n) * (tCrAscales(_2{}, mma_m, _0{}) * tCrBscales(_2{}, mma_n, _0{}));
                tCrD_fp32(_3{}, mma_m, mma_n) = tCrD_fp32(_3{}, mma_m, mma_n) * (tCrAscales(_3{}, mma_m, _0{}) * tCrBscales(_3{}, mma_n, _0{}));
            }
           
        }
    }

    auto sC = make_tensor(make_smem_ptr(Cshm), SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD_fp32);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); 

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

    constexpr int repeat_k = size<2>(tCsC_r2s);
    constexpr int global_k = size<2>(tCrC_r2s)/size<2>(tCsC_r2s);   

    auto cC = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));
    auto tCcC = s2g_thr_copy_c.partition_D(cC);

    for (size_t i = 0; i < size<1>(tCrC_r2s); i++)
    {
        for (size_t k = 0; k < global_k; k++)
        {
            for (size_t t = 0; t < repeat_k; t++)
            {   
                auto fp8_fragment = convert_fp32_fp8(tCrC_r2s(_, i, t+k*repeat_k));                
                cute::copy(r2s_tiled_copy_c, fp8_fragment, tCsC_r2s(_, 0, t));
            }
            __syncthreads();
            if constexpr (Is_Even){
                cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0), tCgC_s2g(_, i, k));
            } else {
                if(get<0>(tCcC(0, i, k)) < residual){
                    cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0), tCgC_s2g(_, i, k));        
                }
            }
            __syncthreads();
        }
    }
}

void run_q8_gemm(int8_t *A, int8_t *B, void *C, float* A_scales, float* B_scales, int BA, int BB, int M, int N, int K, bool fuse_gelu, cudaStream_t stream){
    
    int BATCH;
    TORCH_CHECK(BB == BB || (BA == 1 || BB==1) , "Batch size missmatch");

    if (BA == 1 || BB == 1){
        BATCH = BA * BB;
    } else if(BB == BA){
        BATCH = BA;
    }
    auto BM = Int<128>{};
    auto BN = Int<128>{};
    auto BK = Int<64>{}; // MMA_K=32, 2 CHUNKS
    auto KStages = Int<2>{};

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 4, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));

    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStages>{}))
    );
    
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStages>{}))
    );
    
    using mma_op = SM80_16x8x32_S32S8S8S32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int WARP_ROWS = 2;
    static constexpr int WARP_COLS = 2;

    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int MMA_WARP_M = WARP_ROWS * get<0>(mma_atom_shape{});
    static constexpr int MMA_WARP_N = 1 * WARP_COLS * get<1>(mma_atom_shape{});
    static constexpr int MMA_WARP_K = 1 * get<2>(mma_atom_shape{});
    
    using MMA_WARP_Tile = decltype(
        make_layout(make_shape(Int<WARP_ROWS>{}, Int<WARP_COLS>{}, Int<1>{}))
    );
    using MMA_PARTITION_TILE = Tile<Int<MMA_WARP_M>, Int<MMA_WARP_N>, Int<MMA_WARP_K>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_WARP_Tile{}, MMA_PARTITION_TILE{}));

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, int8_t>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), 
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<16>{})))); 
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op_a = SM75_U32x4_LDSM_N;
    using s2r_copy_traits_a = Copy_Traits<s2r_copy_op_a>;
    using s2r_copy_atom_a = Copy_Atom<s2r_copy_traits_a, int8_t>;

    using s2r_copy_op_b = SM75_U32x2_LDSM_N;
    using s2r_copy_traits_b = Copy_Traits<s2r_copy_op_b>;
    using s2r_copy_atom_b = Copy_Atom<s2r_copy_traits_b, int8_t>;

    using SmemLayoutC = decltype(
        composition(
                    Swizzle<2, 4, 3>{}, 
                    make_layout(make_shape(Int<MMA_WARP_M>{}, Int<MMA_WARP_N*Int<4>{}>{}), 
                    make_stride(Int<MMA_WARP_N*Int<4>{}>{}, Int<1>{})))
    );

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<cute::uint16_t>, float_e4m3_t>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, float_e4m3_t>;

    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<16>{}))));
    
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = BATCH;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, BZ);

    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize =
        cute::max(shm_size_AB, shm_size_C) * sizeof(cute::float_e4m3_t);

    int shm_size = kShmSize;
    
     
    using G2SScales_copy_op = SM80_CP_ASYNC_CACHEALWAYS<float>;
    using G2SScales_copy_traits = Copy_Traits<G2SScales_copy_op>;
    using G2SScales_copy_atom = Copy_Atom<G2SScales_copy_traits, float>;
    
    using SmemLayoutScaleA = Layout<Shape<Int<1>, Int<BM>>, Stride<Int<BM>, Int<1>>>;
    using SmemLayoutScaleB = Layout<Shape<Int<1>, Int<BN>>, Stride<Int<BN>, Int<1>>>;

    using G2SScalesCopyA = decltype(make_tiled_copy(G2SScales_copy_atom{}, make_layout(
                                                                        make_shape(Int<1>{},Int<BM>{}), make_stride(Int<BM>{}, Int<1>{})),
                                                                        make_layout(make_shape(Int<1>{},Int<1>{}), make_stride(Int<1>{}, Int<1>{}))));
    
    using G2SScalesCopyB = decltype(make_tiled_copy(G2SScales_copy_atom{}, make_layout(
                                                                        make_shape(Int<1>{},Int<BN>{}), make_stride(Int<BN>{}, Int<1>{})),
                                                                        make_layout(make_shape(Int<1>{},Int<1>{}), make_stride(Int<1>{}, Int<1>{}))));
    

    using AScale_TLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<_0, _8>, Stride<_64, _0>>>;
    using AScale_VLayoutS = Layout<Shape<_4, _2, _1, decltype(KStages)>, Stride<_1, _4, _128, _256>>;
    using AScale_VLayoutD = Layout<Shape<_4, _2, _1>, Stride<_1, _4, _8>>;
    using AScale_VLayoutD_View = Layout<Shape<Shape<_2, _2>, _4, _1>, Stride<Stride<_0, _1>, _2, _8>>;
    
    using AScaleTV = Layout<Shape<
                                Shape<
                                    Shape<_4, _8>, Shape<_2, _2>,
                                >, 
                                Shape<_2, _4>
                            >, 
                            Stride<
                                Stride<Stride<_0, _1>, Stride<_16, _0>>, 
                                Stride<_8, _32>
                            >>;
    using I2TVAScale = decltype(right_inverse(AScaleTV{}));

    auto BN_elems =  _2{}*BN / _16{};
    auto MMA_N = BN / _16{};
    auto CPY_N = MMA_N / _2{};

    using BScale_TLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<decltype(BN_elems), _0>, Stride<_0, _32>>>;
    using BScale_VLayoutS = Layout<Shape<_4, decltype(CPY_N), _1, decltype(KStages)>, Stride<_1, _4, _96, _96>>;
    using BScale_VLayoutD = Layout<Shape<_4, decltype(CPY_N), _1>, Stride<_1, _4, _16>>;
    using BScale_VLayoutD_view = Layout<Shape<Shape<_2, _2>, decltype(MMA_N), _1>, Stride<Stride<_1, _0>, _2, _8>>;

    using BScaleTV = Layout<Shape<
                                Shape<
                                    Shape<_4, _8>, Shape<_2, _2>,
                                >, 
                                Shape<_2, _8>
                            >, 
                            Stride<
                                Stride<Stride<_2, _0>, Stride<_0, _8>>, 
                                Stride<_1, _16>
                            >>;
    using I2TVBScale = decltype(right_inverse(BScaleTV{}));

    bool is_even = M % BM == 0;
    BOOL_SWITCH(fuse_gelu, fuse_gelu_, 
        BOOL_SWITCH(is_even, Is_Even,
            BATCH_SWITCH(BA, BB, auto kernel = &gemm_q8_kernel<Is_Even, fuse_gelu_, BM, BN, BK, BA_, BB_, KStages, MMA,
                                                G2SCopyA, G2SCopyB, 
                                                SmemLayoutA, SmemLayoutB, SmemLayoutC, 

                                                SmemLayoutScaleA, SmemLayoutScaleA,  
                                                AScale_TLayout, BScale_TLayout,
                                                AScale_VLayoutS, BScale_VLayoutS,
                                                AScale_VLayoutD, BScale_VLayoutD, 
                                                AScale_VLayoutD_View, BScale_VLayoutD_view,
                                                G2SScalesCopyA,G2SScalesCopyB, 

                                                AScaleTV, I2TVAScale,
                                                BScaleTV, I2TVBScale,

                                                s2r_copy_atom_a, s2r_copy_atom_b, 
                                                R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>;
                cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
                kernel<<<grid, block, shm_size, stream>>>((int8_t*)A, (int8_t*)B, A_scales, B_scales, (float_e4m3_t*)C, M, N, K, BATCH););
        );
    );
}