#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>

#include <cute/tensor.hpp>

#include "rms_norm.h"

template<int kNThreads_, int dim, bool norm_affine_, typename input_t_, typename output_t_>
struct rmsnorm_kernel_traits {

    using input_t = input_t_;
    using output_t = output_t_;
    using weights_t = float;
    using vec_t = uint4;
    
    static constexpr bool norm_affine = norm_affine_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNBytes_input = sizeof(input_t);
    static_assert(kNBytes_input == 1 || kNBytes_input == 2 || kNBytes_input == 4);
    static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
    static_assert(ThreadElems * kNThreads == dim);
};


template <int NElems, typename input_t, typename vec_t>
inline __device__ void load_input(input_t *x, float x_vals[NElems]) {
    input_t x_vals_load[NElems] = {0};
    constexpr int num_elems_per_load = sizeof(vec_t)/sizeof(input_t);
    constexpr int num_chunks = NElems/num_elems_per_load;
    
    #pragma unroll
    for (size_t i = 0; i < num_chunks; i++)
    {
        reinterpret_cast<vec_t*>(x_vals_load)[i] = reinterpret_cast<const vec_t*>(x)[num_chunks*threadIdx.x+i];
    }
    
    #pragma unroll  
    for (size_t i = 0; i < NElems; i++)
    {
        x_vals[i] = float(x_vals_load[i]);
    }
}


template <int NElems, typename vec_t, typename output_t>
inline __device__ void store_output(output_t *out, float out_vals[NElems]) {
    output_t out_vals_store[NElems];

    constexpr int num_elems_per_store = sizeof(vec_t)/sizeof(output_t);
    constexpr int num_chunks = NElems/num_elems_per_store;

    #pragma unroll  
    for (size_t i = 0; i < NElems; i++)
    {
        out_vals_store[i] = out_vals[i];
    }
    #pragma unroll
    for (size_t i = 0; i < num_chunks; i++)
    {
        reinterpret_cast<vec_t*>(out)[num_chunks*threadIdx.x+i] = reinterpret_cast<const vec_t*>(out_vals_store)[i];
    }
}


template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

template <>
struct SumOp<float> {
__device__ inline float operator()(float const &x, float const &y) { return x + y; }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};


template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void rms_norm_kernel(RMSNormsParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int ThreadElems = Ktraits::ThreadElems;
    constexpr bool norm_affine = Ktraits::norm_affine;

    using input_t = typename Ktraits::input_t;
    using weights_t = typename Ktraits::weights_t;
    using vec_t = typename Ktraits::vec_t;
    using output_t = typename Ktraits::output_t;


    extern __shared__ float smem_[];

    const int batch_id = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    
    weights_t *weights = norm_affine ? reinterpret_cast<weights_t*>(params.weights_ptr) : nullptr;

    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;


    float oneoverdim = 1.0f/params.dim;
    float x_vals[ThreadElems];
    float weights_vals[ThreadElems];

    load_input<ThreadElems, input_t, vec_t>(x, x_vals);
    
    float thread_sum = x_vals[0] * x_vals[0];
    #pragma unroll  
    for (size_t i = 1; i < ThreadElems; i++)
    {
        thread_sum = thread_sum + x_vals[i] * x_vals[i];
    }

    SumOp<float> sum_op;
    float warp_sum = Allreduce<32>::run(thread_sum, sum_op);
    
    if(threadIdx.x % 32 == 0){
        smem_[warp_id] = warp_sum * oneoverdim;
    }
    __syncthreads();

    float norm = 0.0f;
    #pragma unroll  
    for (size_t i = 0; i < kNWarps; i++)
    {
        norm = norm + smem_[i];
    }
    __syncthreads();
    
    norm = rsqrtf(norm);
    if constexpr (norm_affine){
        load_input<ThreadElems, weights_t, vec_t>(weights, weights_vals);
    }
    #pragma unroll
    for (size_t i = 0; i < ThreadElems; i++)
    {
        if constexpr (norm_affine){
            x_vals[i] = (x_vals[i] * norm) * weights_vals[i];
        } else {
            x_vals[i] = (x_vals[i] * norm);
        }
    }
    store_output<ThreadElems, vec_t, output_t>(out, x_vals);
}


template<int kNThreads, int dim, bool norm_affine, typename input_t, typename output_t>
void rms_norm_launch(RMSNormsParamsBase &params, cudaStream_t stream) {
    using Ktraits = rmsnorm_kernel_traits<kNThreads, dim, norm_affine, input_t, output_t>;
    
    dim3 grid(params.batch);
    auto kernel = &rms_norm_kernel<Ktraits>;
    
    size_t shared_mem = sizeof(float) * kNThreads/32;

    kernel<<<grid, Ktraits::kNThreads, shared_mem, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<bool norm_affine, typename input_t, typename output_t>
void  rms_norm_cuda(RMSNormsParamsBase &params, cudaStream_t stream) {
    if (params.dim == 2048) {
        static constexpr int kNBytes_input = sizeof(input_t);
        static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
        static constexpr int kNThreads = 2048/ThreadElems;
        rms_norm_launch<kNThreads, 2048, norm_affine, input_t, output_t>(params, stream);
    } else if(params.dim == 8192){
        static constexpr int kNBytes_input = sizeof(input_t);
        static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
        static constexpr int kNThreads = 8192/ThreadElems;
        rms_norm_launch<kNThreads, 8192, norm_affine, input_t, output_t>(params, stream);  
    }
}

template void rms_norm_cuda<true, at::Float8_e4m3fn, at::Float8_e4m3fn>(RMSNormsParamsBase &params, cudaStream_t stream);
template void rms_norm_cuda<false, at::Float8_e4m3fn, at::Float8_e4m3fn>(RMSNormsParamsBase &params, cudaStream_t stream);

template void rms_norm_cuda<true, at::Float8_e4m3fn, at::BFloat16>(RMSNormsParamsBase &params, cudaStream_t stream);
template void rms_norm_cuda<false, at::Float8_e4m3fn, at::BFloat16>(RMSNormsParamsBase &params, cudaStream_t stream);

template void rms_norm_cuda<true, at::BFloat16, at::BFloat16>(RMSNormsParamsBase &params, cudaStream_t stream);
template void rms_norm_cuda<false, at::BFloat16, at::BFloat16>(RMSNormsParamsBase &params, cudaStream_t stream);

template void rms_norm_cuda<true, at::BFloat16, at::Float8_e4m3fn>(RMSNormsParamsBase &params, cudaStream_t stream);
template void rms_norm_cuda<false, at::BFloat16, at::Float8_e4m3fn>(RMSNormsParamsBase &params, cudaStream_t stream);

// template void rms_norm_cuda<true, at::Half, at::Half>(RMSNormsParamsBase &params, cudaStream_t stream);
// template void rms_norm_cuda<false, at::Half, at::Half>(RMSNormsParamsBase &params, cudaStream_t stream);

// template void rms_norm_cuda<true, at::Half, at::BFloat16>(RMSNormsParamsBase &params, cudaStream_t stream);
// template void rms_norm_cuda<false, at::Half, at::BFloat16>(RMSNormsParamsBase &params, cudaStream_t stream);

// template void rms_norm_cuda<true, at::BFloat16, at::Half>(RMSNormsParamsBase &params, cudaStream_t stream);
// template void rms_norm_cuda<false, at::BFloat16, at::Half>(RMSNormsParamsBase &params, cudaStream_t stream);

