#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>

#include <cute/tensor.hpp>

#include "rope.h"


template<int kNThreads_, int dim, typename input_t_, typename output_t_>
struct rope_kernel_traits {

    using input_t = input_t_;
    using output_t = output_t_;
    using freqs_t = float;
    using vec_t = uint4;
    
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNBytes_input = sizeof(input_t);
    static_assert(kNBytes_input == 1 || kNBytes_input == 2 || kNBytes_input == 4);
    static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
    static_assert(ThreadElems * kNThreads == dim);
};

template <int NElems, typename input_t, typename vec_t>
inline __device__ void load_input(input_t *x, float x_vals[NElems], int dim) {
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


template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void rope_backward_kernel(RopeBackwardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int ThreadElems = Ktraits::ThreadElems;
    // constexpr int dim = Ktraits::dim_;
    
    using input_t = typename Ktraits::input_t;
    using freqs_t = typename Ktraits::freqs_t;
    using vec_t = typename Ktraits::vec_t;
    using output_t = typename Ktraits::output_t;
    
    extern __shared__ float smem_[];

    const int batch_id = blockIdx.x;
    
    output_t *grad_out = reinterpret_cast<output_t *>(params.grad_out_ptr) + batch_id * params.grad_out_batch_stride;
    freqs_t *cos_freqs = reinterpret_cast<freqs_t*>(params.cos_freq) + batch_id * params.cos_freq_batch_stride;
    freqs_t *sin_freqs = reinterpret_cast<freqs_t*>(params.sin_freq) + batch_id * params.sin_freq_batch_stride;

    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float grad_out_vals[ThreadElems];

    float out_vals[ThreadElems];
    float sin_freqs_vals[ThreadElems];
    float cos_freqs_vals[ThreadElems];

    load_input<ThreadElems, output_t, vec_t>(grad_out, grad_out_vals, params.dim);
    load_input<ThreadElems, freqs_t, vec_t>(cos_freqs, cos_freqs_vals, params.dim);
    load_input<ThreadElems, freqs_t, vec_t>(sin_freqs, sin_freqs_vals, params.dim);
    
    for (int i = 0; i < ThreadElems; i+=2)
    {
        
        // out_vals[i] = 
        // // dL/dX = dL/dO * dO/dX
        // o = ROT_MATRIX @ x.T
        
        out_vals[i] = grad_out_vals[i]*cos_freqs_vals[i] + grad_out_vals[i+1]*sin_freqs_vals[i+1];
        out_vals[i+1] = -grad_out_vals[i]*sin_freqs_vals[i] + grad_out_vals[i+1]*cos_freqs_vals[i+1];

        // out_vals[i] = -x_vals[i+1]*sin_freqs_vals[i] + x_vals[i]*cos_freqs_vals[i];
        // out_vals[i+1] = x_vals[i]*sin_freqs_vals[i+1] + x_vals[i+1]*cos_freqs_vals[i+1];
    }
    
    store_output<ThreadElems, vec_t, output_t>(out, out_vals);
}



template<int kNThreads, int dim, typename input_t, typename output_t>
void rope_backward_launch(RopeBackwardParamsBase &params, cudaStream_t stream) {
    using Ktraits = rope_kernel_traits<kNThreads, dim, input_t, output_t>;
    
    dim3 grid(params.batch);
    auto kernel = &rope_backward_kernel<Ktraits>;
    size_t shared_mem = 0;

    kernel<<<grid, Ktraits::kNThreads, shared_mem, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename input_t, typename output_t>
void  rope_backward_cuda(RopeBackwardParamsBase &params, cudaStream_t stream) {
    if (params.dim == 2048) {
        static constexpr int kNBytes_input = sizeof(input_t);
        static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
        static constexpr int kNThreads = 2048/ThreadElems;
        rope_backward_launch<kNThreads, 2048, input_t, output_t>(params, stream);
    } else if(params.dim == 8192){
        static constexpr int kNBytes_input = sizeof(input_t);
        static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
        static constexpr int kNThreads = 8192/ThreadElems;
        rope_backward_launch<kNThreads, 8192, input_t, output_t>(params, stream);  
    }
}

template void rope_backward_cuda<at::Float8_e4m3fn, at::Float8_e4m3fn>(RopeBackwardParamsBase &params, cudaStream_t stream);
template void rope_backward_cuda<at::Float8_e4m3fn, at::BFloat16>(RopeBackwardParamsBase &params, cudaStream_t stream);
template void rope_backward_cuda<at::BFloat16, at::Float8_e4m3fn>(RopeBackwardParamsBase &params, cudaStream_t stream);
template void rope_backward_cuda<at::BFloat16, at::BFloat16>(RopeBackwardParamsBase &params, cudaStream_t stream);


