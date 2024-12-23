#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>

#include <cute/tensor.hpp>

#include "gelu_act.h"


template<int kNThreads_, int dim, typename input_t_, typename output_t_>
struct gelu_backward_kernel_traits {

    using input_t = input_t_;
    using output_t = output_t_;
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
void gelu_backward_kernel(GELUBackwardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int ThreadElems = Ktraits::ThreadElems;
    
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using output_t = typename Ktraits::output_t;
    
    const int batch_id = blockIdx.x;
    
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;
    output_t *grad_output = reinterpret_cast<output_t *>(params.grad_output_ptr) + batch_id * params.grad_output_batch_stride;

    float x_vals[ThreadElems];
    float out_vals[ThreadElems];
    float grad_output_vals[ThreadElems];
    
    constexpr float sqrthalfpi2 = 0.7978845608028653558798921198687637369517172623298693153318516593f;
    constexpr float factor = 0.044715f;
    
    load_input<ThreadElems, input_t, vec_t>(x, x_vals, params.dim);
    load_input<ThreadElems, output_t, vec_t>(grad_output, grad_output_vals, params.dim);

    for (size_t i = 0; i < ThreadElems; i++){

        float tau = sqrthalfpi2*(x_vals[i] + factor*x_vals[i]*x_vals[i]*x_vals[i]);
        float phi = tanhf(tau);
        float phi_prime = (1.0f - phi*phi)*(sqrthalfpi2*(1.0f+factor*3.0f*x_vals[i]*x_vals[i]));
      
        out_vals[i] = grad_output_vals[i] * (0.5f*(1.0f+phi)+ 0.5f*x_vals[i]*phi_prime);
    }

    store_output<ThreadElems, vec_t, output_t>(out, out_vals);
}


template<int kNThreads, int dim, typename input_t, typename output_t>
void gelu_backward_launch(GELUBackwardParamsBase &params, cudaStream_t stream) {
    using Ktraits = gelu_backward_kernel_traits<kNThreads, dim, input_t, output_t>;
    
    dim3 grid(params.batch);
    auto kernel = &gelu_backward_kernel<Ktraits>;
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename input_t, typename output_t>
void  gelu_backward_cuda(GELUBackwardParamsBase &params, cudaStream_t stream) {
    if (params.dim == 2048) {
         static constexpr int kNBytes_input = sizeof(input_t);
        static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
        static constexpr int kNThreads = 2048/ThreadElems;
        gelu_backward_launch<kNThreads, 2048, input_t, output_t>(params, stream);
    } else if(params.dim == 8192){
        static constexpr int kNBytes_input = sizeof(input_t);
        static constexpr int ThreadElems = kNBytes_input == 1 ? 16 : kNBytes_input == 2 ? 8 : 4;
        static constexpr int kNThreads = 8192/ThreadElems;
        gelu_backward_launch<kNThreads, 8192, input_t, output_t>(params, stream);  
    }
}

template void gelu_backward_cuda<at::Float8_e4m3fn, at::Float8_e4m3fn>(GELUBackwardParamsBase &params, cudaStream_t stream);
template void gelu_backward_cuda<at::Float8_e4m3fn, at::BFloat16>(GELUBackwardParamsBase &params, cudaStream_t stream);
template void gelu_backward_cuda<at::BFloat16, at::Float8_e4m3fn>(GELUBackwardParamsBase &params, cudaStream_t stream);
template void gelu_backward_cuda<at::BFloat16, at::BFloat16>(GELUBackwardParamsBase &params, cudaStream_t stream);


