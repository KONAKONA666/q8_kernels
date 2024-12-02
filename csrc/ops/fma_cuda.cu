#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>

#include <cute/tensor.hpp>

#include "fma.h"

template<int kNThreads_, int dim, typename x_type, typename y_type, typename z_type>
struct fma_kernel_traits {

    using x_type_t = x_type;
    using y_type_t = y_type;
    using z_type_t = z_type;
    using vec_t = uint4;
    
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNBytes_input = sizeof(x_type_t);
    static_assert(kNBytes_input == 1 || kNBytes_input == 2 || kNBytes_input == 4);
    static constexpr int ThreadElems = kNBytes_input == 4 ? 4 : kNBytes_input == 2 ? 8 : 16;
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
    #pragma unroll  
    for (size_t i = 0; i < NElems; i++)
    {
        out_vals_store[i] = out_vals[i];
    }
    reinterpret_cast<vec_t*>(out)[threadIdx.x] = reinterpret_cast<const vec_t*>(out_vals_store)[0];
}



template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fma_kernel(FMABaseParams params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int ThreadElems = Ktraits::ThreadElems;

    using x_type = typename Ktraits::x_type_t;
    using y_type = typename Ktraits::y_type_t;
    using z_type = typename Ktraits::z_type_t;
    using output_t = typename Ktraits::x_type_t;
    using vec_t = typename Ktraits::vec_t;
    
    const int batch_id = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    x_type *x = reinterpret_cast<x_type*>(params.x_ptr) + batch_id * params.x_batch_stride;
    
    y_type *y = reinterpret_cast<y_type*>(params.y_ptr) +  (batch_id / params.y_change_batch_every) * params.y_batch_stride;
    z_type *z = reinterpret_cast<z_type*>(params.z_ptr) +  (batch_id / params.z_change_batch_every) * params.z_batch_stride;

    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[ThreadElems];    
    float y_vals[ThreadElems];    
    float z_vals[ThreadElems];    
    
    load_input<ThreadElems, x_type, vec_t>(x, x_vals, params.dim);
    load_input<ThreadElems, y_type, vec_t>(y, y_vals, params.dim);
    load_input<ThreadElems, z_type, vec_t>(z, z_vals, params.dim);
    
    for (size_t i = 0; i < ThreadElems; i++)
    {
        x_vals[i] = x_vals[i]*(y_vals[i]+params.scale_add) + z_vals[i];
    }
    
    store_output<ThreadElems, vec_t, output_t>(out, x_vals);
}



template<int kNThreads, int dim, typename x_type, typename y_type, typename z_type>
void fma_launch(FMABaseParams &params, cudaStream_t stream) {
    using Ktraits = fma_kernel_traits<kNThreads, dim, x_type, y_type, z_type>;
    
    dim3 grid(params.batch);
    auto kernel = &fma_kernel<Ktraits>;
    size_t shared_mem = 0;

    kernel<<<grid, Ktraits::kNThreads, shared_mem, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename x_type, typename y_type, typename z_type>
void  fma_cuda(FMABaseParams &params, cudaStream_t stream) {
    if (params.dim == 2048) {
        fma_launch<128, 2048, x_type, y_type, z_type>(params, stream);
    } else if(params.dim == 8192){
        fma_launch<512, 8192, x_type, y_type, z_type>(params, stream);  
    }
}

template void fma_cuda<at::Float8_e4m3fn, float, float>(FMABaseParams &params, cudaStream_t stream);
template void fma_cuda<at::Float8_e4m3fn, float, at::Float8_e4m3fn>(FMABaseParams &params, cudaStream_t stream);
template void fma_cuda<at::Float8_e4m3fn, at::BFloat16, at::BFloat16>(FMABaseParams &params, cudaStream_t stream);
