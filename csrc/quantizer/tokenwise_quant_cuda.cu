#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>
#include "tokenwise_quant.h"


template<typename TYPE> struct QuantMax {};
template<> struct QuantMax<int8_t> { static constexpr float value = 127.0; };
template<> struct QuantMax<at::Float8_e4m3fn> { static constexpr float value = 448.0; };


template<int kNThreads_, int dim, typename input_t_, typename output_t_>
struct quantizer_kernel_traits {

    using input_t = input_t_;
    using output_t = output_t_;
    using vec_t = uint4;
    
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNBytes_input = sizeof(input_t);
    static_assert(kNBytes_input == 1 || kNBytes_input == 2 || kNBytes_input == 4);
    static constexpr int ThreadElems = 16;
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


template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return max(x, y); }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
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
void tokenwise_quantization_kernel(QuantizerParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int ThreadElems = Ktraits::ThreadElems;

    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using output_t = typename Ktraits::output_t;

    extern __shared__ float smem_[];

    const int batch_id = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;
    
    float *out_scales = reinterpret_cast<float *>(params.out_scales_ptr) + batch_id;

    float x_vals[ThreadElems];
    load_input<ThreadElems, input_t, vec_t>(x, x_vals, params.dim);
    
    float thread_max = abs(x_vals[0]);
    #pragma unroll  
    for (size_t i = 1; i < ThreadElems; i++)
    {
        thread_max = max(abs(x_vals[i]), thread_max);
    }

    MaxOp<float> max_op;
    float warp_max = Allreduce<32>::run(thread_max, max_op);
    
    if(threadIdx.x % 32 == 0){
        smem_[warp_id] = warp_max;
    }
    __syncthreads();
    #pragma unroll  
    for (size_t i = 0; i < kNWarps; i++)
    {
        thread_max = max(thread_max, smem_[i]);
    }
     __syncthreads();
    float scale = QuantMax<output_t>::value/thread_max;
    #pragma unroll  
    for (size_t i = 0; i < ThreadElems; i++)
    {
        if constexpr(std::is_same<output_t, int8_t>::value){
            x_vals[i] = round(x_vals[i] * scale);
        } // TODO: fix this
        else {
            x_vals[i] = x_vals[i] * scale;
        }
    }

    *out_scales = 1/scale;
    store_output<ThreadElems, vec_t, output_t>(out, x_vals);
}

template<int kNThreads, int dim, typename input_t, typename output_t>
void quantizer_launch(QuantizerParamsBase &params, cudaStream_t stream) {
    using Ktraits = quantizer_kernel_traits<kNThreads, dim, input_t, output_t>;
    
    dim3 grid(params.batch);
    auto kernel = &tokenwise_quantization_kernel<Ktraits>;
    
    size_t shared_mem = sizeof(float) * kNThreads/32;

    kernel<<<grid, Ktraits::kNThreads, shared_mem, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename input_t, typename output_t>
void  quantizer_cuda(QuantizerParamsBase &params, cudaStream_t stream) {
    if (params.dim == 2048) {
        quantizer_launch<128, 2048, input_t, output_t>(params, stream);
    } else if(params.dim == 8192){
        quantizer_launch<512, 8192, input_t, output_t>(params, stream);  
    }
}

template void quantizer_cuda<float, int8_t>(QuantizerParamsBase &params, cudaStream_t stream);
template void quantizer_cuda<at::BFloat16, int8_t>(QuantizerParamsBase &params, cudaStream_t stream);
template void quantizer_cuda<at::Half, int8_t>(QuantizerParamsBase &params, cudaStream_t stream);
template void quantizer_cuda<at::Float8_e4m3fn, int8_t>(QuantizerParamsBase &params, cudaStream_t stream);

template void quantizer_cuda<float, at::Float8_e4m3fn>(QuantizerParamsBase &params, cudaStream_t stream);
template void quantizer_cuda<at::BFloat16, at::Float8_e4m3fn>(QuantizerParamsBase &params, cudaStream_t stream);
template void quantizer_cuda<at::Half, at::Float8_e4m3fn>(QuantizerParamsBase &params, cudaStream_t stream);
template void quantizer_cuda<at::Float8_e4m3fn, at::Float8_e4m3fn>(QuantizerParamsBase &params, cudaStream_t stream);
