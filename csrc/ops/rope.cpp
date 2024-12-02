#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

#include "rope.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


template<typename input_t>
void  rope_cuda(RopeParamsBase &params, cudaStream_t stream);


void set_rope_params(RopeParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x,
                         const at::Tensor cos_freqs,
                         const at::Tensor sin_freqs,
                         const at::Tensor out
                         
                         ) {

    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
  
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    
    params.cos_freq = cos_freqs.data_ptr();
    params.sin_freq = sin_freqs.data_ptr();
    
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.cos_freq_batch_stride = cos_freqs.stride(0);
    params.sin_freq_batch_stride = sin_freqs.stride(0);
}

at::Tensor rope(at::Tensor &x, at::Tensor& cos_freqs, at::Tensor& sin_freqs) {
    auto input_type = x.scalar_type();
    auto freq_type = cos_freqs.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(freq_type == at::ScalarType::Float);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(cos_freqs.is_cuda());
    TORCH_CHECK(sin_freqs.is_cuda());

    TORCH_CHECK(cos_freqs.sizes() == x.sizes());
    

    const auto shapes_og = x.sizes();    
    const int dim_og = x.size(-1);

    x = x.reshape({-1, dim_og});
    cos_freqs = cos_freqs.reshape({-1, dim_og});
    sin_freqs = sin_freqs.reshape({-1, dim_og});

    if (x.stride(-1) != 1) { x = x.contiguous(); }
    
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    const int dim = x.size(1);

  
    at::Tensor out = torch::empty_like(x);
    
    RopeParamsBase params;
    set_rope_params(params, batch_size, dim, x, cos_freqs, sin_freqs, out);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
   
    rope_cuda<at::Float8_e4m3fn>(params, stream);

    return out.reshape(shapes_og);
}
