#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

#include "tokenwise_quant.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

template<typename input_t, typename output_t>
void quantizer_cuda(QuantizerParamsBase &params, cudaStream_t stream);


void set_quantizer_params(QuantizerParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x,
                         const at::Tensor out,
                         const at::Tensor out_scales
                         
                         ) {

    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
  
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.out_scales_ptr = out_scales.data_ptr();

    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.out_scales_batch_stride = out_scales.stride(0);
    
}

std::vector<at::Tensor> tokenwise_quantize(at::Tensor &x) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    
    const int dim_og = x.size(-1);

    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    const int dim = x.size(1);

    auto opts = x.options();
    at::Tensor out = torch::empty({batch_size, dim},  opts.dtype(torch::kInt8));
    at::Tensor out_scales = torch::empty({batch_size}, opts.dtype(torch::kFloat32));

    QuantizerParamsBase params;
    set_quantizer_params(params, batch_size, dim, x, out, out_scales);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
   
    quantizer_cuda<at::Float8_e4m3fn, int8_t>(params, stream);

    return {out.reshape(shapes_og), out_scales.reshape(torch::IntArrayRef(shapes_og.begin(), shapes_og.size()-1))};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tokenwise_quant", &tokenwise_quantize, "tokenwise_quantize int8");
}