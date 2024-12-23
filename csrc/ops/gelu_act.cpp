#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

#include "gelu_act.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define HINPUT_TYPE_SWITCH(ITYPE, ...)      \
    if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__;                                                              \
    } else if(ITYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using input_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__;                                                              \
    }                                                                               

#define HOTYPE_SWITCH(OTYPE, ...)      \
    if (OTYPE == at::ScalarType::BFloat16) {                                 \
        using output_t = at::BFloat16;                                               \
        __VA_ARGS__;                                                              \
    } else if(OTYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using output_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__;                                                              \
    }                                                                               \


template<typename input_t, typename output_t>
void  gelu_backward_cuda(GELUBackwardParamsBase &params, cudaStream_t stream);


void set_gelu_backward_params(GELUBackwardParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x,
                         const at::Tensor grad_output,
                         const at::Tensor out
                         
                         ) {

    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
  
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    
    params.grad_output_ptr = grad_output.data_ptr();
    
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.grad_output_batch_stride = grad_output.stride(0);
}

at::Tensor gelu_backward(at::Tensor &x, at::Tensor& grad_output, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    auto grad_output_type = grad_output.scalar_type();
    
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(grad_output.is_cuda());

    TORCH_CHECK(grad_output.sizes() == x.sizes());
    
    const auto shapes_og = x.sizes();    
    const int dim_og = x.size(-1);

    x = x.reshape({-1, dim_og});
    grad_output = grad_output.reshape({-1, dim_og});
    
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    if (grad_output.stride(-1) != 1) { grad_output = grad_output.contiguous(); }

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    const int dim = x.size(1);

    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = x.scalar_type();
    }
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    
    GELUBackwardParamsBase params;
    set_gelu_backward_params(params, batch_size, dim, x, grad_output, out);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    HOTYPE_SWITCH(out_type, 
        HINPUT_TYPE_SWITCH(x.scalar_type(), 
            gelu_backward_cuda<input_t, output_t>(params, stream);
        );
    );    

    return out.reshape(shapes_og);
}
