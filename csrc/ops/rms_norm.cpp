#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

#include "rms_norm.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define INPUT_TYPE_SWITCH(ITYPE, ...)      \
    if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__;                                                              \
    } else if(ITYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using input_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__;                                                              \
    }                                                                               

#define OTYPE_SWITCH(OTYPE, ...)      \
    if (OTYPE == at::ScalarType::BFloat16) {                                 \
        using output_t = at::BFloat16;                                               \
        __VA_ARGS__;                                                              \
    } else if(OTYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using output_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__;                                                              \
    }                                                                               \



template<bool norm_affine, typename input_t, typename output_t>
void  rms_norm_cuda(RMSNormsParamsBase &params, cudaStream_t stream);
template<bool norm_affine, typename input_t, typename output_t>
void  rms_norm_backward_cuda(RMSNormsBackwardParamsBase &params, cudaStream_t stream);


void set_rms_norm_params(RMSNormsParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x,
                         const at::Tensor out,
                         const at::Tensor weights,
                         const at::Tensor out_scales,
                         bool norm_affine
                         ) {

    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
  
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.out_scales_ptr = out_scales.data_ptr();

    if(norm_affine){
        params.weights_ptr = weights.data_ptr();
    } else {
        params.weights_ptr = nullptr;
    }

    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.out_scales_stride = out_scales.stride(0);
}

void set_rms_norm_backward_params(RMSNormsBackwardParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x_normed,
                         const at::Tensor x_norms,
                         const at::Tensor grad_out,
                         const at::Tensor weights,
                         const at::Tensor out,
                         bool norm_affine
                         ) {
    
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;

    params.x_normed_ptr = x_normed.data_ptr();
    params.x_norms_ptr = x_norms.data_ptr();
    params.grad_out_ptr = grad_out.data_ptr();

    if(norm_affine){
        params.weights_ptr = weights.data_ptr();
    } else {
        params.weights_ptr = nullptr;
    }
    params.out_ptr = out.data_ptr();

    params.x_normed_batch_stride = x_normed.stride(0);
    params.x_norms_batch_stride = x_norms.stride(0);
    params.grad_out_batch_stride = grad_out.stride(0);
    params.out_batch_stride = out.stride(0);
}

std::vector<at::Tensor> rms_norm(at::Tensor &x, c10::optional<at::Tensor>& weights_, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
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

    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = x.scalar_type();
    }
  
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    at::Tensor out_scales = torch::empty({batch_size}, x.options().dtype(torch::kFloat32));
    at::Tensor weights;
    bool norm_affine = false;
    if(weights_.has_value()){
        weights = weights_.value();
        norm_affine = true;
        TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float && weights.is_cuda(), "rms norm: weights error");
    }

    RMSNormsParamsBase params;
    set_rms_norm_params(params, batch_size, dim, x, out, weights, out_scales, norm_affine);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    OTYPE_SWITCH(out_type, 
        INPUT_TYPE_SWITCH(input_type, 
            if (norm_affine){ 
                rms_norm_cuda<true, input_t, output_t>(params, stream); 
            } else {    
                rms_norm_cuda<false, input_t, output_t>(params, stream); 
            } 
        );
    );    
    return {out.reshape(shapes_og), out_scales.reshape(torch::IntArrayRef(shapes_og.begin(), shapes_og.size()-1))};
}


at::Tensor rms_norm_backward(at::Tensor &x_normed, at::Tensor &x_norms, at::Tensor &grad_out, c10::optional<at::Tensor>& weights_, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x_normed.scalar_type();
    TORCH_CHECK(x_normed.is_cuda());

    const auto shapes_og = x_normed.sizes();    
    const int dim_og = x_normed.size(-1);

    x_normed = x_normed.reshape({-1, dim_og});
    x_norms = x_norms.reshape({-1});
    grad_out = grad_out.reshape({-1, dim_og});
    
    if (x_normed.stride(-1) != 1) { x_normed = x_normed.contiguous(); }
    if (grad_out.stride(-1) != 1) { grad_out = grad_out.contiguous(); }

    const auto sizes = x_normed.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x_normed, batch_size, dim_og);
    TORCH_CHECK(x_normed.stride(1) == 1);

    const int dim = x_normed.size(1);

    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = x_normed.scalar_type();
    }
  
    at::Tensor out = torch::empty(x_normed.sizes(), x_normed.options().dtype(out_type));
    at::Tensor weights;
    bool norm_affine = false;
    if(weights_.has_value()){
        weights = weights_.value();
        norm_affine = true;
        TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float && weights.is_cuda(), "rms norm: weights error");
    }

    RMSNormsBackwardParamsBase params;
    set_rms_norm_backward_params(params, batch_size, dim, x_normed, x_norms, grad_out, weights, out, norm_affine);

    at::cuda::CUDAGuard device_guard{(char)x_normed.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    OTYPE_SWITCH(out_type, 
        INPUT_TYPE_SWITCH(input_type, 
            if (norm_affine){ 
                rms_norm_backward_cuda<true, input_t, output_t>(params, stream); 
            } else {    
                rms_norm_backward_cuda<false, input_t, output_t>(params, stream); 
            } 
        );
    );    
    return out.reshape(shapes_og);
}

