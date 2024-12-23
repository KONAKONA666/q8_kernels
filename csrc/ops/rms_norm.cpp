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

