/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

/*
KONAKONA666: added fp8 support
*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "fast_hadamard_transform.h"

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
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}


at::Tensor
fast_hadamard_transform(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::BFloat16 || input_type == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");

    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = x.scalar_type();
    }

    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    HOTYPE_SWITCH(out_type, 
        HINPUT_TYPE_SWITCH(x.scalar_type(), 
            fast_hadamard_transform_cuda<input_t, output_t>(params, stream);
        );
    );    

    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}


