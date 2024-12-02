#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <vector>
#include "fma.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


#define DISPATCH_TYPES(YTYPE, ZTYPE, NAME, ...)                                     \
    if (YTYPE == at::ScalarType::Float && ZTYPE == at::ScalarType::Float) {         \
        using y_type = float;                                                       \
        using z_type = float;                                                       \
        __VA_ARGS__();                                                              \
    } else if (YTYPE == at::ScalarType::Float && ZTYPE == at::ScalarType::Float8_e4m3fn) {      \
        using y_type = float;                                                       \
        using z_type = at::Float8_e4m3fn;                                           \
        __VA_ARGS__();                                                              \
    } else if (YTYPE == at::ScalarType::BFloat16 && ZTYPE == at::ScalarType::BFloat16) {      \
        using y_type = at::BFloat16;                                                       \
        using z_type = at::BFloat16;                                                       \
        __VA_ARGS__();                                                              \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for input type '", toString(YTYPE), "'"); \
    }

template<typename x_type, typename y_type, typename z_type>
void fma_cuda(FMABaseParams &params, cudaStream_t stream);


void set_fma_params(FMABaseParams &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x,
                         const at::Tensor y,
                         const at::Tensor z,
                         const at::Tensor out,

                          int y_batch_change,
                          int z_batch_change,
                          float scale_add
        
                         ) {

    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    
    params.x_ptr = x.data_ptr();
    params.y_ptr = y.data_ptr();
    params.z_ptr = z.data_ptr();
    
    params.out_ptr = out.data_ptr();

    params.x_batch_stride = x.stride(0);
    params.y_batch_stride = y.stride(0);
    params.z_batch_stride = z.stride(0);
    params.out_batch_stride = out.stride(0);

    params.y_change_batch_every = y_batch_change;
    params.z_change_batch_every = z_batch_change;
    
    params.scale_add = scale_add;

}

at::Tensor fma_8bit(at::Tensor &x, at::Tensor& y, at::Tensor& z, float scale_add) {
    auto input_type = x.scalar_type();
   

    TORCH_CHECK(input_type == at::ScalarType::Float8_e4m3fn);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(y.is_cuda());
    TORCH_CHECK(z.is_cuda());
    
    const auto shapes_og = x.sizes();    
    const int dim_og = x.size(-1);

    TORCH_CHECK(shapes_og.size() == 3);
  

    TORCH_CHECK(y.size(-1) == dim_og && z.size(-1) == dim_og, "fma: channels missmatch");
    TORCH_CHECK((y.size(-2) == x.size(-2) || y.size(-2) == 1) && (z.size(-2) == z.size(-2) || z.size(-2) == 1), "fma: token missmatch");
    // add checks
    
    const int y_batch_every = x.size(-2)/y.size(-2);
    const int z_batch_every = x.size(-2)/z.size(-2);

    x = x.reshape({-1, dim_og});
    y = y.reshape({-1, dim_og});
    z = z.reshape({-1, dim_og});
    
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    if (y.stride(-1) != 1) { y = y.contiguous(); }
    if (z.stride(-1) != 1) { z = z.contiguous(); }
    
    int batch = x.size(0);
    const int dim = x.size(1);
    at::Tensor out = torch::empty_like(x);

    FMABaseParams params;
    set_fma_params(params, batch, dim, x, y, z, out, y_batch_every, z_batch_every, scale_add);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    DISPATCH_TYPES(y.scalar_type(), z.scalar_type(), "fma", [&] {
        fma_cuda<at::Float8_e4m3fn, y_type, z_type>(params, stream);
    });

    return out.reshape(shapes_og);
}
