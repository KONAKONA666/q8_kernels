#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <cute/tensor.hpp>
// NOTE:tensor malloc as device before we call
// e.g. data.to("cuda") in python
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)                                                           \


#define OUTPUT_TYPE_SWITCH(OTYPE, ...)      \
    if (OTYPE == at::ScalarType::BFloat16) {                                 \
        using output_t = cute::bfloat16_t;                                               \
        __VA_ARGS__;                                                              \
    } else if(OTYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using output_t = cute::float_e4m3_t;                                          \
        __VA_ARGS__;                                                              \
    }                             
template<typename output_t>
void run_q8_gemm_bias(int8_t *A, int8_t *B,  float* bias, void *C, float* A_scales, float* B_scales, int BA, int BB, int M, int N, int K, bool fuse_gelu, cudaStream_t stream);
template<typename output_t>
void run_q8_gemm(int8_t *A, int8_t *B, void *C, float* A_scales, float* B_scales, int BA, int BB, int M, int N, int K, bool fuse_gelu, cudaStream_t stream);


torch::Tensor q8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor a_scales, torch::Tensor b_scales, bool fuse_gelu, std::optional<at::ScalarType>& out_type_){

    CHECK_INPUT(a);
    CHECK_INPUT(b);

    int m, n, k;

    // batch size
    int a_ndim = a.sizes().size();
    int b_ndim = b.sizes().size();

    int bs_a;
    if(a_ndim == 3){
        bs_a = a.size(0);
        m = a.size(1);
    } else {
        bs_a = 1;
        m = a.size(0);
    }
    
    int bs_b;
    if(b_ndim == 3){
        bs_b = b.size(0);
        n = b.size(1);
    } else {
        bs_b = 1;
        n = b.size(0);
    }

    k = a.size(a_ndim - 1);

    TORCH_CHECK(bs_a == bs_b || bs_a == 1 || bs_b == 1, "Batch missmatch");

    int batch;
    if(bs_a == 1 || bs_b == 1){
        batch = bs_a * bs_b;
    } else {
        batch = bs_a;
    }
    
    at::cuda::CUDAGuard device_guard{(char)a.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = torch::kFloat8_e4m3fn;
    }

    auto opts = a.options();

    auto out = torch::empty({batch, m, n}, opts.dtype(out_type));
    OUTPUT_TYPE_SWITCH(out_type, 
        run_q8_gemm<output_t>(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), out.data_ptr(), a_scales.data_ptr<float>(), b_scales.data_ptr<float>(), bs_a, bs_b, m, n, k, fuse_gelu, stream);
    );
    
    return out;
}


torch::Tensor q8_mm_bias(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor a_scales, torch::Tensor b_scales, bool fuse_gelu, std::optional<at::ScalarType>& out_type_){

    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(bias);
    CHECK_INPUT(a_scales);
    CHECK_INPUT(b_scales);
    
    int m, n, k;

    // batch size
    int a_ndim = a.sizes().size();
    int b_ndim = b.sizes().size();

    int bs_a;
    if(a_ndim == 3){
        bs_a = a.size(0);
        m = a.size(1);
    } else {
        bs_a = 1;
        m = a.size(0);
    }
    
    int bs_b;
    if(b_ndim == 3){
        bs_b = b.size(0);
        n = b.size(1);
    } else {
        bs_b = 1;
        n = b.size(0);
    }

    k = a.size(a_ndim - 1);

    TORCH_CHECK(bs_a == bs_b || bs_a == 1 || bs_b == 1, "Batch missmatch");
    
    int batch;
    if(bs_a == 1 || bs_b == 1){
        batch = bs_a * bs_b;
    } else {
        batch = bs_a;
    }

    at::cuda::CUDAGuard device_guard{(char)a.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = torch::kFloat8_e4m3fn;
    }

    auto opts = a.options();

    auto out = torch::empty({batch, m, n}, opts.dtype(out_type));
    
    OUTPUT_TYPE_SWITCH(out_type, 
        run_q8_gemm_bias<output_t>(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), bias.data_ptr<float>(), out.data_ptr(), 
                a_scales.data_ptr<float>(), b_scales.data_ptr<float>(), 
                bs_a, bs_b, m, n, k, fuse_gelu, stream);
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("q8_mm", &q8_mm,
          "q8 matmul");
    m.def("q8_mm_bias", &q8_mm_bias, 
          "fuse bias add q8 mm");
}

