
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>




at::Tensor rope(at::Tensor &x, at::Tensor& cos_freqs, at::Tensor& sin_freqs, std::optional<at::ScalarType>& out_type_); 
std::vector<at::Tensor> rms_norm(at::Tensor &x, c10::optional<at::Tensor>& weights_, std::optional<at::ScalarType>& out_type_);
at::Tensor fma_8bit(at::Tensor &x, at::Tensor& y, at::Tensor& z, float scale_add);
at::Tensor fast_hadamard_transform(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_);
at::Tensor gelu_backward(at::Tensor &x, at::Tensor& grad_output, std::optional<at::ScalarType>& out_type_);
at::Tensor gelu_forward(at::Tensor &x, std::optional<at::ScalarType>& out_type_);
at::Tensor rms_norm_backward(at::Tensor &x_normed, at::Tensor &x_norms, at::Tensor &grad_out, c10::optional<at::Tensor>& weights_, std::optional<at::ScalarType>& out_type_);
at::Tensor rope_backward(at::Tensor &grad_out, at::Tensor& cos_freqs, at::Tensor& sin_freqs, std::optional<at::ScalarType>& out_type_);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fma_8bit", &fma_8bit, "fma 8bit");
    m.def("rope", &rope, "apply rope");
    m.def("rope_backward", &rope_backward, "Rope backward");
    m.def("rms_norm", &rms_norm, "calculate rms norm");
    m.def("rms_norm_backward", &rms_norm_backward, "rms norm backward");
    m.def("fast_hadamard_transform", &fast_hadamard_transform, "Fast Hadamard transform");
    m.def("gelu_backward", &gelu_backward, "GELU backward");
    m.def("gelu_forward", &gelu_forward, "GELU forward");
    
}
