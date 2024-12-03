
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>



at::Tensor rope(at::Tensor &x, at::Tensor& cos_freqs, at::Tensor& sin_freqs);
at::Tensor rms_norm(at::Tensor &x, c10::optional<at::Tensor>& weights_);
at::Tensor fma_8bit(at::Tensor &x, at::Tensor& y, at::Tensor& z, float scale_add);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fma_8bit", &fma_8bit, "fma 8bit");
    m.def("rope", &rope, "apply rope");
    m.def("rms_norm", &rms_norm, "calculate rms norm");
}