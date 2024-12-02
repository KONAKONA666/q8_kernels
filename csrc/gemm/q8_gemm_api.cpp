#include <torch/extension.h>
#include <torch/python.h>

#include "q8_gemm_api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("q8_mm", &q8_mm,
          "q8 matmul");
    m.def("q8_mm_bias", &q8_mm_bias, 
          "fuse bias add q8 mm");
}

