#include <torch/extension.h>
#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>



void flash_attention_cuda_mask(void* Q_ptr, void* K_ptr, void* V_ptr, int* BatchMask,
                   void* O_ptr, 
                   size_t BATCH, size_t M, size_t N, size_t NUM_HEADS, float softmax_scale, cudaStream_t stream);


void flash_attention_cuda(void* Q_ptr, void* K_ptr, void* V_ptr, 
                   void* O_ptr, 
                   size_t BATCH, size_t M, size_t N, size_t NUM_HEADS, float softmax_scale, cudaStream_t stream);

torch::Tensor flash_attention(torch::Tensor& q, torch::Tensor& k,
              torch::Tensor& v, float softmax_scale, c10::optional<at::Tensor>& batch_mask_){

    int bs = q.size(0);
    int head = q.size(1);
    int seqlen = q.size(2);
    int kseqlen = k.size(2);
    int dim = q.size(3);
    auto opts = q.options();

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    auto out = torch::empty_like(q);
    softmax_scale = softmax_scale*1.44269504088896340736f;

    if(batch_mask_.has_value()){
        at::Tensor batch_mask;
        batch_mask = batch_mask_.value();
        TORCH_CHECK(batch_mask.dtype() == torch::kInt32, "flash attn: mask type error");
        TORCH_CHECK(batch_mask.size(0) == bs, "flash attn: mask batch error");
        flash_attention_cuda_mask(q.data_ptr(), k.data_ptr(), v.data_ptr(), batch_mask.data_ptr<int>(),  out.data_ptr(), 
                  bs, seqlen, kseqlen, head, softmax_scale, stream);
    } else {
        flash_attention_cuda(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                  out.data_ptr(), 
                  bs, seqlen, kseqlen, head, softmax_scale, stream);
    
    }
    return out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention,
          "Flash attention v2 fp8");
}