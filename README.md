# Q8 Kernels

---

Q8Kernels is a efficent implementation of 8bit kernels(FP8 and INT8).
## Features:
-8bit GEMM(with fused gelu and bias) / 2x faster than cuBLAS FP8 and 4x faster than torch.mm <br />
-FP8 Flash Attention 2 with Fast Hadamard Transform(also supports cross attention mask) / 2x faster than flash attention 2 <br />
-Mixed Precision Fast Hadamard Transform  <br />
-RMSNorm <br />
-Mixed Precision FMA <br />
-RoPE Layer <br />
-Quantizers <br />

All operations are implemented in CUDA. 
Current version supports ADA Architecture(Ampere optimizations are coming soon!).

## Installation

q8_kernels requires CUDA Version >= 12.4 and pytorch >=2.4.
q8_kernels was tested on Windows machine. Dont see problem with building on Linux systems.
Install ninja ```pip install ninja```
Make sure that ninja is installed and that it works correctly (e.g. ninja --version).
Without ninja installation is very slow.

```
git clone https://github.com/KONAKONA666/q8_kernels
cd q8_kernels 
git submodule init
git submodule update

python setup.py install
pip install . # for utility
```

It takes ~10-15 minutes to compile and install all modules.


## Supported models
Speed ups are computed relative to transformers with inference with 16bit and flash attention 2 
|Model name | Speed up                                 |
| -------- | -------- |
| [LTXVideo](https://github.com/KONAKONA666/LTX-Video)| up to 2.5x |

## Acknowledgement
Thanks to:
[Flash attention](https://github.com/Dao-AILab/flash-attention/tree/main)

[@66RING](https://github.com/66RING/tiny-flash-attention)

[fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)

[cutlass](https://github.com/NVIDIA/cutlass)

[@weishengying](https://github.com/weishengying): Check his CUTE exercises and flash attn implementations

## Authors
KONAKONA666
## License

MIT
**Free Software**
