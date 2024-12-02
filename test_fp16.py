import torch
import time

# Make sure to use a GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the size of the matrices
n = 4096  # Size of the matrix (n x n)

# Create random matrices
A = torch.rand(n, n, device=device, dtype=torch.float16)
B = torch.rand(n, n, device=device, dtype=torch.float16)

# Warm-up (to ensure fair timing, especially important for GPU)
for _ in range(10):
    _ = torch.mm(A, B)

if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for all kernels to finish (CUDA-specific)

# Time the matrix multiplication
start_time = time.time()
C = torch.mm(A, B)
if torch.cuda.is_available():
    torch.cuda.synchronize()  # Ensure the multiplication is finished
end_time = time.time()

print(f'Time taken for matrix multiplication of size {n}x{n}: {end_time - start_time:.4f} seconds')
print(f"Performance: {((2 * n**3 - n**2) / (end_time - start_time))/1E12:.4f} TFLOPs")