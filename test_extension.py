import os
os.environ["USE_NINJA"] = "0"  # Ninja derleyicisini tamamen devre disi birakir

import time
import torch
from torch.utils.cpp_extension import load
print ("c++ eklentisi derleniyor JIT complation")

#pytorch c++ extension yükleyicisi
custom_matmul = load(
    name="custom_matmul_extension",
    sources=["matmul_kernel.cu"],
    extra_cuda_cflags=["-arch=sm_89", "-Xcompiler", "/Zc:preprocessor" ],
    verbose=True
)

print(" derleme tamamlandı. test baslatılıyor. \n")

N = 1024

# PyTorch GPU Tensorları olusturma
A = torch.ones((N, N), device='cuda', dtype=torch.float32)
B = torch.full((N, N), 2.0, device='cuda', dtype=torch.float32)

# Warm-up  gpu saat hızlarını sabitlemek için
_ = custom_matmul.matmul_vectorized(A, B)
_ = torch.matmul(A, B)
torch.cuda.synchronize()

#kendi kodumuzun performansı kernel.
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
C_custom = custom_matmul.matmul_vectorized(A, B)
end_event.record()
torch.cuda.synchronize()

custom_time = start_event.elapsed_time(end_event)

#Yerel PyTorch cublass performansi
start_event.record()
C_pytorch = torch.matmul(A, B)
end_event.record()
torch.cuda.synchronize()

pytorch_time = start_event.elapsed_time(end_event)

# Doğrulama kontrolü sonuclar aynı mı bizim kernel ile pytorch un.
is_correct = torch.allclose(C_custom, C_pytorch)
print(f"Matrix Size              : {N}x{N}")
print(f"Custom CUDA Kernel Time  : {custom_time:.4f} ms")
print(f"PyTorch (cuBLAS) Time    : {pytorch_time:.4f} ms")
print(f"Sonuclar Eslesiyor mu?   : {is_correct}")