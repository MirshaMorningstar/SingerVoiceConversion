import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch built with):", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


device = torch.device("cuda")

a = torch.randn(1000, 1000).to(device)
b = torch.randn(1000, 1000).to(device)

c = torch.matmul(a, b)

print("Matrix multiplication successful on GPU")
print(c.device)