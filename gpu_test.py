import torch

print("PyTorch version:", torch.__version__)
print("Is GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
