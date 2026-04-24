import torch
print("CUDA verfügbar:", torch.cuda.is_available())
print("CUDA Geräteanzahl:", torch.cuda.device_count())
print("Aktives Gerät:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
