import torch, platform, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")
print("python:", sys.version)
print("os:", platform.platform())
