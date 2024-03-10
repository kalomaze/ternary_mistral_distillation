import os
import torch
from safetensors import safe_open

# Specify the folder path
folder_path = "/home/kalomaze/Downloads/mergekit/random_init_1layer"

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    # Check if the file has a .safetensors extension
    if filename.endswith(".safetensors"):
        file_path = os.path.join(folder_path, filename)
        
        # Load the SafeTensor weights using safetensors
        weights = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                weights[tensor_name] = f.get_tensor(tensor_name)
        
        # Print detailed information about the weights
        print(f"File: {filename}")
        print(f"Number of tensors: {len(weights)}")
        
        # Iterate over each tensor in the weights
        for tensor_name, tensor in weights.items():
            print(f"\nTensor: {tensor_name}")
            print(f"Shape: {tensor.shape}")
            print(f"Data type: {tensor.dtype}")
            print(f"Device: {tensor.device}")
        
        print("---")
