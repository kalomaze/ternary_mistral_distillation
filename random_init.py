import os
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Specify the original folder path
original_folder = "/home/kalomaze/Downloads/mergekit/mergekit/onelayer7b"

# Specify the output folder path
output_folder = "random_init_1layer"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the original model weights
original_weights = {}
for filename in os.listdir(original_folder):
    if filename.endswith(".safetensors"):
        file_path = os.path.join(original_folder, filename)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                original_weights[tensor_name] = f.get_tensor(tensor_name)

# Create a new dictionary to store the randomly initialized weights
random_weights = {}

# Iterate over the original weights and create randomly initialized tensors
for tensor_name, tensor in original_weights.items():
    if tensor_name == "lm_head.weight":
        random_weights[tensor_name] = torch.randn_like(tensor)
    elif tensor_name == "model.embed_tokens.weight":
        random_weights[tensor_name] = torch.randn_like(tensor)
    else:
        random_weights[tensor_name] = torch.randn_like(tensor)

# Save the randomly initialized weights as SafeTensors
output_file = os.path.join(output_folder, "model.safetensors")
save_file(random_weights, output_file)

# Copy the tokenizer files to the output folder
tokenizer_files = ["tokenizer.json", "vocab.json"]  # Adjust the file names according to your tokenizer
for file in tokenizer_files:
    tokenizer_path = os.path.join(original_folder, file)
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, output_folder)

print(f"Randomly initialized 1-layer model and tokenizer saved in: {output_folder}")
