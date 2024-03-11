import os
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Specify the paths for the source and target models
source_model_path = "/home/kalomaze/Downloads/mergekit/mergekit/onelayer7b"
#target_model_path = "/home/kalomaze/Downloads/mergekit/random_init_1layer"
target_model_path = "/home/kalomaze/Downloads/mergekit/mergekit/onelayer7b_dupe"

# Load the source model
source_model = AutoModelForCausalLM.from_pretrained(source_model_path)
source_model.eval()  # Set the source model to evaluation mode
for param in source_model.parameters():
    param.requires_grad = False  # Set source model parameters to not require gradients

# Load the target model tensors
target_model_file = os.path.join(target_model_path, "model.safetensors")
target_tensors = load_file(target_model_file)

# Create a config for the target model
config = AutoConfig.from_pretrained(source_model_path)

# Create the target model from the loaded tensors and config
target_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=target_tensors,
    config=config,
)
target_model.train()  # Set the target model to training mode

# Create a config for the target model
config = AutoConfig.from_pretrained(source_model_path)

# Create the target model from the loaded tensors and config
target_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=target_tensors,
    config=config,
)
target_model.train()

## Add dropout layer to the target model
# target_model.dropout = torch.nn.Dropout(p=0)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(source_model_path)

# Set the device to use (e.g., "cuda" for GPU, "cpu" for CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
source_model.to(device)
target_model.to(device)

# Set the training hyperparameters
num_epochs = 1
batch_size = 2
learning_rate = 0.1 # Define the starting learning rate
context_size = 4096
num_samples_per_epoch = 5000  # Manually specify the number of samples to generate per epoch
num_training_steps = num_epochs * num_samples_per_epoch  # Total number of samples to process

# Define the optimizer
optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

# Define the TernaryLinear module
class TernaryLinear(torch.nn.Linear):
    def forward(self, input):
        quantized_weight = self.quantize_weight(self.weight)
        return torch.nn.functional.linear(input, quantized_weight)

    def quantize_weight(self, weight):
        gamma = torch.mean(torch.abs(weight.data))
        quantized_weight = torch.clamp(F.hardtanh(weight.data / gamma + 1e-8, min_val=-1, max_val=1), -1, 1)
        return quantized_weight

# Quantize the target model and freeze weights except for MLPs and attention
def quantize_model(model, use_ternary_layer):
    for name, module in model.named_children():
        print(f"Processing layer: {name}")  # Print the name of the current layer
        if isinstance(module, torch.nn.Linear):
            if name == "lm_head":
                # Keep lm_head as a regular Linear layer with frozen weights
                print(f"Keeping layer {name} as a regular Linear layer with frozen weights")
                setattr(model, name, module)
                module.weight.requires_grad = False
            else:
                if use_ternary_layer:
                    ternary_linear = TernaryLinear(module.in_features, module.out_features).to(device)
                    ternary_linear.weight.data = module.weight.data.to(device)
                    # Ternary layers are trainable by default
                    ternary_linear.weight.requires_grad = True
                    setattr(model, name, ternary_linear)
                else:
                    # Keep the original module
                    setattr(model, name, module)
                
                # Freeze weights by default
                module.weight.requires_grad = False
                
                # Check if the layer name contains any of the specified strings
                if any(substr in name for substr in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mlp"]):
                    print(f"Condition met for layer {name}")  # Add this line
                    module.weight.requires_grad = True
                    print(f"Adding layer {name} for MLP/Attention")
        else:
            quantize_model(module, use_ternary_layer)
    return model


# Set the boolean flag to control whether to use the ternary layer or not
use_ternary_layer = True

# Quantize the target model
quantized_target_model = quantize_model(target_model, use_ternary_layer)

# Training loop
start_time = time.time()
current_sample = 0
for epoch in range(num_epochs):
    print(f"Initial Learning Rate: {scheduler.get_last_lr()[0]}")

    for batch_start in range(0, num_samples_per_epoch, batch_size):
        # Generate random token IDs for the batch (same for source and target models)
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, context_size), device=device)

        # Get the logits from the source model
        with torch.no_grad():
            source_outputs = source_model(input_ids=input_ids)
            source_logits = source_outputs.logits

        # Get the logits from the target model
        target_outputs = target_model(input_ids=input_ids, output_hidden_states=True)
        target_logits = target_outputs.logits

        # Compute loss
        loss = torch.mean((target_logits.view(-1, target_logits.size(-1)) - source_logits.view(-1, source_logits.size(-1))) ** 2)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_sample += batch_size

        # Print loss statistics every 1 second
        current_time = time.time()
        if current_time - start_time >= 1:
            print(f"Epoch: {epoch+1}, Batch: {batch_start//batch_size}, Loss: {loss.item()}, Learning Rate: {scheduler.get_last_lr()[0]}")
            start_time = current_time

print("Training completed.")
