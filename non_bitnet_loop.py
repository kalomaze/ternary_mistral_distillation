import os
import time
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Specify the paths for the source and target models
source_model_path = "/home/kalomaze/Downloads/mergekit/mergekit/onelayer7b"
target_model_path = "/home/kalomaze/Downloads/mergekit/mergekit/onelayer7b_dupe"

# Load the source model
source_model = AutoModelForCausalLM.from_pretrained(source_model_path)
source_model.eval()

# Load the target model tensors
target_model_file = os.path.join(target_model_path, "model.safetensors")
target_tensors = load_file(target_model_file)

# Create a config for the target model
config = AutoConfig.from_pretrained(source_model_path)

# Set the use_qat flag to control quantization-aware training
use_qat = False

# Create the target model from the loaded tensors and config
target_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=target_tensors,
    config=config,
    quantization_config={"use_quantization_aware_training": use_qat},  # Enable QAT if use_qat is True
)
target_model.train()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(source_model_path)

# Set the device to use (e.g., "cuda" for GPU, "cpu" for CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
source_model.to(device)
target_model.to(device)

# Set the training hyperparameters
num_epochs = 1
batch_size = 1
learning_rate = 0.01  # Define the starting learning rate
context_size = 4096

# Define the optimizer
optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate)

# Define the learning rate scheduler
def lr_lambda(current_step: int):
    num_warmup_steps = 100
    num_training_steps = num_epochs * (tokenizer.vocab_size // batch_size)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)) * learning_rate
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

# Training loop
start_time = time.time()
global_step = 0
for epoch in range(num_epochs):
    for batch_start in range(0, tokenizer.vocab_size, batch_size):
        # Generate random token IDs for the batch
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, context_size), device=device)

        # Get the logits from the source model
        with torch.no_grad():
            source_outputs = source_model(input_ids=input_ids)
            source_logits = source_outputs.logits

        # Get the logits from the target model
        if use_qat:
            # Use ternary precision forward pass
            target_outputs = target_model(input_ids=input_ids, use_ternary_forward=True)
        else:
            # Use full precision forward pass
            target_outputs = target_model(input_ids=input_ids)
        target_logits = target_outputs.logits

        # Compute the cross-entropy loss
        loss = torch.nn.functional.mse_loss(target_logits.view(-1, target_logits.size(-1)), source_logits.view(-1, source_logits.size(-1)))

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1

        # Print loss statistics every 1 second
        current_time = time.time()
        if current_time - start_time >= 1:
            print(f"Epoch: {epoch+1}, Batch: {batch_start//batch_size}, Loss: {loss.item()}, Learning Rate: {scheduler.get_last_lr()[0]}")
            start_time = current_time

print("Training completed.")
