# Per-Layer Distillation Attempts
This is a hacky personal trainer script I created to (hopefully) get ternary weight distillation working, using the Mistral architecture.

The idea is to do transfer learning one layer at a time, by distilling each layer of Mistral 7b with the proposed ternary weight values of BitNet 1.58b (a quantization-aware training technique that has seen a lot of hype recently).
