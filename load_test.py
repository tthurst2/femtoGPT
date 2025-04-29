import torch
import os
from femtoGPT import GPT, GPTConfig, Block, SelfAttention, MLP
import tiktoken

# model = GPT(GPTConfig(vocab_size=50304))


model = torch.load("./log/step_19000_weights.pt", weights_only=False)

model.eval()
device = "cuda"
enc = tiktoken.get_encoding("gpt2")
num_return_sequences = 4
max_length = 32
tokens: list[int] = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)

x = model.generate(xgen, 30)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"sample {i}: {decoded}")

print(x)

print(model)
