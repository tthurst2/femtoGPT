import torch
from femtoGPT import GPT, GPTConfig, Block, SelfAttention, MLP
import tiktoken

model = torch.load("./log/step_19000_weights.pt", weights_only=False)

model.eval()
device = "cuda"
enc = tiktoken.get_encoding("gpt2")
num_return_sequences = 4
max_length = 100
tokens = enc.encode("What is the name of the largest ocean?")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)

x = model.generate(xgen, 100)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"sample {i}: {decoded}")
