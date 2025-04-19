# femtoGPT
An implementation of GPT-2(124 m) with not a lot of code

# Requirements
Python 3.9+
```
pip install -r requiremnts.txt
```

# Training
```python fineweb.py``` will add fineweb 10B into a local dataset directory.
```python femtoGPT.py``` will train, log, and output validations.

# Tokenization
Tiktoken currently but planning to replace with own implementation.
# Pre-loaded weights
None yet
# References
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3](https://arxiv.org/abs/2005.14165)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [GELU](https://arxiv.org/abs/1606.08415)
- [HFT Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

