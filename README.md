# Sequence Optimization

Messing around with sequence optimization in PyTorch

```python
# Example Usage
num_embeddings = 10
embedding_dim = 5

# Create a DifferentiableEmbedding instance
diff_embedding = DifferentiableEmbedding(num_embeddings, embedding_dim)

# Example input - using Gumbel Softmax to get indices
logits = torch.randn(3, num_embeddings, requires_grad=True)

# Forward pass
output = diff_embedding(logits)

# Compute gradients
loss = output.sum()
loss.backward()

print(logits.grad)

```

## Reference Implementations

Wrapper for PyTroch modules to output gradients of input tokens:

https://github.com/QData/TextAttack/blob/57bc36cc622e8c1a993d728066cb9f42cdec217d/textattack/models/wrappers/pytorch_model_wrapper.py#L50

Implementation for one-hot encoding multiplication by embedding matrix in jax:
https://github.com/ur-whitelab/wazy/blob/529eac8b473b9f17d6ff7824230b8fcf35fb99c3/wazy/utils.py#L89
