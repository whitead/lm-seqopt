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