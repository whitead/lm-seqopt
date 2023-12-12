import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DifferentiableEmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, embedding_matrix, hard_indices, soft_indices):
        ctx.save_for_backward(embedding_matrix, soft_indices)
        return F.embedding(hard_indices, embedding_matrix)

    @staticmethod
    def backward(ctx, grad_output):
        embedding_matrix, soft_indices = ctx.saved_tensors
        grad_embedding_matrix = grad_soft_indices = None

        if ctx.needs_input_grad[0]:
            # Gradient with respect to the embedding matrix
            grad_embedding_matrix = torch.matmul(soft_indices.transpose(0, 1), grad_output)

        if ctx.needs_input_grad[2]:
            # Gradient with respect to the soft indices
            grad_soft_indices = torch.matmul(grad_output, embedding_matrix.transpose(0, 1))

        return grad_embedding_matrix, None, grad_soft_indices

class DifferentiableEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, tau=1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.tau = tau

    def forward(self, logits):
        soft_indices = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        hard_indices = soft_indices.argmax(dim=-1)
        return DifferentiableEmbeddingFunction.apply(self.embedding.weight, hard_indices, soft_indices)

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
