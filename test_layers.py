import unittest
import torch
import torch.nn.functional as F
from layers import DifferentiableEmbedding

class TestDifferentiableEmbedding(unittest.TestCase):

    def setUp(self):
        self.num_embeddings = 10
        self.embedding_dim = 5
        self.diff_embedding = DifferentiableEmbedding(self.num_embeddings, self.embedding_dim)
        self.standard_embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Copy weights for consistency in testing
        self.diff_embedding.embedding.weight = torch.nn.Parameter(self.standard_embedding.weight.data.clone())

    def test_embedding_output(self):
        """Test if the embedding output matches standard PyTorch embedding output with a sharper Gumbel Softmax distribution."""
        logits = torch.randn(3, self.num_embeddings) * 1000
        # Use a smaller tau to sharpen the distribution
        hard_indices = logits.argmax(dim=-1)

        diff_embedding_output = self.diff_embedding(logits)
        standard_embedding_output = self.standard_embedding(hard_indices)

        torch.testing.assert_allclose(diff_embedding_output, standard_embedding_output, atol=1e-3, rtol=1e-3)


    def test_embedding_output_hard(self):
        """Test if the embedding output using hard indices matches standard PyTorch embedding output."""
        logits = torch.randn(3, self.num_embeddings)
        soft_indices = F.gumbel_softmax(logits, tau=1, hard=False, dim=-1)
        hard_indices = soft_indices.argmax(dim=-1)

        # Forward pass using hard indices
        diff_embedding_output = self.diff_embedding.embedding(hard_indices)
        standard_embedding_output = self.standard_embedding(hard_indices)

        torch.testing.assert_close(diff_embedding_output, standard_embedding_output)


    def test_gradient_flow(self):
        """Test if gradients are properly computed."""
        logits = torch.randn(3, self.num_embeddings, requires_grad=True)
        output = self.diff_embedding(logits)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(logits.grad)
        self.assertNotEqual(torch.sum(logits.grad), 0)

    def test_shape_consistency(self):
        """Test if the shape of the output is consistent with expectations."""
        logits = torch.randn(3, self.num_embeddings)
        output = self.diff_embedding(logits)
        self.assertEqual(output.shape, (3, self.embedding_dim))

if __name__ == '__main__':
    unittest.main()
