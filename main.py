import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            attention output and attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.combine_heads(attention_output) 
        output = self.W_o(output)
        
        return output, attention_weights

batch_size = 32
seq_len = 50
d_model = 512
num_heads = 8

attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

x = torch.randn(batch_size, seq_len, d_model)

mask = torch.ones(batch_size, 1, seq_len, seq_len)

output, attention_weights = attention(x, mask)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

attention_head = attention_weights[0, 0].detach().numpy()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(attention_head, cmap='viridis')
plt.title('Attention Weights for Head 0')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()

weights_sum = attention_weights.sum(dim=-1)
print(f"Attention weights sum (should be close to 1.0):\n{weights_sum[0, 0, :5]}")

print(f"Output mean and std:\nMean: {output.mean().item():.3f}\nStd: {output.std().item():.3f}")