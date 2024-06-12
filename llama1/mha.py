import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        print(q.shape, k.shape, v.shape)

        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k)
        attn_weights = attn_weights / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )

        output = self.out_proj(attn_output)
        return output


# Example usage
embed_dim = 512
num_heads = 8
seq_len = 1024

model = MultiHeadAttention(embed_dim, num_heads)
x = torch.rand(1, seq_len, embed_dim)
output = model(x)
print(output.shape)
