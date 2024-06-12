from typing import Optional, Tuple

import torch
import torch.nn as nn
from kv_cache import KVCache
from rope import RotaryPositionalEmbeddings
from torch import Tensor


class GQAAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_kv_heads, kv_cache: Optional[KVCache] = None
    ):
        super(GQAAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_cache = kv_cache

        assert (
            num_heads % num_kv_heads == 0
        ), "Number of heads must be divisible by number of KV heads"
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_embeddings = RotaryPositionalEmbeddings(self.head_dim)

    def forward(self, x, input_pos: Optional[Tensor] = None):
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), "Input embedding dimension must match layer embedding dimension"

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)

        # Adjust the size of k and v to match the number of heads in q
        k = (
            k.unsqueeze(1)
            .expand(
                batch_size, self.num_groups, self.num_kv_heads, seq_len, self.head_dim
            )
            .reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        )
        v = (
            v.unsqueeze(1)
            .expand(
                batch_size, self.num_groups, self.num_kv_heads, seq_len, self.head_dim
            )
            .reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        )

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        # Apply positional embeddings
        q = self.pos_embeddings(q.transpose(1, 2), input_pos=input_pos)
        k = self.pos_embeddings(k.transpose(1, 2), input_pos=input_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos.reshape(-1), k, v)

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
num_kv_heads = 2
seq_len = 1024
batch_size = 1
kv_cache = KVCache(
    batch_size=batch_size,
    max_seq_len=seq_len,
    num_heads=num_heads,
    head_dim=embed_dim // num_heads,
    dtype=torch.float32,
)
model = GQAAttention(embed_dim, num_heads, num_kv_heads, kv_cache=kv_cache)
x = torch.rand(1, seq_len, embed_dim)
input_pos = torch.arange(seq_len).reshape(batch_size, -1)  # Example input positions
output = model(x, input_pos=input_pos)
