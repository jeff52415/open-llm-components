import torch
import torch.nn as nn
import torch.nn.functional as F

# reference: https://github.com/ambisinister/mla-experiments/blob/main/modeling/gpt.py
# Use RMSNorm instead of LayerNorm

# reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py


# -------------------------------
# RMSNorm Implementation
# -------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute the RMS value along the last dimension.
        norm_x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * norm_x


# -------------------------------
# Helper Functions for RoPE
# -------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension into two halves and rotates them
    (i.e. concatenates -second half with first half).
    """
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies the rotary positional embedding (RoPE) to x using precomputed cosine and sine schedules.
    """
    return (x * cos) + (rotate_half(x) * sin)


# -------------------------------
# Revised MLA Module with RMSNorm and Pre-Norm
# -------------------------------
class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) with RoPE, using RMSNorm and pre-normalization.

    This module implements a two-stage (LoRA-style) projection for queries and key/value pairs.
    Pre-normalization is applied on the input using RMSNorm, and then the queries and keys are
    split into a “non-RoPE” and a “RoPE” portion. The RoPE part receives rotary positional embeddings.

    Args:
        d_model (int): Input and output feature dimension.
        n_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length (for precomputing RoPE tensors).
        rope_theta (float): Hyperparameter controlling frequency scale for RoPE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 1024,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # Dimension per head

        # Two-stage projection dimensions.
        self.query_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3

        # Within each head, we split features into two parts:
        # one that does not get RoPE and one that does.
        self.qk_no_rope_dim = self.head_dim // 2
        self.qk_rope_dim = self.head_dim // 2

        # Pre-normalization using RMSNorm applied to the input.
        self.pre_norm = RMSNorm(d_model)

        # ----- Query Projections (LoRA-style) -----
        self.w_dq = nn.Parameter(0.01 * torch.randn(d_model, self.query_proj_dim))
        self.w_uq = nn.Parameter(0.01 * torch.randn(self.query_proj_dim, d_model))

        # ----- Key/Value Projections (LoRA-style) -----
        # Note: The down-projection produces extra dimensions to accommodate the RoPE part.
        self.w_dkv = nn.Parameter(
            0.01 * torch.randn(d_model, self.kv_proj_dim + self.qk_rope_dim)
        )
        self.w_ukv = nn.Parameter(
            0.01
            * torch.randn(self.kv_proj_dim, d_model + (n_heads * self.qk_no_rope_dim))
        )

        # ----- Output Projection -----
        self.w_o = nn.Parameter(0.01 * torch.randn(d_model, d_model))

        # ----- Precompute RoPE Cosine and Sine Caches -----
        self.max_len = max_len
        self.rope_theta = rope_theta
        # Compute frequencies for half of the head dimension.
        freqs = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        positions = torch.arange(max_len).float()  # (max_len,)
        angles = torch.outer(positions, freqs)  # (max_len, head_dim/2)
        # Cache cosine and sine values (with extra dimensions for broadcasting).
        cos_cache = angles.cos()[None, None, :, :]  # shape: (1, 1, max_len, head_dim/2)
        sin_cache = angles.sin()[None, None, :, :]
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)

    def forward(
        self, x: torch.Tensor, kv_cache: torch.Tensor = None, past_length: int = 0
    ):
        """
        Forward pass for MLA.

        Args:
            x: Input tensor of shape (B, S, d_model).
            kv_cache: Optional cached key/value projection tensor for autoregressive decoding.
            past_length: Number of tokens processed previously (used for RoPE indexing).

        Returns:
            output: Tensor of shape (B, S, d_model).
            updated_kv_cache: Updated key/value projection tensor.
        """
        B, S, _ = x.size()

        # ----- Pre-Normalization -----
        # Apply RMSNorm to the input.
        x_norm = self.pre_norm(x)

        # ----- Query Path -----
        # Two-stage projection for queries.
        q_down = x_norm @ self.w_dq  # (B, S, query_proj_dim)
        q_up = q_down @ self.w_uq  # (B, S, d_model)
        q = q_up.view(B, S, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, S, head_dim)
        # Split each head into a non-RoPE and a RoPE part.
        q_no_rope, q_rope = torch.split(
            q, [self.qk_no_rope_dim, self.qk_rope_dim], dim=-1
        )
        # Apply RoPE to the designated part of queries.
        cos_q = self.cos_cache[
            :, :, past_length : past_length + S, : self.qk_rope_dim // 2
        ].repeat(1, 1, 1, 2)
        sin_q = self.sin_cache[
            :, :, past_length : past_length + S, : self.qk_rope_dim // 2
        ].repeat(1, 1, 1, 2)
        q_rope = apply_rope_to_tensor(q_rope, cos_q, sin_q)

        # ----- Key/Value Path -----
        # Use pre-normalized input for key/value projections.
        if kv_cache is None:
            kv_down = x_norm @ self.w_dkv  # (B, S, kv_proj_dim + qk_rope_dim)
            kv_no_rope, k_rope = torch.split(
                kv_down, [self.kv_proj_dim, self.qk_rope_dim], dim=-1
            )
        else:
            new_kv = x_norm @ self.w_dkv  # (B, S, kv_proj_dim + qk_rope_dim)
            kv_down = torch.cat([kv_cache, new_kv], dim=1)
            new_no_rope, new_rope = torch.split(
                new_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1
            )
            old_no_rope, old_rope = torch.split(
                kv_cache, [self.kv_proj_dim, self.qk_rope_dim], dim=-1
            )
            kv_no_rope = torch.cat([old_no_rope, new_no_rope], dim=1)
            k_rope = torch.cat([old_rope, new_rope], dim=1)

        # Up-project the non-RoPE portion to form keys and values.
        kv_up = (
            kv_no_rope @ self.w_ukv
        )  # (B, s_full, d_model + n_heads * qk_no_rope_dim)
        kv_up = kv_up.view(
            B, -1, self.n_heads, self.head_dim + self.qk_no_rope_dim
        ).transpose(
            1, 2
        )  # (B, n_heads, S_full, head_dim + qk_no_rope_dim)
        key, value = torch.split(kv_up, [self.qk_no_rope_dim, self.head_dim], dim=-1)
        s_full = key.size(2)

        # Process the RoPE part for keys.
        k_rope = k_rope.view(B, -1, 1, self.qk_rope_dim).transpose(
            1, 2
        )  # (B, 1, s_full, qk_rope_dim)
        cos_k = self.cos_cache[:, :, :s_full, : self.qk_rope_dim // 2].repeat(
            1, 1, 1, 2
        )
        sin_k = self.sin_cache[:, :, :s_full, : self.qk_rope_dim // 2].repeat(
            1, 1, 1, 2
        )
        k_rope = apply_rope_to_tensor(k_rope, cos_k, sin_k)
        # Duplicate the RoPE keys for each head.
        k_rope = k_rope.repeat(1, self.n_heads, 1, 1)

        # Combine the non-RoPE and RoPE parts for the final queries and keys.
        final_q = torch.cat([q_no_rope, q_rope], dim=-1)  # (B, n_heads, S, head_dim)
        final_k = torch.cat([key, k_rope], dim=-1)  # (B, n_heads, s_full, head_dim)

        # ----- Attention Computation -----
        # Create a causal (lower-triangular) mask so that each token attends only to previous tokens.
        mask = torch.ones((S, s_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        attn_mask = mask[None, None, :, :].bool()  # (1, 1, S, s_full)
        # Compute scaled dot-product attention.
        attn_out = F.scaled_dot_product_attention(
            final_q, final_k, value, attn_mask=attn_mask
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.d_model)
        # Final output projection.
        output = attn_out @ self.w_o.T

        return output, kv_down


# ---- Example Integration into a Transformer Encoder Layer ----
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        # The MLA layer already includes internal pre-normalization.
        self.attention = MLA(d_model, n_heads, max_len=max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        # In a pre-norm design, we also apply RMSNorm after adding residuals.
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, kv_cache: torch.Tensor = None):
        # Self-attention with residual connection.
        x = self.norm1(x)
        attn_out, updated_kv = self.attention(x, kv_cache=kv_cache)
        x = x + attn_out
        # Feed-forward network with residual connection.
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        return x, updated_kv


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    d_model = 512
    n_heads = 8
    max_seq_len = 1024

    # Instantiate the MLA module.
    mla = MLA(d_model, n_heads, max_len=max_seq_len)
    x = torch.randn(4, 128, d_model)
    output, kv_cache = mla(x)
    print("MLA output shape:", output.shape)
    print("KV cache shape:", kv_cache.shape)

    encoder_layer = TransformerEncoderLayer(d_model, n_heads, max_seq_len)
    encoder_out, encoder_kv = encoder_layer(x)
    print("Encoder output shape:", encoder_out.shape)