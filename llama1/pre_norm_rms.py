import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * norm_x


class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(PreNormTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Pre-normalization
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x, src_mask, src_key_padding_mask):
        x, _ = self.self_attn(
            x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


# Example usage
d_model = 512
nhead = 8
dim_feedforward = 2048
dropout = 0.1

pre_norm_block = PreNormTransformerBlock(d_model, nhead, dim_feedforward, dropout)
src = torch.rand(10, 32, d_model)  # (sequence length, batch size, d_model)
output = pre_norm_block(src)

print(output.shape)  # Should output: torch.Size([10, 32, 512])
