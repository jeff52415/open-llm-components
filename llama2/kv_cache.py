# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# source: torchtune
# https://github.com/pytorch/torchtune/blob/main/torchtune/modules/kv_cache.py


from typing import Tuple

import torch
from torch import Tensor, nn


class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Update KV cache and return the updated cache.

        Args:
            input_pos (Tensor): Current position tensor with shape [S]
            k_val (Tensor): Current key tensor with shape [B, H, S, D]
            v_val (Tensor): Current value tensor with shape [B, H, S, D]

        Raises:
            ValueError: if ``input_pos`` is longer than the maximum sequence length

        Returns:
            Tuple[Tensor, Tensor]: Updated KV cache with key first
        """
        assert (
            input_pos.shape[0] == k_val.shape[2]
        ), "Input position shape and key value shape must match."

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
