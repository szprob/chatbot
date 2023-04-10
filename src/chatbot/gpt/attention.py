# import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class GPT2Attention(nn.Module):
    """Modeling GPT2Attention.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        if config is None:
            config = {}
        self.config = config

        n_positions = config.get("n_positions", 1024)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((n_positions, n_positions), dtype=torch.uint8)).view(
                1, 1, n_positions, n_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.get("n_embd", 768)
        self.num_heads = config.get("n_head", 12)
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("`embed_dim` must be divisible by num_heads.")

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.get("attn_pdrop", 0.1))
        self.resid_dropout = nn.Dropout(config.get("resid_dropout", 0.1))

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].to(torch.bool)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor,
        # otherwise we get error:
        # `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device,
        # otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision)
        # -- No-Op otherwise
        # attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """gpt attention forward .

        Args:
            hidden_states (Optional[Tuple[torch.FloatTensor]]):
                hideden states of block .
            layer_past (Optional[Tuple[torch.Tensor]], optional):
                not used yet .
                Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional):
                pad mask .
                Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional):
                not used yet.
                Defaults to None.
            use_cache (Optional[bool], optional):
                not used yet.
                Defaults to False.
        Returns:
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
                attention result .
        """

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, _ = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present, (attentions)
