import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .attention import GPT2Attention


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function
    currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class GPT2MLP(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        if config is None:
            config = {}
        self.config = config

        hidden_size = config.get("n_embd", 768)
        inner_dim = config.get("n_inner", 4 * hidden_size)

        self.c_fc = nn.Linear(hidden_size, inner_dim)
        self.c_proj = nn.Linear(inner_dim, hidden_size)
        self.act = NewGELU()
        self.dropout = nn.Dropout(config.get("resid_dropout", 0.1))

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    """Modeling GPT2Block.

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

        hidden_size = config.get("n_embd", 768)

        self.ln_1 = nn.LayerNorm(
            hidden_size, eps=config.get("layer_norm_epsilon", 1e-5)
        )
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(
            hidden_size, eps=config.get("layer_norm_epsilon", 1e-5)
        )

        self.mlp = GPT2MLP(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        """gpt block forward function .

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
           torch.Tensor:
                block result .
        """

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
