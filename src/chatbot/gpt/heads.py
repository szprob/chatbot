from typing import Dict, Optional, Tuple

import torch
from block.module_utils import PreTrainedModule
from torch import nn

from .gpt import GPT2


class GPT2LMHeadModel(PreTrainedModule, nn.Module):
    """Modeling gpt2 lm head.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        PreTrainedModule.__init__(self)
        nn.Module.__init__(self)

        if config is None:
            config = {}
        self.config = config

        self.model = GPT2(self.config)
        self.lm_head = nn.Linear(
            self.config.get("n_embd", 768),
            self.config.get("vocab_size", 50257),
            bias=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.model(inputs, segment_ids=segment_ids, attention_mask=attention_mask)
        lm_logits = self.lm_head(x)
        return lm_logits
