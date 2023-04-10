from typing import Dict, Optional

import torch
from torch import nn


class Embeddings(nn.Module):
    """Embedding layer for gpt2.

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

        self.vocab_size = config.get("vocab_size", 50257)
        self.n_positions = config.get("n_positions", 1024)
        self.embed_dim = config.get("n_embd", 768)
        self.pad_idx = config.get("pad_idx", 0)
        self.embd_pdrop = config.get("embd_pdrop", 0.1)
        self.segment_size = config.get("segment_size", 3)

        self.wte = nn.Embedding(
            self.vocab_size, self.embed_dim, padding_idx=self.pad_idx
        )
        self.wpe = nn.Embedding(self.n_positions, self.embed_dim)

        self.wse = nn.Embedding(self.segment_size, self.embed_dim)

        self.drop = nn.Dropout(self.embd_pdrop)

    def forward(
        self, token_ids: torch.Tensor, segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of embedding.

        Args:
            token_ids (torch.Tensor):
                The index of words. shape:(b,l)
            segment_ids (Optional[torch.Tensor]):
                The index of segments. shape:(b,l)
                defaults to None .
        Returns:
            torch.Tensor:
                Embedding result of token_ids. shape:(b,l,d)
        """

        seq_length = token_ids.size(1)

        # pos embed
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=token_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        position_embeds = self.wpe(position_ids)

        # word embed
        inputs_embeds = self.wte(token_ids)

        # seg embed
        if segment_ids is None:
            segment_ids = torch.zeros_like(
                token_ids, dtype=torch.long, device=token_ids.device
            )
        segment_embeddings = self.wse(segment_ids)

        # sum of embeddings
        hidden_states = inputs_embeds + position_embeds + segment_embeddings
        hidden_states = self.drop(hidden_states)
        return hidden_states
