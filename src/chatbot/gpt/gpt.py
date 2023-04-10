import math
import os
from typing import Dict, Optional

import torch
from block.module_utils import PreTrainedModule
from torch import nn
from torch.nn import functional as F

from .block import GPT2Block
from .embedding import Embeddings


class GPT2(PreTrainedModule, nn.Module):

    """Modeling gpt2.

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

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self.vocab_size = config.get("vocab_size", 50257)
        self.embd_pdrop = config.get("embd_pdrop", 0.1)
        self.n_embd = config.get("n_embd", 768)
        self.n_head = config.get("n_head", 12)
        self.n_positions = config.get("n_positions", 1024)
        self.n_layer = config.get("n_layer", 12)
        self.attn_pdrop = config.get("attn_pdrop", 0.1)
        self.resid_dropout = config.get("resid_dropout", 0.1)
        self.n_inner = config.get("n_inner", 768 * 4)
        self.layer_norm_epsilon = config.get("layer_norm_epsilon", 1e-5)
        self.pad_idx = config.get("pad_idx", 0)
        self.dtype = config.get("dtype", torch.float32)
        self.segment_size = config.get("segment_size", 3)

        self.e = Embeddings(config)
        self.h = nn.ModuleList([GPT2Block(config) for i in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

        if "lm_head" in config:

            self.lm_head = nn.Linear(
                config.get("n_embd", 768), config.get("vocab_size", 50257), bias=False
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version
            # which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization -->
                # There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * self.n_layer)))

    def forward(
        self,
        inputs: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward function of bert.

        Args:
            inputs (torch.Tensor):
                The index of words. shape:(b,l)
            segment_ids (Optional[torch.Tensor]):
                The index of segments. shape:(b,l)
                defaults to None .
            attention_mask: Optional[torch.FloatTensor] :
                attention mask .
                defaults to None.
        Returns:
            torch.Tensor:
                GPT result. shape:(b,l,d)
        """

        if attention_mask is None:
            # (b,l)
            attention_mask = (inputs == self.pad_idx).to(dtype=self.dtype)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask * torch.finfo(self.dtype).min
            # (b,1,1,l)

        # (b,l)
        x = self.e(inputs, segment_ids)
        # (b , l ,d)
        for block in self.h:
            outputs = block(x, attention_mask=attention_mask)
            x = outputs[0]

        x = self.ln_f(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        max_num: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate sequence.

        Take a conditioning sequence of `inputs` .
        Complete the sequence `max_num` times.
        Feeding the predictions back into the model each time.
        Make sure to be in model.eval() mode of operation for this.

        Args:
            inputs (torch.Tensor):
                The index of words. shape:(b,l)
            max_num (int, optional):
                max num of words model predict .
                Defaults to 1.
            temperature (float, optional):
                scale logits.
                Defaults to 1.0.
            do_sample (bool, optional):
                sample from the distribution
                Defaults to False.
            top_k (Optional[int], optional):
                Defaults to None.
        Returns:
            torch.Tensor:
                generate result. shape:(b,l,d)

        """
        x = inputs
        for _ in range(max_num):
            # if the sequence context is growing too long we must crop it at n_positions
            x_cond = x if x.size(1) <= self.n_positions else x[:, -self.n_positions :]

            # forward the model to get the logits for the index in the sequence
            hidden_states = self.forward(x_cond)
            logits = self.lm_head(hidden_states)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution
            # or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1)

        return x

    def load(self, model: str) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                A string, the path of a pretrained model.

        Raises:
            ValueError: str model should be a path!
        """

        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "gpt2")
                if os.path.exists(dir):
                    pass
                else:
                    os.mkdir(dir)
                self._unzip2dir(model, dir)
                self._load_from_dir(dir)
            else:
                raise ValueError("""str model should be a path!""")

        else:
            raise ValueError("""str model should be a path!""")

    def _load_from_dir(self, model_dir: str) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """
        model_files = os.listdir(model_dir)

        # config
        if "config.pkl" not in model_files:
            raise FileNotFoundError("""config should in model dir!""")

        config = self._load_pkl(os.path.join(model_dir, "config.pkl"))
        self.config = config

        # model
        if "model.pkl" not in model_files:
            raise FileNotFoundError("""model should in model dir!""")

        self.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pkl"), map_location="cpu")
        )
        self.eval()
