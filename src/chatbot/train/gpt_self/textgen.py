# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""


import pickle

import torch
from block.bricks.models.gpt.heads import GPT2LMHeadModel
from block.bricks.tokenizations.bert.tokenization import Tokenizer
from torch.nn import functional as F

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = torch.device(device)
n_device = torch.cuda.device_count()
torch.backends.cudnn.is_available()
torch.backends.cudnn.version()
torch.set_default_tensor_type(torch.FloatTensor)
torch.cuda.set_device(0)


def pad_list(text, maxlen=1024, pad=0):
    x = [pad] * maxlen
    length = len(text)
    if length > 0:
        if length <= maxlen:
            x[-length:] = text
        else:
            x = text[-maxlen:]
    return x


vocab_path = (
    "/data/home/ze.song/git/block/src/block/bricks/train/gpt/model_files/vocab.pkl"
)


class Bot:
    def __init__(
        self,
        model_path="/data/home/ze.song/models/gpt/model_0_20000.pkl",
        vocab_path=vocab_path,
        config={
            "vocab_size": 13317,
            "embd_pdrop": 0.1,
            "n_embd": 1536,
            "n_head": 24,
            "n_positions": 320,
            "n_layer": 18,
            "attn_pdrop": 0.1,
            "resid_dropout": 0.1,
            "n_inner": 1536 * 4,
            "layer_norm_epsilon": 1e-5,
            "pad_idx": 0,
            "dtype": torch.float32,
            "segment_size": 3,
        },
    ):
        self.config = config
        model = GPT2LMHeadModel(config=config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.cuda()
        model.eval()
        self.model = model
        with open(
            vocab_path,
            "rb",
        ) as f:
            vocab = pickle.load(f)
        # tokenizer
        tok = Tokenizer(vocab=vocab)
        tok.init_model()
        self.tok = tok

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        max_num: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k=None,
        top_p=0.6,
        min_tokens_to_keep=5,
    ) -> torch.Tensor:
        x = inputs.to(self.model.lm_head.weight.device)
        for idx, _ in enumerate(range(max_num)):
            # if the sequence context is growing too long
            # we must crop it at n_positions
            x_cond = (
                x
                if x.size(1) <= self.config["n_positions"]
                else x[:, -self.config["n_positions"] :]
            )
            # forward the model to get the logits
            # for the index in the sequence
            hidden_states = self.model.model.forward(inputs=x_cond)
            logits = self.model.lm_head(hidden_states[:, -1, :])

            # pluck the logits at the final step and
            # scale by desired temperature
            logits = logits / temperature
            logits = logits.cpu()

            # optionally crop the logits to only the top k options
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float(-1e8)

            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                # (token with 0 are kept)
                sorted_indices_to_remove = cumulative_probs > top_p
                if min_tokens_to_keep > 1:
                    # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1
                    # because we add the first one below)
                    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
                # Shift the indices to the right to keep also the first token
                # above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float(-1e8)

            # # cls
            # logits[0][100] = float(-1e8)
            # logits[0][101] = float(-1e8)
            # logits[0][103] = float(-1e8)
            # logits[0][104] = float(-1e8)
            # logits[0][105] = float(-1e8)

            # # first sep
            # if idx == 0:
            #     logits[0][102] = logits[0][102] / 10

            # apply softmax to convert logits to (normalized) probabilities
            logits = logits.to(self.model.lm_head.weight.device)
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution
            # or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1)
            # if idx_next[0][0].item() == 102:
            #     break
        if idx_next[0][0].item() != 102:
            x = torch.cat(
                (x, torch.tensor([[102]]).long().to(self.model.lm_head.weight.device)),
                dim=1,
            )
        return x

    @torch.no_grad()
    def gen(
        self,
        input="",
        max_num=100,
        top_k=None,
        top_p=0.5,
        min_tokens_to_keep=5,
        temperature=1.0,
    ):

        words = []
        text = self.tok.tokenize(input)
        words.extend(text)
        words.append(self.tok.sep_token)
        ints = self.tok.convert_tokens_to_id(words)

        inputs = torch.tensor(ints).long().view(1, -1)

        res = self.generate(
            inputs,
            max_num=max_num,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep,
            temperature=temperature,
        )

        xx = res[0].tolist()
        xx = [i for i in xx if i != 0]
        result = [self.tok._ids_to_tokens[i] for i in xx]
        ret = " ".join("".join(result).split("[SEP]")[:-1])
        self.his = "".join(result).split("[SEP]")[:-1]
        return ret


class Adder:
    def __init__(
        self,
        model_path="/data/home/ze.song/models/gpt/model_0_20000.pkl",
        vocab_path=vocab_path,
        config={
            "vocab_size": 178,
            "embd_pdrop": 0.1,
            "n_embd": 768,
            "n_head": 12,
            "n_positions": 20,
            "n_layer": 12,
            "attn_pdrop": 0.1,
            "resid_dropout": 0.1,
            "n_inner": 768 * 4,
            "layer_norm_epsilon": 1e-5,
            "pad_idx": 0,
            "dtype": torch.float32,
            "segment_size": 3,
        },
    ):
        self.config = config
        model = GPT2LMHeadModel(config=config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.cuda()
        model.eval()
        self.model = model
        with open(
            vocab_path,
            "rb",
        ) as f:
            vocab = pickle.load(f)
        voc = {}
        for k, v in vocab.items():
            if v >= 178:
                break
            voc[k] = v
        # tokenizer
        tok = Tokenizer(vocab=voc)
        tok.init_model()
        self.tok = tok

        self.init()

    def init(self):
        self.x = None
        self.his = []

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        max_num: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k=None,
        top_p=0.6,
        min_tokens_to_keep=5,
    ) -> torch.Tensor:
        x = inputs.to(self.model.lm_head.weight.device)
        for idx, _ in enumerate(range(max_num)):
            # if the sequence context is growing too long
            # we must crop it at n_positions
            x_cond = (
                x
                if x.size(1) <= self.config["n_positions"]
                else x[:, -self.config["n_positions"] :]
            )
            # forward the model to get the logits
            # for the index in the sequence
            hidden_states = self.model.model.forward(inputs=x_cond)
            logits = self.model.lm_head(hidden_states)
            logits = logits[:, -1, :]
            # pluck the logits at the final step and
            # scale by desired temperature
            logits = logits / temperature
            logits = logits.cpu()

            # optionally crop the logits to only the top k options
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float(-1e8)

            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                # (token with 0 are kept)
                sorted_indices_to_remove = cumulative_probs > top_p
                if min_tokens_to_keep > 1:
                    # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1
                    # because we add the first one below)
                    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
                # Shift the indices to the right to keep also the first token
                # above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float(-1e8)

            # # cls
            # logits[0][100] = float(-1e8)
            # logits[0][101] = float(-1e8)
            # logits[0][103] = float(-1e8)
            # logits[0][104] = float(-1e8)
            # logits[0][105] = float(-1e8)

            # # first sep
            # if idx == 0:
            #     logits[0][102] = logits[0][102] / 10

            # apply softmax to convert logits to (normalized) probabilities
            logits = logits.to(self.model.lm_head.weight.device)
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution
            # or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1)
            if idx_next[0][0].item() == 102:
                break
        return x

    @torch.no_grad()
    def gen(
        self,
        input="",
        max_num=100,
        top_k=None,
        top_p=0.5,
        min_tokens_to_keep=5,
        temperature=1.0,
    ):

        words = list(input)
        words = [w for w in words if w != " "]
        words = [w for w in words if w != ""]
        if words[-1] != "=":
            words.append("=")
        # words.append(self.tok.sep_token)
        ints = self.tok.convert_tokens_to_id(words)
        inputs = torch.tensor(ints).long().view(1, -1)

        res = self.generate(
            inputs,
            max_num=max_num,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep,
            temperature=temperature,
        )

        xx = res[0].tolist()
        xx = [i for i in xx if i != 0]
        result = [self.tok._ids_to_tokens[i] for i in xx if i != 102]
        ret = "".join(result)
        self.his = [ret.split("=")]
        return ret


if __name__ == "__main__":
    bot = Bot()
