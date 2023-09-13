# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""


import json
import shutil

# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import torch
from huggingface_hub import HfApi
from torch.amp import autocast
from transformers import (
    BloomForSequenceClassification,
    BloomTokenizerFast,
)


@torch.no_grad()
def rank(x, tokenizer, model, gpu=True):
    inputs = tokenizer.bos_token + x
    input_ids = tokenizer.encode(inputs, return_tensors="pt")
    if gpu:
        input_ids = input_ids.to(device)
    logits = model(input_ids, return_dict=True)["logits"]
    print(x)
    print(logits.item())


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model_config = json.load(open("config/reward_model_config.json"))
    # load_in_8bit
    tokenizer = BloomTokenizerFast.from_pretrained(
        "/data/home/ze.song/models/chatbot/reward_model"
    )
    model = BloomForSequenceClassification.from_pretrained(
        "/data/home/ze.song/models/chatbot/reward_model",
    )
    model.score = torch.nn.Linear(in_features=1536, out_features=1, bias=False)
    model.config.num_labels = 1
    model.score.weight

    tokenizer = BloomTokenizerFast.from_pretrained(
        "bigscience/bloom-1b1", cache_dir="/data/home/ze.song/models/tmp"
    )
    model = BloomForSequenceClassification.from_pretrained(
        "bigscience/bloom-1b1", cache_dir="/data/home/ze.song/models/tmp"
    )

    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    path = "/data/home/ze.song/models/tmp2"
    with open(f"{path}/latest", "r") as file:
        name = file.read()
    states_path = f"{path}/{name}/mp_rank_00_model_states.pt"
    print(states_path)
    model.load_state_dict(torch.load(states_path, map_location="cpu")["module"])

    model.half()
    model.to(device)
    model.eval()
    tokenizer.save_pretrained("/data/home/ze.song/models/chatbot/tmp")
    model.save_pretrained("/data/home/ze.song/models/chatbot/tmp")
    shutil.move(
        "/data/home/ze.song/models/chatbot/tmp/pytorch_model.bin",
        "/data/home/ze.song/models/chatbot/reward_model/pytorch_model.bin",
    )

    x = "Human: 帮我判断下面这个短语是不是恶意的 : \n 我操你妈 \n\nAssistant: 恶意"
    x2 = "Human: 帮我判断下面这个短语是不是恶意的 : \n 我操你妈 \n\nAssistant: 不是恶意"

    x = "Human: Human: 把下面这句话翻译成中文 : Main hand : \n\nAssistant: 主手"
    x2 = "Human: Human: 把下面这句话翻译成中文 : Main hand : \n\nAssistant: 右手"

    rank(x, tokenizer, model, gpu=True)
    rank(x2, tokenizer, model, gpu=True)
