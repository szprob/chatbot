# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""
import json
import os
import shutil
from glob import glob
from itertools import chain
from pathlib import Path

import dataset
import pandas as pd

# import nets
import torch
import torch.distributed as dist
import trainer
from torch import nn
from tqdm import tqdm
from transformers import BloomForCausalLM, BloomTokenizerFast

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # ddp setting
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_config = json.load(open("config/model_config.json"))
    tokenizer = BloomTokenizerFast.from_pretrained(model_config["model_name_or_path"])
    tokenizer.pad_token_id = model_config["pad_token_id"]
    tokenizer.padding_side = model_config["padding_side"]

    # dataset
    # data_path = model_config["data_path"]
    data_path = "/data/home/ze.song/data/corpus/dialogue"
    data_set = dataset.DataSet(
        config=model_config, tokenizer=tokenizer, data_path=data_path
    )
    trainloader = data_set.make_dataloader(batch_size=3)

    # model
    model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-1b1",
    )
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # model.load_state_dict(torch.load('./model.pkl', map_location="cpu"))

    # trainer
    #     imp.reload(trainer)
    t = trainer.Trainer(
        device=device,
        train_loader=trainloader,
        opt_freq=32,
    )
    # t.optim.n_current_steps=148000
    t.train(model=model, file_path="/data/home/ze.song/models/tmp", max_num=1e6)


#     model.load_state_dict(
#         torch.load("./check_points/bert/model_1000000.pkl", map_location="cpu")
#     )
