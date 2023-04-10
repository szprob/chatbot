# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

for ordinal classfication
"""

import argparse
import importlib as imp

# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import os

import dataset
import nets
import torch
import torch.distributed as dist
import trainer
from sprite import BertTokenizer
from torch import nn

# config
config = {
    "vocab_size": 50000,
    "maxlen": 384,
    "hidden_size": 256,
    "n_layers": 4,
    "num_heads": 8,
    "pad_idx": 0,
}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # ddp setting
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument(
        "-g", "--gpus", default=3, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tok = BertTokenizer()
    tok.from_pretrained()

    # dataset
    imp.reload(dataset)
    data_set = dataset.DataSet(config=config, tokenizer=tok)
    trainloader = data_set.make_dataloader(batch_size=128)

    # model
    model = nets.Classifier(config=config)
    model.bert.from_pretrained("bert_16m")
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # model.load_state_dict(torch.load('./model.pkl', map_location="cpu"))

    # trainer
    #     imp.reload(trainer)
    t = trainer.Trainer(device=device, train_loader=trainloader, opt_freq=1)
    # t.optim.n_current_steps=148000
    t.train(model=model, file_path="/data/home/ze.song/models/sa2", max_num=1e5)


#     model.load_state_dict(
#         torch.load("./check_points/bert/model_1000000.pkl", map_location="cpu")
#     )
