# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""

import argparse
import importlib as imp

# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import os

import dataset

# import nets
import torch
import torch.distributed as dist
import trainer
from torch import nn
from transformers import BartForConditionalGeneration, BertTokenizer

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

    tok = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

    # dataset
    imp.reload(dataset)
    data_set = dataset.DataSet(tokenizer=tok)
    trainloader = data_set.make_dataloader(batch_size=32)

    # model
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
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
        opt_freq=8,
    )
    # t.optim.n_current_steps=148000
    t.train(model=model, file_path="/data/home/ze.song/models/chatbot", max_num=1e6)


#     model.load_state_dict(
#         torch.load("./check_points/bert/model_1000000.pkl", map_location="cpu")
#     )
