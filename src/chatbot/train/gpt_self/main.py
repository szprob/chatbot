# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""

import importlib as imp
import os
import pickle

import dataset
import torch
import trainer
from block.bricks.models.gpt.heads import GPT2LMHeadModel

from chatbot.tokenization.tokenization import Tokenizer

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = torch.device(device)
n_device = torch.cuda.device_count()
torch.backends.cudnn.is_available()
torch.backends.cudnn.version()
torch.set_default_tensor_type(torch.FloatTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
torch.cuda.set_device(3)


# config
config = {
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
}

# data 7.5亿
with open("/data/home/ze.song/data/corpus/dialogue/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

print("corpus ok")
# with open("/data/home/ze.song/data/corpus/dialogue/50w.pkl", "rb") as f:
#     corpus = pickle.load(f)

with open(
    "/data/home/ze.song/git/block/src/block/bricks/train/gpt/model_files/vocab.pkl",
    "rb",
) as f:
    vocab = pickle.load(f)

# tokenizer
tok = Tokenizer(vocab=vocab)
tok.init_model()

# dataset
imp.reload(dataset)
data_set = dataset.DataSet(corpus=corpus, config=config, tokenizer=tok)
data_loader = data_set.make_dataloader(batch_size=16)

# model
model = GPT2LMHeadModel(config=config)
# model.load_state_dict(torch.load('./model.pkl', map_location="cpu"))

# trainer
imp.reload(trainer)
t = trainer.Trainer(model=model, train_loader=data_loader, opt_freq=32)
# t.optim.n_current_steps=148000
t.train(file_path="/data/home/ze.song/models/gpt", max_num=1e6)


model.load_state_dict(
    torch.load("./check_points/bert/model_1000000.pkl", map_location="cpu")
)
