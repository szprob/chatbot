# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""

import importlib as imp
import pickle

import textgen
import torch

import chatbot

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = torch.device(device)
n_device = torch.cuda.device_count()
torch.backends.cudnn.is_available()
torch.backends.cudnn.version()
torch.set_default_tensor_type(torch.FloatTensor)
torch.cuda.set_device(0)

vocab_path = (
    "/data/home/ze.song/git/block/src/block/bricks/train/gpt/model_files/vocab.pkl"
)


def get_latest_model(path="/data/home/ze.song/models/gptb"):

    with open(
        f"{path}/logs.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    epoch = log[-1][0].replace("epoch : ", "")
    iter = log[-1][1].replace("iter: ", "")

    model_path = f"{path}/model_{epoch}_{iter}.pkl"
    return model_path


def get_latest_model2(path="/data/home/ze.song/models/gpt"):

    with open(
        f"{path}/logs.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    iter = log[-1][0].replace("iter: ", "")

    model_path = f"{path}/model_{iter}.pkl"
    return model_path


if __name__ == "__main__":

    # large
    imp.reload(chatbot)
    model_path = get_latest_model2("/data/home/ze.song/models/gpt")
    print(model_path)
    model_path = "/data/home/ze.song/models/gpt/model_350000.pkl"
    bot = chatbot.Bot(model_path=model_path, vocab_path=vocab_path)
    bot.init()
    bot.talk("你好", top_k=None, top_p=0.7)
    bot.talk("去吃火锅吗?", top_k=None, top_p=0.5)
    bot.talk("再见", top_k=None, top_p=0.6)
    bot.talk("你是谁", top_k=None, top_p=0.6)
    bot.talk("在哪里见面", top_k=None, top_p=0.6)
    bot.talk("你喜欢什么", top_k=None, top_p=0.6)
    bot.talk("对啊", top_k=None, top_p=0.6)
    bot.result

    with open(
        "/data/home/ze.song/models/gptb/logs.pkl",
        "rb",
    ) as f:
        a3 = pickle.load(f)
    a3

    # lm

    config = {
        "vocab_size": 13317,
        "embd_pdrop": 0.1,
        "n_embd": 1024,
        "n_head": 16,
        "n_positions": 512,
        "n_layer": 12,
        "attn_pdrop": 0.1,
        "resid_dropout": 0.1,
        "n_inner": 1024 * 4,
        "layer_norm_epsilon": 1e-5,
        "pad_idx": 0,
        "dtype": torch.float32,
    }
    imp.reload(textgen)
    model_path = get_latest_model2("/data/home/ze.song/models/gptlm")
    print(model_path)
    model_path = "/data/home/ze.song/models/gptlm/model_980000.pkl"
    bot2 = textgen.Bot(model_path=model_path, vocab_path=vocab_path, config=config)
    bot2.gen("中国政府", top_k=None, top_p=0.7, max_num=100)
    bot2.gen("中国政府", top_k=20, max_num=100)
    bot2.gen("共产党", top_k=20, top_p=0.5)
    bot2.gen("条件", top_k=None, top_p=0.8)
    bot2.gen("武侠", top_k=None, top_p=0.7)
    bot2.gen("苹果", top_k=None, top_p=0.6)
    bot2.gen("电脑", top_k=None, top_p=0.6)
    bot2.gen("圣剑", top_k=None, top_p=0.6)
    bot2.gen("脂肪", top_k=None, top_p=0.9)
    bot.result

    with open(
        "/data/home/ze.song/models/gptlm/logs.pkl",
        "rb",
    ) as f:
        a3 = pickle.load(f)
    a3

    # add
    imp.reload(textgen)
    model_path = get_latest_model2("/data/home/ze.song/models/gpta")
    print(model_path)
    model_path = "/data/home/ze.song/models/gpta/model_210000.pkl"
    bot3 = textgen.Adder(model_path=model_path)
    bot3.gen("8999-9999", top_k=1, top_p=0.6, max_num=100)
    bot3.his

    with open(
        "/data/home/ze.song/models/gptl/logs.pkl",
        "rb",
    ) as f:
        a3 = pickle.load(f)
    a3
