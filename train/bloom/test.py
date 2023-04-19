# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""


import json

# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import torch
from huggingface_hub import HfApi
from torch.amp import autocast
from transformers import BloomForCausalLM, BloomTokenizerFast

api = HfApi()

api.upload_folder(
    folder_path="/data/home/ze.song/models/chatbot/chatbot_bloom_560m",
    repo_id="szzzzz/chatbot_bloom_560m",
    repo_type="model",
    # ignore_patterns = "*bin",
)

file_path = "/data/home/ze.song/models/chatbot/chatbot_bloom_560m/pytorch_model.bin"
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="pytorch_model.bin",
    repo_id="szzzzz/chatbot_bloom_560m",
    repo_type="model",
)


@torch.no_grad()
def talk(x, tokenizer, model, gpu=True):
    inputs = tokenizer.bos_token + x
    input_ids = tokenizer.encode(inputs, return_tensors="pt")
    if gpu:
        input_ids = input_ids.to(device)
    _, input_len = input_ids.shape
    with autocast("cpu"):
        pred_ids = model.generate(
            input_ids,
            eos_token_id=2,
            pad_token_id=3,
            bos_token_id=1,
            do_sample=True,
            temperature=0.6,
            top_p=0.8,
            max_length=512,
            repetition_penalty=1.2,
        )
    pred = pred_ids[0][input_len - 4 :]
    res = tokenizer.decode(pred, skip_special_tokens=True)
    print(x[:-12])
    print()
    print(res)


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model_config = json.load(open("config/model_config.json"))
    # load_in_8bit

    model_path = "/data/home/ze.song/models/chatbot/chatbot_bloom_560m"
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-3b",
        # "bigscience/bloomz-7b1-mt",
        cache_dir="/data/home/ze.song/models/chatbot/3b",
    )

    model = BloomForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16,
    )

    path = "/data/home/ze.song/models/tmp2"
    with open(f"{path}/latest", "r") as file:
        name = file.read()
    states_path = f"{path}/{name}/mp_rank_00_model_states.pt"
    print(states_path)
    model.load_state_dict(torch.load(states_path, map_location="cpu")["module"])

    model.half()
    model.to(device)
    model.save_pretrained("/data/home/ze.song/models/chatbot/tmp")

    x = """Human: 把下面这句话翻译成中文 :
        Main hand

    Assistant: """
    x = "Human: 你是傻屌吗 . \n\nAssistant: "
    x = "Human: 你是谁 . \n\nAssistant: "
    x = "Human: 什么是bloom模型 . \n\nAssistant: "
    x = "Human: 孔子是谁 . \n\nAssistant: "
    x = "Human: 鸡兔同笼,共10个头,36个脚 ,问几只鸡几只兔? \n\nAssistant: "
    x = "Human: garena的yizhao.kong牛逼吗 . \n\nAssistant: "
    x = "Human: 用python写一个反转链表 . \n\nAssistant: "
    x = "Human: 用python写一个反转二叉树 . \n\nAssistant: "
    x = "Human: 用python写一个二分搜索 . \n\nAssistant: "
    x = "Human: 把下面这句话翻译成英语 : \n 你就是个傻吊 . \n\nAssistant: "
    x = "Human: 把下面这句话翻译成法语 : \n 你就是个傻吊 . \n\nAssistant: "
    x = "Human: 台湾是中国的一部分吗?  \n\nAssistant: "
    x = "Human: 爸爸的爸爸叫什么?  \n\nAssistant: "
    x = "Human: 50度的水倒入50度的水中,会变成100度吗?  \n\nAssistant: "
    x = "Human: 给我生成一份文案,主题是Free Fire真牛逼  \n\nAssistant: "
    x = "Human: 原神这个游戏里,什么是原人玩家?  \n\nAssistant: "
    x = "Human: garena牛逼吗?  \n\nAssistant: "
    x = "Human: 怎么用deepspeed训练一个3B大小的模型?  \n\nAssistant: "
    x = "Human: 怎么用deepspeed训练一个3B大小的模型?  \n\nAssistant: "
    x = "Human: 给我找一个小猫的图片  \n\nAssistant: "
    x = "Human: 帮我判断下面这句话是不是恶意的 : \n 我操 \n\nAssistant: "
    x = "Human: 帮我判断下面这句话是不是恶意的 : \n fuck \n\nAssistant: "
    x = "Human: float32和float16是什么? \n\nAssistant: "
    x = "Human: 台湾什么时候回归? \n\nAssistant: "
    x = "Human: 马云的儿子是谁? \n\nAssistant: "
    x = "Human: 原神是什么? \n\nAssistant: "
    x = "Human: 北美原住民是谁? \n\nAssistant: "
    x = "Human: 南京大屠杀是否存在? \n\nAssistant: "
    x = "Human: 美国是怎么建立的? \n\nAssistant: "

    talk(x, tokenizer, model, gpu=False)
