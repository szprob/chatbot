# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""


import torch
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel


class Bot:
    def __init__(
        self,
        model_path="/data/home/ze.song/models/chatbot/gpt2_base.pkl",
    ):
        self.tok = BertTokenizer(vocab_file="./vocab.txt")
        self.config = GPT2Config(
            eos_token_id=102, vocab_size=len(self.tok.vocab), n_positions=768
        )
        model = GPT2LMHeadModel(config=self.config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.cuda()
        model.eval()
        self.model = model
        self.init()

    def init(self):
        self.x = ""
        self.his = []

    @torch.no_grad()
    def talk(self, input=""):
        if input == "重启":
            self.init()
            return "重启完毕"

        if self.x:
            self.x = self.x + self.tok.sep_token + input
            x = self.x
        else:
            x = input

        input_ids = self.tok.encode(x, return_tensors="pt")
        if len(input_ids[0]) > self.config.n_positions:
            input_ids = input_ids[:, (-self.config.n_positions + 1) :]
            cls_token = torch.tensor([[self.tok.cls_token_id]]).long()
            input_ids = torch.cat([cls_token, input_ids], axis=1)

        input_ids = input_ids.cuda()
        pred_ids = self.model.generate(
            input_ids,
            eos_token_id=102,
            pad_token_id=102,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_length=384,
            repetition_penalty=1.2,
        )
        _, input_len = input_ids.shape
        pred = pred_ids[0][input_len:]
        res = self.tok.decode(pred, skip_special_tokens=True)
        res = res.replace("[ sep ] ", "")
        self.x = x + self.tok.sep_token + res
        self.his.append((input, res))

        return res

    def __call__(self, text: str):
        res = self.talk(text)
        self.init()
        return res


if __name__ == "__main__":
    bot = Bot(
        model_path="/data/home/ze.song/models/tmp/model_290000.pkl",
    )
    bot.init()
    bot.talk("hello")
    bot.talk("你好")
    bot.talk("你是谁?")
    bot.talk("再见")
    bot.talk("去吃火锅吗")
    bot.talk("为什么不吃")
    bot.talk("啥意思")
    bot.talk("中国的首都是哪里")
    bot.talk("什么是新立镇站")
    bot("什么是孔子")
    bot("什么是保险")
    bot("保险不赔付有什么后果")
    bot("保险不赔付有什么后果")
    bot("什么是基金")
    bot("被保险期间被保险人身亡了怎么办")
    bot("孔子是什么?")
    bot("孩子感冒了怎么办?")
    bot("假如你是一只狗,你会干什么?")
    bot("爸爸的爸爸叫什么?")
    bot("翻译下面这句话:我喜欢自由")
    bot("告诉我怎么做咖啡")
    bot("为给定的主题生成几个相关的问题:经济下滑")
    bot("中国的四大古都是哪四个")
    bot("完成下面的句子.我今天去看了个电影,但是情节")
    bot("翻译下面这句话: i love playing free fire")

    bot.init()
    input = "hello"
    if bot.x:
        bot.x = bot.x + bot.tok.sep_token + input
        x = bot.x
    else:
        x = input

    x
    input_ids = bot.tok.encode(x, return_tensors="pt")
    if len(input_ids[0]) > bot.config.n_positions:
        input_ids = input_ids[:, (-bot.config.n_positions + 1) :]
        cls_token = torch.tensor([[bot.tok.cls_token_id]]).long()
        input_ids = torch.cat([cls_token, input_ids], axis=1)

    input_ids = input_ids.cuda()
    pred_ids = bot.model.generate(
        input_ids,
        eos_token_id=102,
        pad_token_id=102,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        max_length=256,
        repetition_penalty=1.2,
    )

    _, input_len = input_ids.shape
    pred = pred_ids[0][input_len:]

    res = bot.tok.decode(pred, skip_special_tokens=True)
    res
    bot.x = bot.x + bot.tok.sep_token + res
    bot.his.append((input, res))
