# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""

import torch
from transformers import BartForConditionalGeneration, BertTokenizer


class Bot:
    def __init__(
        self,
        model_path="/data/home/ze.song/models/chatbot/model_10000.pkl",
    ):
        model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.cuda()
        model.eval()
        self.model = model
        # tokenizer
        tok = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.tok = tok
        self.init()

    def init(self):
        self.x = ""
        self.his = []

    @torch.no_grad()
    def talk(self, input=""):
        if input == "重启":
            self.init()
            return "重启完毕"

        self.x = self.x + self.tok.sep_token + input
        x = self.x
        input_ids = self.tok.encode(x, return_tensors="pt")
        if len(input_ids[0]) > 511:
            input_ids = input_ids[:, -511:]
            cls_token = torch.tensor([[self.tok.cls_token_id]]).long()
            input_ids = torch.cat([cls_token, input_ids], axis=1)

        input_ids = input_ids.cuda()
        pred_ids = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_length=126,
            repetition_penalty=1.2,
        )
        res = "".join(self.tok.convert_ids_to_tokens(pred_ids[0][1:-1]))
        self.x = self.x + self.tok.sep_token + res
        self.his.append((input, res))

        return res

    def __call__(self, text: str):
        res = self.talk(text)
        self.init()
        return res


if __name__ == "__main__":
    bot = Bot(
        model_path="/data/home/ze.song/models/chatbot/model_800000.pkl",
    )
    bot.init()
    bot.talk("hello")
    bot.talk("你好")
    bot.talk("再见")
    bot.talk("去吃火锅吗")
    bot.talk("为什么不吃")
    bot.talk("啥意思")
    bot.talk("中国的首都是哪里")
    bot("什么是孔子")
    bot("什么是保险")
    bot("保险不赔付有什么后果")
    bot("保险不赔付有什么后果")
    bot("什么是基金")
    bot("被保险期间被保险人身亡了怎么办")
    bot("孔子是什么")
    bot("孩子感冒了怎么办")
