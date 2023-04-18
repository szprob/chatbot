# chatbot

中文聊天机器人.

交互的demo部署到了huggingface:
https://huggingface.co/spaces/szzzzz/chatbot

## 安装

```shell
git clone git@github.com:szprob/chatbot.git
cd chatbot
python setup.py install --user
```

或者直接

```shell
pip install git+https://github.com/szprob/chatbot.git
```

## 模型

因为bloom的多语言支持较好,本模型主要微调了bloom.


预训练模型全部开源,可以直接下载,也可以直接在代码中读取远端模型.

1b7模型:

huggingface : https://huggingface.co/szzzzz/chatbot_bloom_1b7


## 数据集

总结了一下或许有用的中英文对话数据集.包括不限于:

1. KdConv. 4500条,知识驱动的中文对话. https://github.com/thu-coai/KdConv

2. LCCC. 1200万中文,包含了微博,贴吧,PTT,豆瓣,小黄鸡等多个语料来源,并且经过了清洗. https://github.com/thu-coai/CDial-GPT

3. Covid19-NLP. 中文医生和患者的对话,轮数较多,大概8000多条. https://github.com/lwgkzl/Covid19-NLP

4. chatterbot-corpus.数量比较少,可以作为chatbot的回复. https://github.com/gunthercox/chatterbot-corpus

5. insuranceqa. 保险业务的单轮语料,质量较高,包含中英文,中文为英文的翻译版,有些瑕疵. https://github.com/chatopera/insuranceqa-corpus-zh

6. dailydialog. 很经典的英文对话语料,13,118条. http://yanran.li/dailydialog

7. EmpatheticDialogues. 25k条英文对话,提供情感标签. https://github.com/facebookresearch/EmpatheticDialogues

8. esconv.1,300条英文对话,提供不同主题. https://github.com/thu-coai/Emotional-Support-Conversation


chatgpt生成语料:

1. belle .  可以参考https://github.com/LianjiaTech/BELLE

2. alpaca . https://github.com/tloen/alpaca-lora

3. gpt4all . https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release

4. 其它


## 使用

使用gradio搭建机器人的详细代码见examples/gradio_demo.py

简单使用方法如下:

```python
from chatbot import Bot

model = Bot()

# 如果模型down到了本地
model.load(model_path)
# 也可以直接使用远端
model.load('szzzzz/chatbot_bloom_1b7')

# 模型预测
inputs = "Human : 你好啊! \nnAssistant: "
response = model.generate(inputs)
'''
response
你好,我是你的ai助手
'''

```


## 训练

提供了基于deepspeed的微调. 训练方式可以按照如下三步:

1. 在大规模语料上训练模型基座,可以直接使用开源的基座,如bloom.

2. 在对话格式的语料上微调模型,可以在chatgpt或者其他领域内开源对话语料上微调.

3. 如果只是想训练一个能开放对话的chatbot,第二步已经可以满足.如果想做针对性训练,有条件可以训练RL模型做调整,没有条件可以在领域内的少量数据上做最后的微调.
