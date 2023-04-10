# chatbot

中英文聊天机器人.

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

模型backbone是roberta.

分类头有三个,分别是2分类(1:正向,0:负向),3分类(2:正,1:中,0:负),5分类(0-4分别对应1星-5星).


预训练模型全部开源,可以直接下载,也可以直接在代码中读取远端模型.

16m参数模型:

百度云地址：链接：https://pan.baidu.com/s/1tzcY98JuQ75XoPzjzLQGnA 提取码：qewg

huggingface : https://huggingface.co/szzzzz/sentiment_classifier_sentence_level_bert_16m


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

9. chatgpt生成语料,可以参考https://github.com/LianjiaTech/BELLE

10. 其它


## 使用

文本目前只做了英文,只使用了5个类别的分类头.

使用方法如下:

```python
from sentiment_classification import SentimentClassifier

model = SentimentClassifier()

# 如果模型down到了本地
model.load(model_path)
# 也可以直接使用远端
model.load('szzzzz/sentiment_classifier_sentence_level_bert_16m')

# 模型预测
result = model.rank("I like it.")
'''
result
4.79
'''

result = model.rank("I hate it.")
'''
result
2.42
'''

star = round(model.rank("I hate it."))
'''
star
2
'''


```
