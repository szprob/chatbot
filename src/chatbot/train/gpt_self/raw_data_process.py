import gzip
import json
import os
import pickle

import pandas as pd
import zhconv
from tqdm import tqdm


def save_pkl(path, file):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pkl(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def process_50w(path):
    """process 50w dialogue"""
    with open(path, "r") as f:
        d = f.readlines()

    corpus = []
    res = []
    for x in tqdm(d):
        x = x.strip()
        if x == "":
            corpus.append(res)
            res = []
        else:
            res.append(x)
    return corpus


def process_lccc(path):
    with open(path, "r") as f:
        b = json.load(f)

    corpus = []
    for s in tqdm(b):
        res = []
        for t in s:
            t = t.strip()
            this_res = ""
            for i, ch in enumerate(t):
                if ch == " ":
                    if ord(t[i - 1]) <= 128 and ord(t[i + 1]) <= 128:
                        this_res = this_res + ch
                else:
                    this_res = this_res + ch
            res.append(this_res)
        corpus.append(res)
    return corpus


def wiki_process(path):
    corpus = []
    doc = []
    for root, _, names in os.walk(path):
        for name in tqdm(names):
            file_path = f"{root}/{name}"
            with open(file_path, "r") as f:
                a = f.readlines()
                a = [i for i in a if i != "\n"]
                a = [zhconv.convert(i, "zh-hans") for i in a]
            for s in a:
                if s.startswith("<doc id="):
                    doc = []
                elif s.startswith("</doc>"):
                    corpus.append(doc)
                    doc = []
                else:
                    doc.append(s.strip())
    return corpus


def process_kdconv(path):
    with open(path, "r") as file:
        data = file.read()
    data = json.loads(data)
    corpus = []
    for dia in data:
        res = []
        dia = dia["messages"]
        for s in dia:
            res.append(s["message"])
        corpus.append(res)
    return corpus


def process_covid19(path):
    data = load_pkl(path)
    corpus = []
    for dia in data:
        res = []
        text = ""
        id = None
        for s in dia:
            if id is None:
                id = s["id"]
                text = text + s["text"] + "\n"
            else:
                if s["id"] == id:
                    text = text + s["text"] + "\n"
                else:
                    res.append(text)
                    text = s["text"] + "\n"
                    id = s["id"]
        res.append(text)
        corpus.append(res)

    corpus_dis = []
    for i, d in enumerate(corpus):
        if i == 0:
            corpus_dis.append(d)
            continue
        else:
            if d == corpus[i - 1]:
                continue
            else:
                corpus_dis.append(d)

    return corpus_dis


def process_insuranceqa_once(train_path, ans_path):
    with gzip.open(train_path, "rb") as file:
        train = file.read()
    train = json.loads(train)
    with gzip.open(ans_path, "rb") as file:
        a = file.read()
    a = json.loads(a)

    corpus = []
    for k in train.keys():
        zh = train[k]["zh"]
        en = train[k]["en"]
        if type(zh) != str:
            raise
        if type(en) != str:
            raise
        for ans in train[k]["answers"]:
            dia1 = [zh, a[ans]["zh"]]
            dia2 = [en, a[ans]["en"]]
            corpus.append(dia1)
            corpus.append(dia2)
    return corpus


def process_insuranceqa(path):
    train_path = f"{path}train.json.gz"
    valid_path = f"{path}valid.json.gz"
    test_path = f"{path}test.json.gz"
    ans_path = f"{path}answers.json.gz"

    corpus_train = process_insuranceqa_once(train_path, ans_path)
    corpus_valid = process_insuranceqa_once(valid_path, ans_path)
    corpus_test = process_insuranceqa_once(test_path, ans_path)

    return corpus_train + corpus_valid + corpus_test


def process_ch_medical(path):
    corpus = []
    for i in os.listdir(path):
        p = path + "/" + i
        for j in os.listdir(p):
            p = p + "/" + j

            df = pd.read_csv(p, encoding="GB18030")

            c = []
            for row in df.iterrows():
                row = row[1]
                if row["ask"] == "无" or row["answer"] == "无":
                    continue
                c.append([row["ask"], row["answer"]])
            corpus.extend(c)
    return corpus


if __name__ == "__main__":

    # KdConv
    # https://github.com/thu-coai/KdConv
    path = "/data/home/ze.song/git/KdConv/data/film/train.json"
    corpus = process_kdconv(path)
    path = "/data/home/ze.song/data/corpus/dialogue/KdConv_film.pkl"
    save_pkl(path, corpus)
    path = "/data/home/ze.song/git/KdConv/data/music/train.json"
    corpus = process_kdconv(path)
    path = "/data/home/ze.song/data/corpus/dialogue/KdConv_music.pkl"
    save_pkl(path, corpus)
    path = "/data/home/ze.song/git/KdConv/data/travel/train.json"
    corpus = process_kdconv(path)
    path = "/data/home/ze.song/data/corpus/dialogue/KdConv_travel.pkl"
    save_pkl(path, corpus)

    # Covid19-NLP
    # https://github.com/lwgkzl/Covid19-NLP
    path = "/data/home/ze.song/git/Covid19-NLP/COVID_Dialogue_Dataset1.pk"
    corpus = process_covid19(path)
    path = "/data/home/ze.song/git/Covid19-NLP/COVID_Dialogue_Dataset2.pk"
    corpus2 = process_covid19(path)
    corpus = corpus + corpus2
    path = "/data/home/ze.song/data/corpus/dialogue/covid_dialogue.pkl"
    save_pkl(path, corpus)

    # insuranceqa
    # https://github.com/chatopera/insuranceqa-corpus-zh
    path = "/data/home/ze.song/git/insuranceqa-corpus-zh/corpus/pool/"
    corpus = process_insuranceqa(path)
    path = "/data/home/ze.song/data/corpus/dialogue/insuranceqa.pkl"
    save_pkl(path, corpus)

    # chinese medical dialogue_datasets
    path = (
        "/data/home/ze.song/data/raw_corpus/dialogue/chinese medical dialogue_datasets"
    )
    corpus = process_ch_medical(path)
    path = "/data/home/ze.song/data/corpus/dialogue/cn_medical.pkl"
    save_pkl(path, corpus)

    # 50w
    path = "/data/home/ze.song/data/raw_corpus/dialogue/50w/train.txt"
    corpus = process_50w(path)
    path = "/data/home/ze.song/data/corpus/dialogue/50w.pkl"
    save_pkl(path, corpus)

    # 100w
    path = "/data/home/ze.song/data/raw_corpus/dialogue/100w/train_100w.txt"
    corpus = process_50w(path)
    path = "/data/home/ze.song/data/corpus/dialogue/100w.pkl"
    save_pkl(path, corpus)

    # LCCC
    # https://github.com/thu-coai/CDial-GPT
    path = "/data/home/ze.song/data/raw_corpus/dialogue/LCCD.json"
    corpus = process_lccc(path)
    path = "/data/home/ze.song/data/corpus/dialogue/lccc_large.pkl"
    save_pkl(path, corpus)

    path = "/data/home/ze.song/data/raw_corpus/dialogue/LCCC-base_train.json"
    corpus = process_lccc(path)
    path = "/data/home/ze.song/data/corpus/dialogue/lccc_base.pkl"
    save_pkl(path, corpus)

    # wiki
    path = "/data/home/ze.song/data/raw_corpus/wikicorpus/zh/AA"
    corpus = wiki_process(path)
    path = "/data/home/ze.song/data/corpus/zh_wiki.pkl"
    save_pkl(path, corpus)
