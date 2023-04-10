from tqdm import tqdm
from utils import load_pkl, split_list_by2

from chatbot.tokenization.tokenization import Tokenizer


def make_multiturn(tok, dialogue):
    dialogues = split_list_by2(dialogue)
    res = []
    for dia in dialogues:
        context = []
        for text in dia[:-1]:
            context.extend(tok.tokenize(text))
            context.append(tok.sep_token)
        response = tok.tokenize(dia[-1]) + [tok.sep_token]
        res.append((context, response))

    return res


if __name__ == "__main__":
    path = "/data/home/ze.song/data/corpus/dialogue/corpus_all.pkl"
    corpus = load_pkl(path)

    tok = Tokenizer()
    tok.load("../train/t5/cn_vocab.pkl")

    new_corpus = []
    for c in tqdm(corpus):
        if len(c) % 2 != 0:
            c = c[:-1]
        di = make_multiturn(tok, c)
        new_corpus.extend(di)
