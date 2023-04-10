import copy
import pickle

import torch
from utils import make_wiki_question, split_list_by2


def pad_list(text, maxlen=1024, pad=0):
    x = [pad] * maxlen
    length = len(text)
    x[:length] = text
    return x


def load_pkl(path):
    with open(path, "rb") as f:
        corpus = pickle.load(f)
        print(path + " : " + str(len(corpus)))
    return corpus


class DataSet(torch.utils.data.Dataset):
    """for dialogue"""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.source_max_token_len = 512
        self.target_max_token_len = 128
        self.load_data()

    def load_data(self):

        self.corpus = load_pkl("/data/home/ze.song/data/corpus/dialogue/corpus_all.pkl")
        self.wiki = load_pkl("/data/home/ze.song/data/corpus/dialogue/wiki_dia.pkl")
        self.lccc_base = load_pkl(
            "/data/home/ze.song/data/corpus/dialogue/lccc_base.pkl"
        )
        self.len1 = len(self.corpus)
        self.len2 = len(self.wiki)
        self.len3 = len(self.lccc_base)
        self.length = self.len1 + self.len2 + self.len3 // 2

    def __len__(self):
        return self.length

    def make_multiturn(self, dialogue):
        dialogues = split_list_by2(dialogue)
        contexts = []
        responses = []
        for dia in dialogues:
            context = []
            for text in dia[:-1]:
                context.append(text)
            context = self.tokenizer.sep_token.join(context)
            response = dia[-1]

            contexts.append(context)
            responses.append(response)
        return contexts, responses

    def batch_encode_plus(self, c):
        res = []
        maxlen = 0
        for t in c:
            code = self.tokenizer.encode(t)
            if len(code) > self.source_max_token_len:
                code = code[-(self.source_max_token_len - 1) :]
                code = [self.tokenizer.cls_token_id] + code
            maxlen = max(maxlen, len(code))
            res.append(code)
        input_ids = []
        attention_mask = []
        for r in res:
            input_ids.append(pad_list(r, maxlen=maxlen))
            attention_mask.append([1] * len(r) + [0] * (maxlen - len(r)))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def process(self, idx):
        if idx < self.len1:
            dialogue = self.corpus[idx]
            if len(dialogue) % 2 != 0:
                dialogue = dialogue[:-1]
        else:
            idx -= self.len1
            dialogue = self.wiki[idx]
            dialogue[0] = make_wiki_question(dialogue[0])

        return dialogue

    def __getitem__(self, idx):
        return self.process(idx)

    def _collate_fn(self, batch):
        """pad seq"""

        c = []
        r = []
        for dialogue in batch:
            contexts, responses = self.make_multiturn(dialogue)
            c.extend(contexts)
            r.extend(responses)

        input_encodings = self.batch_encode_plus(c)
        target_encodings = self.tokenizer.batch_encode_plus(
            r,
            max_length=self.target_max_token_len,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = input_encodings["input_ids"]
        attention_mask = input_encodings["attention_mask"]
        target_ids = target_encodings["input_ids"]
        target_attention_mask = target_encodings["attention_mask"]
        label_ids = copy.deepcopy(target_ids)
        label_ids[label_ids == 0] = -100

        return input_ids, attention_mask, target_ids, target_attention_mask, label_ids

    def make_dataloader(self, batch_size=16, num_workers=8):
        train_sampler = torch.utils.data.distributed.DistributedSampler(self)
        train_loader = torch.utils.data.DataLoader(
            dataset=self,
            sampler=train_sampler,
            batch_size=batch_size // 3,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        i = 1
        while True:
            train_sampler.set_epoch(i)
            i += 1
            for data in train_loader:
                bsz, _ = data[0].shape
                if bsz > batch_size:
                    data = [d[-batch_size:, :].contiguous() for d in data]
                yield data
