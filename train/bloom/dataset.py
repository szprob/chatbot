import pickle
import random

import torch
import transformers


def pad_list(text, maxlen=1024, pad=0):
    x = [pad] * maxlen
    length = len(text)
    x[:length] = text
    return x


def load_pkl(path, ratio=1):
    with open(path, "rb") as f:
        corpus = pickle.load(f)
        corpus = [c for c in corpus if len(c) >= 2]
        if ratio < 0.99:
            corpus = random.sample(corpus, int(len(corpus) * ratio))
        print(path + " : " + str(len(corpus)))
    return corpus


class DataSet(torch.utils.data.Dataset):
    """for dialogue"""

    def __init__(self, config, tokenizer, data_path, all=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = self.config.get("max_length", 1024)
        self.data_path = data_path
        self.load_data(all)

    def load_data(self, all):

        # 100k
        alpaca_gpt4 = load_pkl(f"{self.data_path}/alpaca_data_gpt4.pkl")
        cn_alpaca_data = load_pkl(f"{self.data_path}/alpaca_data_gpt4_zh.pkl")
        unnatural = load_pkl(f"{self.data_path}/unnatural_instruction_gpt4_data.pkl")

        self.corpus = alpaca_gpt4 + cn_alpaca_data + unnatural
        self.corpus = self.corpus
        self.len1 = len(self.corpus)

        self.multi = load_pkl(f"{self.data_path}/multiturn_chat_0.8M.pkl", ratio=1)
        self.len_multi = len(self.multi)

        self.length = self.len1 + self.len_multi

        if all:
            belle_school = load_pkl(f"{self.data_path}/school_math_0.25M.pkl", ratio=1)
            generated_chat = load_pkl(
                f"{self.data_path}/generated_chat_0.4M.pkl", ratio=0.5
            )
            self.c2 = belle_school + generated_chat
            self.len_c2 = len(self.c2)

            # belle500k = load_pkl(f"{self.data_path}/train_0.5M_CN.pkl",ratio=1)
            # belle1m = load_pkl(f"{self.data_path}/train_1M_CN.pkl",ratio=1)
            belle2m = load_pkl(f"{self.data_path}/train_2M_CN.pkl", ratio=1)
            self.c3 = belle2m  # + belle1m + belle500k
            self.len_c3 = len(self.c3)

            self.corpus = self.corpus + self.c2 + self.c3
            self.len1 = self.len1 + self.len_c2 + self.len_c3
            self.length = self.length + self.len_c2 + self.len_c3

        print("length : ", self.length)
        print("len1 : ", self.len1)
        print("len_multi : ", self.len_multi)

    def __len__(self):
        return self.length

    def tokenize(self, text, add_eos_token=True):
        res = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        if (
            res["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(res["input_ids"]) < self.max_length
            and add_eos_token
        ):
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)

        if add_eos_token and len(res["input_ids"]) >= self.max_length:
            res["input_ids"][self.max_length - 1] = self.tokenizer.eos_token_id
            res["attention_mask"][self.max_length - 1] = 1

        res["labels"] = res["input_ids"].copy()
        return res

    def generate_and_tokenize_prompt(self, data_point, is_multi=False):
        input_text = data_point[0]

        # single turn data
        if not is_multi:
            if random.random() < 0.5:
                input_text = "Human: " + input_text + "\n\nAssistant: "
            else:
                input_text = "Human: " + input_text + "\nAssistant: "

            target_text = data_point[1]

        # multi turn data
        else:
            # try :
            #     text = data_point[0] + data_point[1]
            #     text = text.split('\nAssistant:')
            #     # at least 2 turns
            #     target_idx = random.randint(2,len(text)-1)
            #     target_text = text[target_idx].split('\nHuman:')[0]
            #     input_text = '\nAssistant:'.join(text[:target_idx])
            #     input_text = input_text + "\nAssistant: "
            # except ValueError:
            input_text = data_point[0]
            target_text = data_point[1]

        if self.tokenizer.bos_token is not None:
            input_text = self.tokenizer.bos_token + input_text

        target_text = target_text + self.tokenizer.eos_token
        full_prompt = input_text + target_text

        tokenized_full_prompt = self.tokenize(full_prompt)

        # not train on inputs
        user_prompt = input_text
        tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def process(self, idx):
        if idx < self.len1:
            dialogue = self.corpus[idx]
            tokenized_full_prompt = self.generate_and_tokenize_prompt(
                dialogue, is_multi=False
            )
        else:
            idx = idx - self.len1
            dialogue = self.multi[idx]
            tokenized_full_prompt = self.generate_and_tokenize_prompt(
                dialogue, is_multi=True
            )

        return tokenized_full_prompt

    def __getitem__(self, idx):
        return self.process(idx)

    def make_dataloader(self, batch_size=16, num_workers=8):
        train_sampler = torch.utils.data.distributed.DistributedSampler(self)
        train_loader = torch.utils.data.DataLoader(
            dataset=self,
            sampler=train_sampler,
            batch_size=batch_size // 3,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        i = 1
        while True:
            train_sampler.set_epoch(i)
            i += 1
            for data in train_loader:
                yield data
