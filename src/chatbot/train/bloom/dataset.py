import pickle

import torch


def pad_list(text, maxlen=1024, pad=0):
    x = [pad] * maxlen
    length = len(text)
    x[:length] = text
    return x


def load_pkl(path):
    with open(path, "rb") as f:
        corpus = pickle.load(f)
        corpus = [c for c in corpus if len(c) >= 2]
        print(path + " : " + str(len(corpus)))
    return corpus


class DataSet(torch.utils.data.Dataset):
    """for dialogue"""

    def __init__(self, config, tokenizer, data_path):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = self.config.get("max_length", 1024)
        self.data_path = data_path
        self.load_data()

    def load_data(self):

        # belle500k = load_pkl(f"{self.data_path}/train_0.5M_CN.pkl")
        # belle1m = load_pkl(f"{self.data_path}/train_1M_CN.pkl")
        # belle_school = load_pkl(f"{self.data_path}/school_math_0.25M.pkl")
        belle_multi = load_pkl(f"{self.data_path}/multiturn_chat_0.8M.pkl")
        # alpaca_data = load_pkl(f"{self.data_path}/alpaca_data.pkl")
        cn_alpaca_data = load_pkl(f"{self.data_path}/cn_alpaca_data.pkl")
        # generated_chat = load_pkl(f"{self.data_path}/generated_chat_0.4M.pkl")
        # belle2m = load_pkl(f"{self.data_path}/train_2M_CN.pkl")

        self.corpus = cn_alpaca_data
        # self.corpus  = belle500k + belle1m + belle_school
        # self.corpus  = self.corpus + alpaca_data + cn_alpaca_data
        # self.corpus  = self.corpus + generated_chat + belle2m
        self.len1 = len(self.corpus)

        self.multi = belle_multi
        self.len2 = len(self.multi)
        self.length = self.len1 + self.len2

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
        if not is_multi:
            input_text = "Human: " + input_text + "\n\nAssistant: "

        if self.tokenizer.bos_token is not None:
            input_text = self.tokenizer.bos_token + input_text

        target_text = data_point[1] + self.tokenizer.eos_token
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
