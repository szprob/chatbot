import pickle
import random
import copy
import pandas as pd
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
            corpus = corpus[: int(len(corpus) * ratio)]
            # corpus = random.sample(corpus, int(len(corpus) * ratio))
        print(path + " : " + str(len(corpus)))
        print(corpus[0])
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
        self.corpus = []
        # log data
        translation_davinci = load_pkl(f"{self.data_path}/translation_davinci.pkl")
        translation_turbo = load_pkl(f"{self.data_path}/translation_turbo.pkl")
        toxic_turbo = load_pkl(f"{self.data_path}/toxic_turbo.pkl")
        self.corpus = (
            self.corpus + translation_davinci + translation_turbo + toxic_turbo
        )
        # alpaca
        alpaca_gpt4 = load_pkl(f"{self.data_path}/alpaca_data_gpt4.pkl")
        cn_alpaca_data = load_pkl(f"{self.data_path}/alpaca_data_gpt4_zh.pkl")
        unnatural = load_pkl(f"{self.data_path}/unnatural_instruction_gpt4_data.pkl")
        alpaca_en = load_pkl(f"{self.data_path}/alpaca_data_en.pkl")
        self.corpus = self.corpus + alpaca_gpt4 + cn_alpaca_data + unnatural + alpaca_en
        for lang in ["pt", "es", "ca", "eu", "gl", "at"]:
            c = load_pkl(f"{self.data_path}/alpaca_data_{lang}.pkl")
            self.corpus = self.corpus + c

        ru = load_pkl(f"{self.data_path}/alpaca_data_gpt4_ru.pkl")
        es = load_pkl(f"{self.data_path}/alpaca_data_gpt4_es.pkl")
        it = load_pkl(f"{self.data_path}/alpaca_data_gpt4_it.pkl")

        self.corpus = self.corpus + ru + es + it
        self.len1 = len(self.corpus)

        self.multi = load_pkl(f"{self.data_path}/multiturn_chat_0.8M.pkl", ratio=0.2)
        self.len_multi = len(self.multi)

        self.length = self.len1 + self.len_multi

        if all:
            belle_school = load_pkl(
                f"{self.data_path}/school_math_0.25M.pkl", ratio=0.1
            )
            generated_chat = load_pkl(
                f"{self.data_path}/generated_chat_0.4M.pkl", ratio=0.1
            )
            self.c2 = belle_school + generated_chat
            self.len_c2 = len(self.c2)

            # belle500k = load_pkl(f"{self.data_path}/train_0.5M_CN.pkl",ratio=1)
            # belle1m = load_pkl(f"{self.data_path}/train_1M_CN.pkl",ratio=1)
            belle2m = load_pkl(f"{self.data_path}/train_2M_CN.pkl", ratio=0.25)
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

    def generate_and_tokenize_prompt(self, data_point, is_multi=False):
        input_text = data_point[0]

        # single turn data
        if not is_multi:
            input_text = "Human: \n" + input_text + "\n\nAssistant: \n"
            target_text = data_point[1]

        # multi turn data
        else:
            # try:
            #     text = data_point[0] + data_point[1]
            #     text = text.split("\nAssistant:")
            #     # at least 2 turns
            #     target_idx = random.randint(2, len(text) - 1)
            #     target_text = text[target_idx].split("\nHuman:")[0]
            #     input_text = "\nAssistant:".join(text[:target_idx])
            #     input_text = input_text + "\nAssistant: "
            # except ValueError:
            input_text = data_point[0]
            target_text = data_point[1]

        input_ids = []
        labels = []

        input_text_id = self.tokenizer.encode(
            input_text, add_special_tokens=False
        )  # do not add bos_token_id
        label_text_id = [-100] * len(input_text_id)
        input_ids += input_text_id
        labels += label_text_id

        target_text_id = self.tokenizer.encode(
            target_text, add_special_tokens=False
        )  # do not add bos_token_id
        label_target_id = copy.deepcopy(target_text_id)
        input_ids += target_text_id
        labels += label_target_id

        input_ids += [self.tokenizer.eos_token_id]  # make sure eos_token_id is correct
        labels += [self.tokenizer.eos_token_id]

        input_ids = input_ids[: self.max_length - 1]
        labels = labels[: self.max_length - 1]

        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[
                18:24
            ]  # labels can not have all values being -100. 18 and 24 are just random numbers

        attention_mask = [1] * len(input_ids)

        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
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

    # def make_dataloader(self, batch_size=16, num_workers=8):
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(self)
    #     train_loader = torch.utils.data.DataLoader(
    #         dataset=self,
    #         sampler=train_sampler,
    #         batch_size=batch_size // 3,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #         shuffle=False,
    #         collate_fn=transformers.DataCollatorForSeq2Seq(
    #             self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #         ),
    #     )
    #     i = 1
    #     while True:
    #         train_sampler.set_epoch(i)
    #         i += 1
    #         for data in train_loader:
    #             yield data


class RewardDataSet(torch.utils.data.Dataset):
    """for critic model"""

    def __init__(self, config, tokenizer, data_path):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = self.config.get("max_length", 1024)
        self.data_path = data_path
        self._collate_fn = transformers.DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        self.load_data()

    def load_data(self):

        # log data
        translation_davinci = pd.read_parquet(
            f"{self.data_path}/translation_davinci.parquet"
        )
        translation_turbo = pd.read_parquet(
            f"{self.data_path}/translation_turbo.parquet"
        )
        toxic_turbo = pd.read_parquet(f"{self.data_path}/toxic_turbo.parquet")
        self.corpus = pd.concat(
            [translation_davinci, translation_turbo, toxic_turbo], axis=0
        )
        self.corpus = self.corpus.reset_index(drop=True)

        # alpaca
        alpaca_data_en = pd.read_parquet(f"{self.data_path}/alpaca_data_en.parquet")
        alpaca_data_es = pd.read_parquet(f"{self.data_path}/alpaca_data_es.parquet")
        alpaca_data_zh = pd.read_parquet(f"{self.data_path}/alpaca_data_zh.parquet")
        belle_multi = pd.read_parquet(f"{self.data_path}/belle_multi.parquet")
        self.corpus = pd.concat(
            [
                self.corpus,
                alpaca_data_en,
                alpaca_data_es,
                alpaca_data_zh,
            ],
            axis=0,
        )

        self.multi = belle_multi
        self.corpus = self.corpus.reset_index(drop=True)
        self.len1 = len(self.corpus)
        self.len_multi = len(self.multi)
        self.length = self.len1 + self.len_multi
        print("length : ", self.length)
        print("len1 : ", self.len1)
        print("len_multi : ", self.len_multi)

    def __len__(self):
        return self.length

    def generate_and_tokenize_prompt(self, prompt, response, is_multi):
        # single turn data
        if not is_multi:
            input_text = "Human: \n" + prompt + "\n\nAssistant: \n"
            target_text = response

        # multi turn data
        else:
            input_text = prompt
            target_text = response

        input_ids = []

        input_text_id = self.tokenizer.encode(
            input_text, add_special_tokens=False
        )  # do not add bos_token_id
        input_ids += input_text_id

        target_text_id = self.tokenizer.encode(
            target_text, add_special_tokens=False
        )  # do not add bos_token_id
        input_ids += target_text_id

        input_ids += [self.tokenizer.eos_token_id]  # make sure eos_token_id is correct
        input_ids = input_ids[: self.max_length - 1]

        attention_mask = [1] * len(input_ids)
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return tokenized_full_prompt

    def process(self, idx):
        if idx < self.len1:
            prompt = self.corpus["prompt"][idx]
            chosen = self.corpus["chosen"][idx]
            rejected = self.corpus["rejected"][idx]
            is_multi = False
        else:
            idx = idx - self.len1
            prompt = self.multi["prompt"][idx]
            chosen = self.multi["chosen"][idx]
            rejected = self.multi["rejected"][idx]
            is_multi = True

        tokenized_full_prompt = self.generate_and_tokenize_prompt(
            prompt, chosen, is_multi=is_multi
        )
        tokenized_full_prompt_rejected = self.generate_and_tokenize_prompt(
            prompt, rejected, is_multi=is_multi
        )

        return tokenized_full_prompt, tokenized_full_prompt_rejected

    def collate_fn(self, batch):
        batch1 = [b[0] for b in batch]
        batch2 = [b[1] for b in batch]
        b1 = self._collate_fn(batch1)
        b2 = self._collate_fn(batch2)
        return b1, b2

    def __getitem__(self, idx):
        return self.process(idx)


if __name__ == "__main__":
    import json

    model_config = json.load(open("config/reward_model_config.json"))
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(
        "/data/home/ze.song/models/chatbot/reward_model"
    )
    ds = RewardDataSet(model_config, tokenizer, "/data/home/ze.song/data/corpus/reward")
    collate_fn = transformers.DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    batch1 = [b[0] for b in batch]
    batch2 = [b[1] for b in batch]

    collate_fn(batch1)["input_ids"]
    collate_fn(batch1)["attention_mask"]

    collate_fn(batch2)["input_ids"]
    collate_fn(batch2)["attention_mask"]

    batch = [ds[0], ds[1]]
    out = collate_fn(batch)

    tokenizer.pad(batch)
