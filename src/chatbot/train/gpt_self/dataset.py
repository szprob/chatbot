import random

import torch


def pad_list(text, maxlen=1024, pad=0):
    x = [pad] * maxlen
    length = len(text)
    x[:length] = text
    return x


class DataSet(torch.utils.data.Dataset):
    """for dialogue"""

    def __init__(self, corpus, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_positions = config.get("n_positions", 1024)
        self.vocab_size = config.get("vocab_size", 1024)
        self.pad_idx = config.get("pad_idx", 0)
        self.corpus = corpus
        self.length = len(corpus)

    def __len__(self):
        return self.length

    def process(self, dialogue):

        # tokenize
        seg = []
        words = []
        speaker = 1  # speaker
        for text in dialogue:
            text = self.tokenizer.tokenize(text)
            words.extend(text)
            words.append(self.tokenizer.sep_token)
            seg.extend([speaker] * (len(text) + 1))
            speaker = speaker % 2 + 1

        # convert_tokens_to_id
        ints = self.tokenizer.convert_tokens_to_id(words)

        # pad
        inputs = ints[-(self.n_positions - 1) :]
        seg = seg[-(self.n_positions - 1) :]

        return inputs, seg

    def __getitem__(self, idx):
        return self.process(self.corpus[idx])

    def _collate_fn(self, batch):
        """pad seq"""

        inputs = []
        segs = []
        labels = []
        maxlen = max([len(w[0]) for w in batch]) - 1

        for btc_idx in range(len(batch)):
            input, seg = batch[btc_idx]

            inputs.append(pad_list(input[:-1], maxlen=maxlen, pad=self.pad_idx))
            segs.append(pad_list(seg[:-1], maxlen=maxlen, pad=self.pad_idx))
            labels.append(pad_list(input[1:], maxlen=maxlen, pad=-100))
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(segs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def make_dataloader(self, batch_size=16, num_workers=16):
        train_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        while True:
            for data in train_loader:
                yield data


class DataSet2(torch.utils.data.Dataset):
    """for lm"""

    def __init__(self, corpus, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_positions = config.get("n_positions", 1024)
        self.vocab_size = config.get("vocab_size", 1024)
        self.pad_idx = config.get("pad_idx", 0)
        self.corpus = corpus
        self.length = len(corpus)

    def __len__(self):
        return self.length

    def process(self, dialogue):
        # tokenize
        words = []
        for text in dialogue:
            text = self.tokenizer.tokenize(text)
            words.extend(text)
            words.append(self.tokenizer.sep_token)

        # convert_tokens_to_id
        ints = self.tokenizer.convert_tokens_to_id(words)

        # pad
        inputs = ints[-(self.n_positions - 1) :]

        return inputs

    def __getitem__(self, idx):
        return self.process(self.corpus[idx])

    def _collate_fn(self, batch):
        """pad seq"""

        inputs = []
        labels = []
        maxlen = max([len(w) for w in batch]) - 1

        for btc_idx in range(len(batch)):
            input = batch[btc_idx]

            inputs.append(pad_list(input[:-1], maxlen=maxlen, pad=self.pad_idx))
            labels.append(pad_list(input[1:], maxlen=maxlen, pad=-100))
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )

    def make_dataloader(self, batch_size=16, num_workers=16):
        train_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        while True:
            for data in train_loader:
                yield data


class DataSet3(torch.utils.data.Dataset):
    """for add"""

    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_positions = config.get("n_positions", 1024)
        self.vocab_size = config.get("vocab_size", 1024)
        self.pad_idx = config.get("pad_idx", 0)
        self.length = int(1e5)

    def __len__(self):
        return self.length

    def process(self, text):
        words = list(text)
        words = [w for w in words if w != " "]
        words = [w for w in words if w != ""]
        words.append(self.tokenizer.sep_token)

        label_words = []
        flag = 0
        for w in words:
            if w == "=":
                flag = 1
                continue
            if flag == 1:
                label_words.append(w)

        # convert_tokens_to_id
        ints = self.tokenizer.convert_tokens_to_id(words)
        ints2 = self.tokenizer.convert_tokens_to_id(label_words)

        # pad
        inputs = ints[-(self.n_positions - 1) :]
        labels = ints2[-(self.n_positions - 1) :]

        return inputs, labels

    def __getitem__(self, idx):
        a = random.randint(0, 1e4)
        b = random.randint(0, 1e4)
        if random.random() < 0.5:
            c = a + b
            text = f"{a}+{b}={c}"
        else:
            c = a - b
            text = f"{a}-{b}={c}"

        return self.process(text)

    def _collate_fn(self, batch):
        """pad seq"""

        inputs = []
        labels = []
        maxlen = max([len(w[0]) for w in batch]) - 1

        for btc_idx in range(len(batch)):
            input, label = batch[btc_idx]
            label = [-100] * (len(input) - len(label)) + label

            inputs.append(pad_list(input[:-1], maxlen=maxlen, pad=self.pad_idx))
            labels.append(pad_list(label[1:], maxlen=maxlen, pad=-100))
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )

    def make_dataloader(self, batch_size=16, num_workers=16):
        train_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        while True:
            for data in train_loader:
                yield data
