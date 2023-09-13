# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""


import json
import shutil

# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import torch
from huggingface_hub import HfApi
from torch.amp import autocast
from transformers import (
    BloomForCausalLM,
    BloomForSequenceClassification,
    BloomTokenizerFast,
)

api = HfApi()

api.upload_folder(
    folder_path="/data/home/ze.song/models/chatbot/chatbot_bloom_560m",
    repo_id="szzzzz/chatbot_bloom_560m",
    repo_type="model",
    # ignore_patterns = "*bin",
)

file_path = "/data/home/ze.song/models/chatbot/chatbot_bloom_560m/pytorch_model.bin"
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="pytorch_model.bin",
    repo_id="szzzzz/chatbot_bloom_560m",
    repo_type="model",
)


@torch.no_grad()
def talk(x, tokenizer, model, gpu=True):
    inputs = "Human: \n" + x + "\n\nAssistant: \n"
    inputs = tokenizer.bos_token + inputs
    input_ids = tokenizer.encode(inputs, return_tensors="pt")
    if gpu:
        input_ids = input_ids.to(device)
    _, input_len = input_ids.shape
    pred_ids = model.generate(
        input_ids,
        eos_token_id=2,
        pad_token_id=3,
        bos_token_id=1,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        max_new_tokens=1023 - input_len,
        min_new_tokens=10,
        repetition_penalty=1.2,
    )

    pred = pred_ids[0][input_len - 4 :]
    res = tokenizer.decode(pred, skip_special_tokens=True)
    print(x[:-12])
    print()
    print(res)


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model_config = json.load(open("config/model_config.json"))
    # load_in_8bit
    model_path = "/data/home/ze.song/models/chatbot/3b"
    model_path = "/data/home/ze.song/models/chatbot/chatbot_bloom_3b_back"
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    tokenizer = BloomTokenizerFast.from_pretrained(
        "bigscience/bloomz-7b1-mt",
        cache_dir="/data/home/ze.song/models/chatbot/7b1-mt",
    )
    model = BloomForCausalLM.from_pretrained(
        "bigscience/bloomz-7b1-mt",
        cache_dir="/data/home/ze.song/models/chatbot/7b1-mt",
    )

    model = BloomForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16,
    )
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    path = "/data/home/ze.song/models/tmp2"
    with open(f"{path}/latest", "r") as file:
        name = file.read()
    states_path = f"{path}/{name}/mp_rank_00_model_states.pt"
    print(states_path)
    model.load_state_dict(torch.load(states_path, map_location="cpu")["module"])

    model.half()
    model.to(device)
    tokenizer.save_pretrained("/data/home/ze.song/models/chatbot/tmp")
    model.save_pretrained("/data/home/ze.song/models/chatbot/tmp")
    shutil.move(
        "/data/home/ze.song/models/chatbot/tmp/pytorch_model.bin",
        "/data/home/ze.song/models/chatbot/chatbot_bloom_3b/pytorch_model.bin",
    )

    x = """把下面这句话翻译成中文 :
        Main hand"""
    x = "把下面这句话翻译成英文:我爱你"
    x = """tell me which of the nicknames bellow is toxic :
        1. [ＦㅤUㅤꮯㅤᴋ]
        2. [MÓȚHÉŘĆHÓĐㅤ]
        3. [hacker,hackd]
        4. [चुत_की_रानी√]
        5. [✓XXX®_VIDEO™]
        6. [KALAㅤLUN̾D✭]
        7. [Izm7I2p9N2&1]
        """

    x = """Translate these words into Chinese , only reply the translated result :
        1. city girl
        2. cheerful lad
        3. Marshmellow
        4. Caramel Girl
        5. hello

    """

    x = """summarize this paragraph in shortest woirds:
        Life is a chess-board The chess-board is the world: the pieces are the phenomena of the universe; the rules of the game are what we call the laws of nature. The player on the other side is hidden from us. We know that his play is always fair, just and patient. But also we know, to our cost, that he never overlooks a mistake, or makes the smallest allowance for ignorance.
    """

    x = """translate this into english:
        我爱你,但是我恨你"""

    talk(x, tokenizer, model, gpu=True)
