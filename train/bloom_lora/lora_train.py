# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""

import json

# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import os
import sys

import dataset

# import nets
import torch
import transformers
from logger import get_logger
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import BloomForCausalLM, BloomTokenizerFast

if __name__ == "__main__":
    torch.cuda.device_count()
    torch.backends.cudnn.is_available()
    torch.backends.cudnn.version()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
    # torch.cuda.set_device(3)

    # ddp
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}

    # model
    model_config = json.load(open("config/model_config.json"))
    if local_rank == 0:
        logger = get_logger("train", model_config["output_dir"])
        for key, value in model_config.items():
            logger.info(f"{key} : {value}")

    # load_in_8bit

    tokenizer = BloomTokenizerFast.from_pretrained(model_config["model_name_or_path"])
    tokenizer.pad_token_id = model_config["pad_token_id"]
    tokenizer.padding_side = model_config["padding_side"]

    model = BloomForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=True,
    )

    data_path = model_config["data_path"]
    # data_path = '/data/home/ze.song/data/corpus/dialogue'
    data_set = dataset.DataSet(
        config=model_config, tokenizer=tokenizer, data_path=data_path
    )

    # lora
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(lora_config)
    model = get_peft_model(model, lora_config)

    # trainer
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=model_config["per_device_train_batch_size"],
        gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
        warmup_steps=model_config["warmup_steps"],
        num_train_epochs=model_config["num_epochs"],
        learning_rate=model_config["learning_rate"],
        fp16=True,
        logging_steps=model_config["logging_steps"],
        evaluation_strategy="steps" if model_config["val_set_size"] > 0 else "no",
        save_strategy="steps",
        eval_steps=model_config["eval_steps"]
        if model_config["val_set_size"] > 0
        else None,
        save_steps=model_config["save_steps"],
        output_dir=model_config["output_dir"],
        save_total_limit=5,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        deepspeed=None,
        group_by_length=False,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data_set,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=False)
    logger.info("Save checkpointing...")
    model.save_pretrained(model_config["output_dir"])
    tokenizer.save_pretrained(model_config["output_dir"])

    logger.info("Training succeeded")
