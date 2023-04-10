# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""

import argparse
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # ddp setting
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_file", default="config/model_config.json", type=str
    )
    parser.add_argument("--deepspeed", default="config/deepspeed_config.json", type=str)
    parser.add_argument(
        "--use_lora", action="store_true", default=False, help="Use lora"
    )
    parser.add_argument(
        "--lora_hyperparams_file", default="config/lora_hyperparams.json", type=str
    )
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    # ddp
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}

    # model
    model_config = json.load(open(args.model_config_file))
    if local_rank == 0:
        logger = get_logger("train", model_config["output_dir"])
        logger.info(f"args.__dict__ : {args.__dict__}")
        for key, value in model_config.items():
            logger.info(f"{key} : {value}")

    # load_in_8bit
    load_in_8bit = True if args.use_lora else False

    tokenizer = BloomTokenizerFast.from_pretrained(model_config["model_name_or_path"])
    tokenizer.pad_token_id = model_config["pad_token_id"]
    tokenizer.padding_side = model_config["padding_side"]

    model = BloomForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        device_map=device_map,
        load_in_8bit=load_in_8bit,
    )

    data_path = model_config["data_path"]
    # data_path = '/data/home/ze.song/data/corpus/dialogue'
    data_set = dataset.DataSet(
        config=model_config, tokenizer=tokenizer, data_path=data_path
    )

    # lora
    if args.use_lora:
        model = prepare_model_for_int8_training(model)
        lora_hyperparams = json.load(open(args.lora_hyperparams_file))
        for key, value in lora_hyperparams.items():
            logger.info("{} : {}".format(key, value))
        lora_config = LoraConfig(
            r=lora_hyperparams["lora_r"],
            lora_alpha=lora_hyperparams["lora_alpha"],
            target_modules=lora_hyperparams["lora_target_modules"],
            lora_dropout=lora_hyperparams["lora_dropout"],
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
        deepspeed=args.deepspeed if not args.use_lora else None,
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
    if args.use_lora:
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
