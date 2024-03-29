# -*- coding: utf-8 -*-
"""
Created on 20230109
@author: sz

"""

import argparse
import json

import dataset
import deepspeed
import numpy as np
import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast


def add_argument():
    parser = argparse.ArgumentParser(description="sz")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # ddp setting
    args = add_argument()

    model_config = json.load(open("config/model_config.json"))
    tokenizer = BloomTokenizerFast.from_pretrained(model_config["model_name_or_path"])
    tokenizer.pad_token_id = model_config["pad_token_id"]
    tokenizer.padding_side = model_config["padding_side"]

    # dataset
    data_path = model_config["data_path"]
    # data_path = '/data/home/ze.song/data/corpus/dialogue'
    data_set = dataset.DataSet(
        config=model_config, tokenizer=tokenizer, data_path=data_path, all=False
    )
    # trainloader = data_set.make_dataloader(batch_size=4)

    # model
    model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-3b",
        # "bigscience/bloom-1b7",
    )
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    model.transformer.word_embeddings.weight.requires_grad = False

    # opt = torch.optim.AdamW(
    #     model.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.001
    # )
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_rule)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        training_data=data_set,
        collate_fn=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    for epoch in range(1):
        total_loss = []
        for i, data in enumerate(trainloader):
            input_ids = data["input_ids"].to(model_engine.local_rank)
            attention_mask = data["attention_mask"].to(model_engine.local_rank)
            labels = data["labels"].to(model_engine.local_rank)
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            model_engine.backward(loss)
            model_engine.step()

            # print statistics
            if i % 100 == 0 and args.local_rank == 0:
                loss_num = loss.item()
                total_loss.append(loss_num)
                log = (
                    f"iter: {i}",
                    f"avg_loss : {round(np.mean(total_loss[-200:]),3)}",
                    f"loss : {round(loss_num,3)}",
                )
                print(log)

            if i == 8888:
                print("save model")
                model_engine.save_checkpoint(model_config["output_dir"])
                print("save model finished")

            if i % 40001 == 40000:
                print("save model")
                model_engine.save_checkpoint(model_config["output_dir"])
                print("save model finished")

        print("save model")
        model_engine.save_checkpoint(model_config["output_dir"])
        print("save model finished")
