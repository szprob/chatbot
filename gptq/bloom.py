# import math
# import time

# import torch
# import torch.nn as nn
# import transformers
# from modelutils import *
# from quant import *
# from transformers import BloomForCausalLM

# from gptq import *


# def get_bloom(model):
#     def skip(*args, **kwargs):
#         pass

#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip

#     model = BloomForCausalLM.from_pretrained(model, torch_dtype="auto")
#     model.seqlen = 2048
#     return model


# if __name__ == "__main__":
#     model_path = "/data/home/ze.song/models/chatbot/chatbot_bloom_1b7"
#     model = get_bloom(model_path)
#     model.eval()
