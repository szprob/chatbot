import importlib as imp
import pickle

import gradio as gr
import textgen
import torch

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = torch.device(device)
n_device = torch.cuda.device_count()
torch.backends.cudnn.is_available()
torch.backends.cudnn.version()
torch.set_default_tensor_type(torch.FloatTensor)
torch.cuda.set_device(0)


def get_latest_model(path="/data/home/ze.song/models/gpt"):

    with open(
        f"{path}/logs.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    epoch = log[-1][0].replace("epoch : ", "")
    iter = log[-1][1].replace("iter: ", "")

    model_path = f"{path}/model_{epoch}_{iter}.pkl"
    return model_path


def get_latest_model2(path="/data/home/ze.song/models/gpt"):

    with open(
        f"{path}/logs.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    iter = log[-1][0].replace("iter: ", "")

    model_path = f"{path}/model_{iter}.pkl"
    return model_path


model_path = get_latest_model2("/data/home/ze.song/models/gpta")
model_path = "/data/home/ze.song/models/gpta/model_210000.pkl"
print(model_path)
vocab_path = (
    "/data/home/ze.song/git/block/src/block/bricks/train/gpt/model_files/vocab.pkl"
)
config = {
    "vocab_size": 13317,
    "embd_pdrop": 0.1,
    "n_embd": 1536,
    "n_head": 24,
    "n_positions": 320,
    "n_layer": 18,
    "attn_pdrop": 0.1,
    "resid_dropout": 0.1,
    "n_inner": 1536 * 4,
    "layer_norm_epsilon": 1e-5,
    "pad_idx": 0,
    "dtype": torch.float32,
    "segment_size": 3,
}


# imp.reload(chatbot)
# bot = chatbot.Bot(model_path=model_path, vocab_path=vocab_path, config=config)
imp.reload(textgen)
bot = textgen.Adder(model_path=model_path)
print(bot.model.lm_head.weight.device)


def predict(input):

    _ = bot.gen(input, top_k=1, top_p=0.6, max_num=20)
    return bot.his


with gr.Blocks() as demo:
    gr_chatbot = gr.Chatbot()

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="输入100+200,得到300").style(
            container=False
        )

    txt.submit(predict, txt, gr_chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True, server_port=7861)
