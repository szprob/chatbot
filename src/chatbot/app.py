import gradio as gr
import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

device = torch.device("cuda:0")
tokenizer = BloomTokenizerFast.from_pretrained(
    "/data/home/ze.song/models/chatbot/chatbot_bloom_3b",
)

model = BloomForCausalLM.from_pretrained(
    "/data/home/ze.song/models/chatbot/chatbot_bloom_3b",
    # torch_dtype=torch.float16,
)

model.half()
model.to(device)


def talk(x, tokenizer, model, gpu=True):
    max_length = 1024
    inputs = tokenizer.bos_token + x
    input_ids = tokenizer.encode(inputs, return_tensors="pt")
    _, input_len = input_ids.shape
    if input_len >= max_length - 3:
        res = "对话超过字数限制,请重新开始."
        return res

    if gpu:
        input_ids = input_ids.to(device)
    pred_ids = model.generate(
        input_ids,
        eos_token_id=2,
        pad_token_id=3,
        bos_token_id=1,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        max_new_tokens=max_length - input_len,
        min_new_tokens=10,
        repetition_penalty=1.2,
    )
    pred = pred_ids[0][input_len:]
    res = tokenizer.decode(pred, skip_special_tokens=True)
    return res


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    prompt = ""
    for i, h in enumerate(history):
        prompt = prompt + "\nHuman: " + h[0]
        if i != len(history) - 1:
            prompt = prompt + "\nAssistant: " + h[1]
        else:
            prompt = prompt + "\nAssistant: "

    response = talk(prompt, tokenizer, model)
    history[-1][1] = response
    return history


def regenerate(history):
    prompt = ""
    for i, h in enumerate(history):
        prompt = prompt + "\nHuman: " + h[0]
        if i != len(history) - 1:
            prompt = prompt + "\nAssistant: " + h[1]
        else:
            prompt = prompt + "\nAssistant: "

    response = talk(prompt, tokenizer, model)
    history[-1][1] = response
    return history


with gr.Blocks() as demo:
    gr.Markdown("""chatbot .""")

    with gr.Tab("chatbot"):
        gr_chatbot = gr.Chatbot([], elem_id="chatbot").style(height=300)

        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter",
        ).style(container=False)
        with gr.Row():
            clear = gr.Button("重新开始新的对话")
            regen = gr.Button("重新生成当前回答")

        # func
        txt.submit(add_text, [gr_chatbot, txt], [gr_chatbot, txt]).then(
            bot, gr_chatbot, gr_chatbot
        )

        clear.click(lambda: None, None, gr_chatbot, queue=False)
        regen.click(regenerate, [gr_chatbot], [gr_chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True, server_port=7862)
