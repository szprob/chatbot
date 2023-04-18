import gradio as gr

from chatbot import Bot

model = Bot()
model.load("szzzzz/chatbot_bloom_1b7")


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

    response = model.generate(prompt)
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

    response = model.generate(prompt)
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
            clear = gr.Button("Restart")
            regen = gr.Button("Regenerate response")

        # func
        txt.submit(add_text, [gr_chatbot, txt], [gr_chatbot, txt]).then(
            bot, gr_chatbot, gr_chatbot
        )

        clear.click(lambda: None, None, gr_chatbot, queue=False)
        regen.click(regenerate, [gr_chatbot], [gr_chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True, server_port=7862)
