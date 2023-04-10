from transformers import T5Config, T5ForConditionalGeneration

from chatbot.tokenization.tokenization import Tokenizer

tok = Tokenizer()
tok.load("./cn_vocab.pkl")


config = {
    "vocab_size": 21128,
    "n_positions": 512,
    "source_max_token_len": 512,
    "target_max_token_len": 256,
    "hidden_size": 256,
    "n_layers": 4,
    "num_heads": 8,
    "pad_idx": 0,
}


conf = T5Config(
    vocab_size=21128,
    d_model=1024,
    d_ff=4096,
    eos_token_id=102,
    num_heads=16,
    num_layers=24,
    pad_token_id=0,
)
conf

m = T5ForConditionalGeneration.from_pretrained("t5-large")
print("Total Parameters:", sum([p.nelement() for p in m.parameters()]))

m.config

m2 = T5ForConditionalGeneration(conf)
print("Total Parameters:", sum([p.nelement() for p in m2.parameters()]))
