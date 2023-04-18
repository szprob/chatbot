import os

from transformers import BloomForCausalLM, BloomTokenizerFast

from chatbot.module_utils import PreTrainedModule


class Bot(PreTrainedModule):
    """chatbot .

    Attributes:
        gpu (bool):
            device.
            Defaults to False.
    """

    def __init__(
        self,
        *,
        gpu: bool = False,
    ) -> None:

        super().__init__()
        self.gpu = gpu
        self.tokenizer = None
        self.model = None
        self.max_length = 1024

    def load(self, model: str) -> None:
        """Load  state dict from huggingface repo or local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                Can be either:
                    path of a pretrained model.
                    model repo.

        Raises:
            ValueError: str model should be a path!
        """
        if model in self._PRETRAINED_LIST:
            model = self.download(model)
        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "chatbot")
                if os.path.exists(dir):
                    pass
                else:
                    os.mkdir(dir)
                self._unzip2dir(model, dir)
                self._load_from_dir(dir)
            else:
                raise ValueError("""str model should be a path!""")

        else:
            raise ValueError("""str model should be a path!""")

    def _load_from_dir(self, model_dir: str) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """
        model_files = os.listdir(model_dir)

        # config
        if "config.json" not in model_files:
            raise FileNotFoundError("""config should in model dir!""")

        # model
        if self.gpu:
            self.model = BloomForCausalLM.from_pretrained(
                model_dir,
            )
            self.model.half()
            self.model.cuda()
        else:
            self.model = BloomForCausalLM.from_pretrained(
                model_dir,
            )
        self.model.eval()
        # tokenizer
        self.tokenizer = BloomTokenizerFast.from_pretrained(model_dir)

    def generate(self, inputs: str) -> str:
        """generate content on inputs .

        Args:
            inputs (str):
                example :'Human: 你好 .\n \nAssistant: '

        Returns:
            str:
                bot response
                example : '你好!我是你的ai助手!'

        """
        input_text = self.tokenizer.bos_token + inputs
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        _, input_len = input_ids.shape
        if input_len >= self.model.config.seq_length - 4:
            res = "对话超过字数限制,请重新开始."
            return res
        if self.gpu:
            input_ids = input_ids.cuda()
        pred_ids = self.model.generate(
            input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.8,
            max_new_tokens=self.model.config.seq_length - input_len,
            repetition_penalty=1.2,
        )
        pred = pred_ids[0][input_len:]
        res = self.tokenizer.decode(pred, skip_special_tokens=True)
        return res
