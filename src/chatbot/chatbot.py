import io
import os

import requests
import torch
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

    def load(self, model: str, low_disk_usage: bool = False) -> None:
        """Load  state dict from huggingface repo or local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                Can be either:
                    path of a pretrained model.
                    model repo.
            low_disk_usage (bool):
                If True : model will download state dict file to disk.
                Only needed when downloading from hub.
                Defaults to False.

        Raises:
            ValueError: str model should be a path!
        """
        url = ""
        if model in self._PRETRAINED_LIST:
            model, url = self.download(model, low_disk_usage=low_disk_usage)
        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(
                    model_dir=model, url=url, low_disk_usage=low_disk_usage
                )
            else:
                raise ValueError("""str model should be a dir!""")

        else:
            raise ValueError("""str model should be a dir!""")

    def _load_from_dir(
        self,
        model_dir: str,
        url: str,
        low_disk_usage: bool = False,
    ) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """
        model_files = os.listdir(model_dir)

        # config
        if "config.json" not in model_files:
            raise FileNotFoundError("""config should in model dir!""")

        # download model
        if low_disk_usage:
            state_dict = torch.load(
                io.BytesIO(requests.get(url).content), map_location=torch.device("cpu")
            )
            self.model = BloomForCausalLM.from_pretrained(
                pretrained_model_name_or_path=None,
                state_dict=state_dict,
                config=f"{model_dir}/config.json",
            )

        # model
        if self.gpu:
            self.model = BloomForCausalLM.from_pretrained(
                model_dir,
            )
            self.model.half()
            self.model.cuda()

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
