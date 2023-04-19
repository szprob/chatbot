import os
import pickle
import tarfile
import tempfile
from abc import abstractmethod
from typing import Dict, Optional, Union

import huggingface_hub

TEMP_PATH = "/tmp/.chatbot"


class PreTrainedModule:

    """Pretrained module for all modules.

    Basic class takes care of storing the configuration of the models
    and handles methods for loading ,downloading and saving.

    """

    _TEMP_PATH = TEMP_PATH
    _PRETRAINED_LIST = [
        "szzzzz/chatbot_bloom_1b7",
        "szzzzz/chatbot_bloom_560m",
        "szzzzz/chatbot_bloom_3b",
        "szzzzz/chatbot_bloom_7b",
    ]

    def __init__(
        self,
    ) -> None:
        # Model temp dir for documents or state dicts
        if not os.path.exists(TEMP_PATH):
            os.mkdir(TEMP_PATH)
        self._tmpdir = tempfile.TemporaryDirectory(prefix=f"{TEMP_PATH}/")

    @abstractmethod
    def load(self, model: Union[str, Dict], low_disk_usage: bool = False) -> None:
        """Load  state dict from local model path.

        Args:
            model (str):
                Model file need to be loaded.
        """

        pass

    def _load_pkl(self, path: str) -> Dict:
        with open(path, "rb") as f:
            file = pickle.load(f)
        return file

    def _save_pkl(self, file: Dict, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        with open(path, "wb") as f:
            pickle.dump(file, f)

    def _zip_dir(self, dir: str, path: str) -> None:
        tar = tarfile.open(path, "w")
        for files in os.listdir(dir):
            tar.add(os.path.join(dir, files), arcname=files)
        tar.close()

    def _unzip2dir(self, file: str, dir: Optional[str] = None) -> None:
        if dir is None:
            dir = self._tmpdir.name
        if not os.path.isdir(dir):
            raise ValueError("""`dir` shoud be a dir!""")
        tar = tarfile.open(file, "r")
        tar.extractall(path=dir)
        tar.close()

    def download(
        self,
        repo_id: str,
        cache_dir: Optional[str] = None,
        low_disk_usage: bool = False,
    ) -> str:
        """download model from huggingface and return local dir"""
        if cache_dir is None:
            cache_dir = self._tmpdir.name
        if low_disk_usage:
            local_path = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                ignore_patterns="*.bin",
            )
            url = huggingface_hub.file_download.hf_hub_url(repo_id, "pytorch_model.bin")
        else:
            local_path = huggingface_hub.snapshot_download(
                repo_id=repo_id, cache_dir=cache_dir
            )
            url = ""
        return local_path, url
