import warnings
from abc import abstractmethod
from pathlib import Path

import torch.nn

from utils import file_utils
from utils.util_deep_learning import device


class Model(torch.nn.Module):
    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""
        pass

    @classmethod
    def _fetch_model(cls, model_path: str):
        """
        Fetch the model file path. Can be overridden by subclasses to download models from remote.
        :param model_path: Path to the model file
        :return: Path to the model file
        """
        return Path(model_path)

    @classmethod
    def load(cls, model):
        """
        Loads the model from the given file.
        :param model: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location='cpu', weights_only=False)  # type: ignore[arg-type]

        model = cls._init_model_with_state_dict(state)
        
        if model:
            model.eval()
            model.to(device)

        return model
