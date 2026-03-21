from pathlib import Path
from typing import Callable

import torch

from api.models.base import ModelManager

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


class ViTManager(ModelManager):
    def __init__(
        self,
        name: str,
        model_fn: Callable,
        weights,
        weights_path: Path | None = None,
    ):
        super().__init__(name=name)
        self._model_fn = model_fn
        self._weights = weights
        self._weights_path = weights_path

    def get_config(self) -> dict:
        return {
            "architecture": self._model_fn.__name__,
            "weights": str(self._weights),
            "input_size": 224,
            "num_classes": 1000,
            "device": str(self.device),
        }

    def load(self):
        if self._weights_path is not None:
            self.model = self._model_fn(weights=None)
            state_dict = torch.load(self._weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            self.model = self._model_fn(weights=self._weights)
        self.model.to(self.device)
        self.model.eval()
