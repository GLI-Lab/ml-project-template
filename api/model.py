import json
import os
from io import BytesIO
from pathlib import Path
from typing import Callable

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = os.getenv("DEVICE", "cpu")
_MODELS_DIR = Path(__file__).parent.parent / "models"

_DATA_DIR = Path(__file__).parent.parent / "dataset" / "imagenet"
_LABELS_PATH = _DATA_DIR / "imagenet-simple-labels.json"
_LABELS_KO_PATH = _DATA_DIR / "imagenet_ko.json"

_imagenet_labels: list[str] | None = None
_imagenet_ko: dict[str, str] | None = None


def _load_imagenet_labels() -> list[str]:
    global _imagenet_labels
    if _imagenet_labels is not None:
        return _imagenet_labels
    with open(_LABELS_PATH, encoding="utf-8") as f:
        _imagenet_labels = json.load(f)
    return _imagenet_labels


def _load_imagenet_ko() -> dict[str, str]:
    global _imagenet_ko
    if _imagenet_ko is not None:
        return _imagenet_ko
    with open(_LABELS_KO_PATH, encoding="utf-8") as f:
        _imagenet_ko = json.load(f)
    return _imagenet_ko


_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ModelManager:
    def __init__(self, name: str, model_fn: Callable, weights, weights_path: Path | None = None):
        self.name = name
        self._model_fn = model_fn
        self._weights = weights
        self._weights_path = weights_path
        self.model: torch.nn.Module | None = None
        self.device = torch.device(DEVICE)

    def load(self):
        if self._weights_path is not None:
            self.model = self._model_fn(weights=None)
            state_dict = torch.load(self._weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            self.model = self._model_fn(weights=self._weights)
        self.model.to(self.device)
        self.model.eval()

    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_image(self, image_bytes: bytes) -> list[dict]:
        if not self.is_loaded:
            raise RuntimeError(f"Model '{self.name}' is not loaded")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = _transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top5_probs, top5_indices = torch.topk(probabilities, 5)
        labels = _load_imagenet_labels()
        labels_ko = _load_imagenet_ko()

        return [
            {
                "label": labels[idx.item()],
                "label_ko": labels_ko.get(labels[idx.item()]),
                "confidence": round(prob.item(), 4),
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]


model_managers: list[ModelManager] = [
    ModelManager(
        name="resnet50",
        model_fn=models.resnet50,
        weights=models.ResNet50_Weights.IMAGENET1K_V1,
    ),
    ModelManager(
        name="efficientnet_b0",
        model_fn=models.efficientnet_b0,
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
    ModelManager(
        name="efficientnet_b3",
        model_fn=models.efficientnet_b3,
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    ),
    # 로컬 가중치 사용 예시: models/resnet50/model.pt
    # ModelManager(
    #     name="resnet50_custom",
    #     model_fn=models.resnet50,
    #     weights=None,
    #     weights_path=_MODELS_DIR / "resnet50" / "model.pt",
    # ),
]
