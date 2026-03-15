import json
import os
from io import BytesIO
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = os.getenv("DEVICE", "cpu")

# ImageNet class labels (imagenet-simple-labels.json)
# IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
_BASE_DIR = Path(__file__).parent
_LABELS_PATH = _BASE_DIR / "imagenet-simple-labels.json"
_LABELS_KO_PATH = _BASE_DIR / "imagenet_ko.json"

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
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.device = torch.device(DEVICE)

    def load(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device)
        self.model.eval()

    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_image(self, image_bytes: bytes) -> list[dict]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

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


model_manager = ModelManager()
