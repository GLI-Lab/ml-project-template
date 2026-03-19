"""ResNet50 inference + evaluation script — adapt for your dataset and model."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from train.utils import IMAGENET_TRANSFORM, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", model_path)
    return model


def predict_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> int:
    image = Image.open(image_path).convert("RGB")
    tensor = IMAGENET_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(1).item()


def run_inference(config: dict):
    device = torch.device(config["device"])
    data_dir = Path(config["data_dir"])
    model_path = Path(config["model_dir"]) / "model.pt"

    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return

    num_classes = 10
    model = load_model(str(model_path), num_classes, device)

    image_files = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        logger.warning("No images found in %s", data_dir)
        return

    logger.info("Running inference on %d images...", len(image_files))

    predictions = []
    for image_path in image_files:
        pred = predict_image(model, image_path, device)
        predictions.append((image_path.name, pred))
        logger.info("  %s → class %d", image_path.name, pred)

    correct = 0
    labeled = 0
    for name, pred in predictions:
        try:
            true_label = int(name.split("_")[0])
            labeled += 1
            if pred == true_label:
                correct += 1
        except (ValueError, IndexError):
            pass

    if labeled > 0:
        accuracy = correct / labeled
        logger.info("Accuracy: %d/%d = %.4f", correct, labeled, accuracy)
    else:
        logger.info("No ground-truth labels found. Predictions logged above.")


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    run_inference(config)
