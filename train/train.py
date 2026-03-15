"""Training script skeleton — adapt for your dataset and model."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "train/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DummyDataset(Dataset):
    """Replace with your real dataset."""

    def __init__(self, data_dir: str, transform=None, size: int = 100):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Placeholder: random image tensor and label
        image = torch.rand(3, 224, 224)
        label = idx % 10
        return image, label


def train(config: dict):
    set_seed(config["seed"])

    device = torch.device(config["device"])
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DummyDataset(config["data_dir"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Replace final layer for your number of classes
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    logger.info("Starting training: epochs=%d batch_size=%d lr=%s device=%s",
                config["epochs"], config["batch_size"], config["lr"], device)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / len(dataset)
        logger.info("Epoch [%d/%d] loss=%.4f accuracy=%.4f", epoch, config["epochs"], avg_loss, accuracy)

    save_path = model_dir / "model.pt"
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved to %s", save_path)


if __name__ == "__main__":
    config = load_config()
    train(config)
