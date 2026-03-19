"""ResNet50 training script — adapt for your dataset and model."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from train.utils import IMAGENET_TRANSFORM, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


class DummyDataset(Dataset):
    """Replace with your real dataset."""

    def __init__(self, data_dir: str, transform=None, size: int = 100):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.rand(3, 224, 224)
        label = idx % 10
        return image, label


def train(config: dict):
    set_seed(config["seed"])

    device = torch.device(config["device"])
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = DummyDataset(config["data_dir"], transform=IMAGENET_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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

        for images, labels in dataloader:
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
    config = load_config(CONFIG_PATH)
    train(config)
