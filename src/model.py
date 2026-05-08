"""
src/model.py
Model definitions: SimpleCNN, ResNet18, AlexNet
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleCNN(nn.Module):
    """Lightweight CNN for MNIST / CIFAR-10."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128) if in_channels == 1 else nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AlexNetFL(nn.Module):
    """Simplified AlexNet adapted for smaller FL inputs."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _replace_bn_with_gn(model: nn.Module, num_groups: int = 4) -> nn.Module:
    """
    Replace every BatchNorm2d in a model with GroupNorm.
    GroupNorm works correctly with batch_size=1 and small batches,
    whereas BatchNorm2d requires batch_size >= 2 during training.
    num_groups=4 is a safe default — must evenly divide the channel count.
    For channels < 4 we fall back to num_groups=1 (LayerNorm style).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            groups = min(num_groups, num_channels)
            # find largest divisor of num_channels that is <= num_groups
            while num_channels % groups != 0:
                groups -= 1
            setattr(model, name, nn.GroupNorm(groups, num_channels))
        else:
            _replace_bn_with_gn(module, num_groups)
    return model


def get_model(architecture: str, num_classes: int, dataset: str) -> nn.Module:
    """Factory function to instantiate the right model."""
    in_channels = 1 if dataset in ("MNIST", "FMNIST") else 3

    if architecture == "SimpleCNN":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

    elif architecture == "ResNet18":
        model = tv_models.resnet18(weights=None)
        if in_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # Replace BatchNorm with GroupNorm so training never crashes on
        # small / single-sample batches (FL clients often have few samples).
        model = _replace_bn_with_gn(model, num_groups=4)
        return model

    elif architecture == "AlexNet":
        return AlexNetFL(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_params(model: nn.Module):
    """Return model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: nn.Module, params):
    """Set model parameters from a list of numpy arrays."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
