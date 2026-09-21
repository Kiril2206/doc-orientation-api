"""Backbone factory and versioned checkpoint contract."""
import torch
from torch import nn
from torchvision import models

BACKBONES = ("resnet18", "resnet34", "resnet50", "efficientnet_b0", "convnext_tiny")


def build_model(backbone="efficientnet_b0", pretrained=True):
    weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    if backbone not in weights:
        raise ValueError(f"Unsupported backbone {backbone}")
    model = getattr(models, backbone)(weights=weights[backbone] if pretrained else None)
    if backbone.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, 4)
    else:
        last = len(model.classifier) - 1
        model.classifier[last] = nn.Linear(model.classifier[last].in_features, 4)
    return model


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" not in checkpoint:
        return {"model_state": checkpoint, "config": {
            "backbone": "resnet18", "input_size": 224, "resize_mode": "stretch",
            "model_version": "v1", "angles_ccw": [0, 90, 180, 270]}}
    config = checkpoint["config"]
    if config.get("angles_ccw") != [0, 90, 180, 270]:
        raise ValueError("Unsupported checkpoint class ordering")
    return checkpoint
