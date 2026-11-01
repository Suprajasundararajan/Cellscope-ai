import torch.nn as nn
from torchvision import models


def build_model(name: str, n_classes: int = 5, pretrained: bool = True):
    if name == "mobilenetv2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.last_channel, n_classes)
    elif name == "efficientnetb0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(name)
    return m


def cam_layer(model, name):
    return {"mobilenetv2": lambda m: m.features[-1],
            "efficientnetb0": lambda m: m.features[-1],
            "resnet50": lambda m: m.layer4[-1]}[name](model)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
