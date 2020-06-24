import torch
import torchvision.models as torch_models
from torch import nn
import copy
import numpy as np
from utils.singleton import SingletonInstance
from .transforms import transforms_test


class Model(nn.Module, SingletonInstance):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model = torch_models.resnet50(pretrained=True)
        self.model = nn.Sequential(
            model.conv1,
            model.bn1,
            nn.ReLU(True),
            model.layer1,
            model.layer2,
            model.layer3,   
        )
        self.class_head = nn.Sequential(
            model.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, num_classes)
        )
        self.signature_head = nn.Sequential(
            copy.deepcopy(model.layer4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        features = self.signature_head(x)
        clas = self.class_head(x)
        return features, clas


def build_model(cfg, from_scratch=False):
    model = Model.instance(cfg.MODEL.NUM_CLASSES)
    model.to(cfg.MODEL.DEVICE)
    loaded = torch.load(cfg.MODEL_PATH)
    model.load_state_dict(loaded['model'])
    return model


def build_training_model(cfg):
    model = Model(cfg.MODEL.num_classes)
    model.to(cfg.MODEL.DEVICE)
    return model


def load_new_weights(cfg, new_path):
    model = Model.instance(cfg.MODEL.NUM_CLASSES)
    model.to(cfg.MODEL.DEVICE)
    loaded = torch.load(new_path)
    model.load_state_dict(loaded['model'])
    return model


def extracting(cfg, image):
    with torch.no_grad():
        model.eval()
        model = Model.instance(cfg.MODEL.NUM_CLASSES)
        # assert isinstance(image, np.ndarray)
        # image = torch.from_numpy(image)
        image = transforms_test(image)
        image = image.to(cfg.MODEL.DEVICE)
        return model(image)
