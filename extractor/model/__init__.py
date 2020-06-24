from .model import extracting, build_model
from train import train


def init_model(cfg):
    build_model(cfg)
