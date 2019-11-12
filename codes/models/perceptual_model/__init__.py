from .vgg import *
from .resnet import *


def build_perceptual_model(name, model_kwargs=None):
    return globals()[name](model_kwargs=model_kwargs)
