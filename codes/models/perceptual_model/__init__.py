from .vgg import *


def build_perceptual_model(name):
    return globals()[name]()
