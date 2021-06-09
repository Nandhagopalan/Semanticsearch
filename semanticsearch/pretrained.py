from collections import namedtuple

import cv2
import numpy as np
import torch
from PIL import Image

from cassava.model import CassavaClassifier

model = namedtuple("model", ["url", "model"])
models = {
    "tf_efficientnet_b4": model(
        url="https://github.com/p-s-vishnu/cassava-leaf-disease-classification/releases/download/v0.1-efficientnet-b4/tf_efficientnet_b4_fold0.zip",  # noqa
        model=CassavaClassifier,
    )
}


def list_models():
    """
    Print all available pretrained models
    """
    return list(models.keys())


def get_model(name):
    """
    Load the pretrained weights and return model
    Example:
    image = Image.open(<image path>)
    image = np.array(image)
    model: CassavaClassifier = get_model("tf_efficientnet_b4")

    model.predit_as_json(image)
    """
    if not models.get(name):
        raise Exception("Model name not found!")
    model_class = models[name].model(model_name=name)
    state_dict = torch.hub.load_state_dict_from_url(models[name].url, map_location="cpu")
    model_class.load_state_dict(state_dict)
    return model_class
