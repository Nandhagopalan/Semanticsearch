from collections import namedtuple

import cv2
import numpy as np
import torch
from sentence_transformers import (
    CrossEncoder,
    InputExample,
    SentenceTransformer,
    datasets,
    losses,
    models,
    util,
)

model = namedtuple("model", ["url", "model"])
models = {
    "retrieve_rank": model(
        url="https://github.com/Nandhagopalan/Semanticseach/releases/download/0.0.1/retrieve_rerank.zip",
        model=SentenceTransformer,
    )
}


def list_models():
    """
    Print all available pretrained models
    """
    return list(models.keys())


def get_model(name):
    """
    Load the pretrained weights and return search results
    Example:
    query = ''
    model: RetrieveRerank = get_model("retrieve_rank")

    model.predit_as_json(query)
    """
    if not models.get(name):
        raise Exception("Model name not found!")

    model_class = models[name].model(models[name].url)
    return model_class
