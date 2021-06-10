from collections import namedtuple

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
import faiss
import urllib
import zipfile
import os

model = namedtuple("model", ["url", "model"])
models = {
    "retrieve_rank": model(
        url="https://github.com/Nandhagopalan/Semanticsearch/releases/download/0.0.1/retrieve_rerank.zip",
        model=SentenceTransformer,
    ),
    "faiss_index":model(
        url="https://github.com/Nandhagopalan/Semanticsearch/releases/download/0.0.1/search.index.zip",
        model=faiss
    )
}

def list_models():
    """
    Print all available pretrained models
    """
    return list(models.keys())


def get_model(embedder,faissix):
    """
    Load the pretrained weights and return search results
    Example:
    query = ''
    model: RetrieveRerank = get_model("retrieve_rank")

    model.query_as_json(query)
    """
    if not models.get(embedder):
        raise Exception("Model name not found!")

    model_class = models[embedder].model(models[embedder].url)
    
    ### check if file exist and dont download again
    if not os.path.isfile('../model/search.index'):
        urllib.request.urlretrieve(models[faissix].url, "search.index.zip")

        with zipfile.ZipFile("search.index.zip", 'r') as zip_ref:
            zip_ref.extractall('../model/')

        os.remove("search.index.zip")

    index=models[faissix].model.read_index('../model/search.index')
    
    return model_class,index
