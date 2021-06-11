import time

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sentence_transformers import (
    SentenceTransformer,
    util,
    CrossEncoder,
    InputExample,
    losses,
    models,
    datasets,
)
import torch
import os
import csv
import pickle
import time
import faiss
import glob
from pprint import pprint
from tqdm import tqdm
from torch import nn
import random
import gc
from semanticsearch import search,utils,config
from semanticsearch.pretrained import get_model
import urllib


# # Initializations
LOGGER = utils.init_logger()

def main():

    """
    Prepare: 1.index  2.embed  3.search 4.show
    """

    try:
        query=str(input("Please enter your query :::: "))
    except TypeError:
        print("TypeError: query should be a string")

    bi_encoder,index,passages=get_model(config.BI_ENCODER,config.INDEX,config.DATA)
    cross_encoder = CrossEncoder(config.CROSS_ENCODER)

    results=search.search(query,index,bi_encoder,cross_encoder,passages)

    


if __name__ == "__main__":
    main()
