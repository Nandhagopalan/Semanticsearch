import time

import utils
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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from torch import nn
import random
import gc

# # Initializations
OUTPUT_DIR = "/"
train = pd.read_csv("data/train.csv")

LOGGER = utils.init_logger()


class faiss_index:

    LOGGER.info("========== Creating: faiss index ==========")

    def __init__(self, data, model):
        self.data = data
        self.model = model

    # ====================================================
    # indexing
    # ====================================================

    def index(self):
        encoded_data = self.model.encode(self.data["abstract"].values.tolist())
        encoded_data = np.asarray(encoded_data.astype("float32"))
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.index.add_with_ids(encoded_data, np.array(range(0, len(self.data))))
        faiss.write_index(self.index, f"{OUTPUT_DIR}test.index")

    # ====================================================
    # fetch
    # ====================================================

    def fetch(self, idx):
        LOGGER.info("========== Fetch data ==========")
        info = self.data.iloc[idx]
        meta_dict = {}
        meta_dict["abstract"] = info["abstract"]
        return meta_dict

    def search(self, query, top_k):
        LOGGER.info("========== Search data ==========")
        t = time.time()
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        print(">>>> Results in Total Time: {}".format(time.time() - t))
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results = [self.fetch(idx) for idx in top_k_ids]
        return results


def main():

    """
    Prepare: 1.index  2.embed  3.search 4.show
    """


if __name__ == "__main__":
    main()
