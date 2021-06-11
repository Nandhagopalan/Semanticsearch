# Semantic search with FAISS

The idea of this project is to build a semantic search engine which can search across multiple research papers related to covid and return the response. This can pretty much help people who want to know about ongoing research with respect to covid'19

I have used - `retrieval-ranking method with faiss index` approach for the faster retrieval of data for the given query.


## Installation

`pip install semantic-search-faiss`


## Inference example

```python
from semanticsearch import search,utils,config
from semanticsearch.pretrained import get_model
from sentence_transformers import CrossEncoder

bi_encoder,index,documents=get_model(config.BI_ENCODER,config.INDEX,config.DATA)
cross_encoder = CrossEncoder(config.CROSS_ENCODER)

query='death rates of covid'
results=search.search(query,index,bi_encoder,cross_encoder,documents)

```

## Training pipeline

```

1. Synthetic query generation using T5
2. Finetuning Bi-encoder using the synthetic query
3. Indexing the data with FAISS using finetuned BI-encoder
4. Bi-encoder + Cross encoder with FAISS search

```
Try out the code on google colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f4fm6RD08Llc15kG7q-wCt7YQUi-aAn7?usp=sharing)


## Kaggle

Detailed walk through of the solution can be found in the below kaggle notebook

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/nandhuelan/semantic-search)



## Acknowledgements

I would like to thank Kaggle community as a whole for providing an avenue to learn and discuss latest data science/machine learning advancements.

1. Vladimir Iglovikov for his wonderful article ["I trained a model. What is next?"](https://ternaus.blog/tutorial/2020/08/28/Trained-model-what-is-next.html)

2. [Xhululu](https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv) for the dataset.
