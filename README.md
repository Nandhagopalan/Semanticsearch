# Cassava leaf disease classification
[![PyPI version shields.io](https://img.shields.io/badge/pypi-0.0.2-blue)](https://pypi.org/project/cassava-classifier/)  [![Downloads](https://pepy.tech/badge/cassava-classifier)](https://pepy.tech/project/cassava-classifier)  


The idea of this project is to build an image classifier to find out healthy and diseased cassava leaves.

There are 4 different classes of leaf diseases namely - `Cassava Bacterial Blight (CBB),Cassava Brown Streak Disease (CBSD),Cassava Green Mottle (CGM) and Cassava Mosaic Disease (CMD)`.


## Web app
[![Open Web App in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/p-s-vishnu/cassava_app/main/web/app.py)
<img src="static/cassava.gif" alt="inference" style="width:80%;" />


## Swagger documentation for API
[![API Link](https://img.shields.io/badge/Launch%20Cassava%20API-Swagger-blue?style=for-the-badge&logo=microsoft%20azure)](http://52.224.254.7:8003/docs)
<img src="static/api.gif" alt="inference" style="width:80%;" />


## Installation

`pip install cassava-classifier`



## Inference example

```python
import PIL import Image
from cassava.pretrained import get_model

image = Image.open("<insert your image path here>")

# Use cassava.list_models() to list of available trained models
model = get_model(name:str)
model.predict_as_json(image: np.array)
>> {"class_name":str, "confidence": np.float}

```
Try out the inference code either on google colab or kaggle.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gPLY6nqF6P4WdvIRIAH_aYQn-iWkzvqs?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/vpkprasanna/cassava-inference-from-pypi)

## Other details
- Training data can be found on the [Kaggle competition page](https://www.kaggle.com/c/cassava-leaf-disease-classification)

- Streamlit app code can be found [here](https://github.com/p-s-vishnu/cassava_app).

[Github discussion forum](https://github.com/p-s-vishnu/cassava-leaf-disease-classification/discussins)


## Blog

[Medium link]



## Acknowledgements

We would like to thank Kaggle community as a whole for providing an avenue to learn and discuss latest data science/machine learning advancements but a hat tip to whose code was used / who inspired us.

1. Vladimir Iglovikov for his wonderful article ["I trained a model. What is next?"](https://ternaus.blog/tutorial/2020/08/28/Trained-model-what-is-next.html)

2. [Y. Nakama](https://www.kaggle.com/yasufuminakama) for the baseline code.


<a href="https://www.buymeacoffee.com/vpkprasanna" target="_blank"><img src="https://github.com/appcraftstudio/buymeacoffee/raw/master/Images/snapshot-bmc-button.png" width="300"></a>
