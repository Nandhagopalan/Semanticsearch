import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="semantic_search_faiss",
    version="0.0.9",
    author="Nandhagopalan Elangovan",
    description="Semantic search to query covid related papers",
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    license="MIT",
    keywords=["natural language processing", "semantic search", "pytorch"],
    url="https://github.com/Nandhagopalan/Semanticsearch",
    packages=find_packages(exclude=["tests", "docs", "images"]),
    install_requires=[
       "certifi==2021.5.30",
        "chardet==4.0.0",
        "click==8.0.1",
        "dataclasses",
        "faiss-cpu==1.7.1",
        "filelock==3.0.12",
        "huggingface-hub==0.0.8",
        "idna==2.10",
        "importlib-metadata==4.5.0",
        "joblib==1.0.1",
        "nltk==3.6.2",
        "numpy==1.19.5",
        "packaging==20.9",
        "pandas==1.1.5",
        "Pillow==8.2.0",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.1",
        "pytz==2021.1",
        "regex==2021.4.4",
        "requests==2.25.1",
        "sacremoses==0.0.45",
        "scikit-learn==0.24.2",
        "scipy==1.5.4",
        "sentence-transformers==1.2.0",
        "sentencepiece==0.1.95",
        "six==1.16.0",
        "threadpoolctl==2.1.0",
        "tokenizers==0.10.3",
        "torch==1.8.1",
        "torchvision==0.9.1",
        "tqdm==4.61.0",
        "transformers==4.6.1",
        "typing-extensions==3.10.0.0",
        "urllib3==1.26.5",
        "zipp==3.4.1",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
