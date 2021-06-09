import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="cassava_classifier",
    version="0.0.2",
    author="Prasanna Kumar, PS Vishnu",
    description="Cassava leaf disease classification using Deep neural network in Pytorch",
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    license="MIT",
    keywords=["image classification", "leaf disease classifier", "pytorch"],
    url="https://github.com/p-s-vishnu/cassava-leaf-disease-classification",
    packages=find_packages(exclude=["tests", "docs", "images"]),
    install_requires=[
        "albumentations==0.5.2",
        "apex==0.9.10dev",
        "opencv-python==4.5.1.48",
        "pandas==1.2.3",
        "scikit_learn==0.24.1",
        "timm==0.4.5",
        "torch==1.8.1",
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
