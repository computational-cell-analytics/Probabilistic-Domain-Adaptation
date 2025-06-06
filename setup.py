#!/usr/bin/env python

import runpy
from distutils.core import setup


__version__ = runpy.run_path("prob_utils/__version__.py")["__version__"]


setup(
    name="prob_utils",
    version=__version__,
    description="Probabilistic Domain Adaptation for Biomedical Image Segmentation",
    author="Anwai Archit, Constantin Pape",
    author_email="anwai.archit@uni-goettingen.de, constantin.pape@informatik.uni-goettingen.de",
    url="https://user.informatik.uni-goettingen.de/~pape41/",
    packages=["prob_utils"],
)
