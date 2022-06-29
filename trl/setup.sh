#!/bin/bash

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
cd /content/drive/MyDrive/RCT/transformers/examples/pytorch/summarization/
pip install -r requirements.txt

pip install bert_score==0.3.11
pip install meteor
pip install nltk==3.5