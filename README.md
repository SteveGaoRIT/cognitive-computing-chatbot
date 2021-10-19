#T5 Question Answering Chat-bot
This program implements data pre-process and preparation, fine-tuning 
process of T5 model, inference process of T5 model. We used the pre-trained 
T5 model published by hugging-face because of the shortage of GPU resources.

You can run scripts in following order to fine-tune a T5:

* python inference_sample.py
* python fine_tune.py

## requirements
numpy, pip install numpy
pytorch, see https://pytorch.org/get-started/locally/
transformers, https://pytorch.org/get-started/locally/
tensorboardX, pip install tensorboardX

## inference_sample.py
Test the inference function of pre-trained T5 
model and save it into pre-train model directory.

You need to run this script to get the pre-trained T5 model.

## fine_tune.py
Fine-tune T5 model using ambignq dataset. Here I set the epochs as
20, learning rate variances from 3e-4 to 1e-5 using a Cosine function. 
Accumulation gradient is used here to overcome the short of GPU resources.

## dataset.py
I built my dataset here.

## data_processing.py
This script is used to test the data processing approach, you do not 
need to run it when training T5.
