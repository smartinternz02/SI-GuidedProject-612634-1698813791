# Real/Fake Logo Detection using Deep Learning

## Overview

This project implements a Real/Fake Logo detection system using deep learning. The goal is to train a Convolutional Neural Network (CNN) to distinguish between real and fake logos. The dataset consists of images of both genuine and fake logos, and the model is trained to classify these images into their respective categories.

## Requirements

Before running the code, ensure that you have the required Python packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```
## Usage 
# Clone the repository:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

# Run the provided code:
```bash
python detection.py
```
### Directory Structure

- `dataset/`: Contains Trainn and Split dataset.
- `script/`: Contains the entire model and python script.
- `notebooks/` : Contains data visualization and prediction
- `output/`: Contains the model, However the github upload limit exists , it only contains in the admins directory not on github.

### Code Explanation 
```bash
# Import necessary modules
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.image import img_to_array, load_img
```

##  Loading and Preprocessing Images

The script first loads and preprocesses the images from the specified paths. Images are resized to a common size and converted to NumPy arrays. The Inception V3 preprocessing function is applied to the image arrays.

## Concatenating Data

Fake and genuine data from the training and test sets are concatenated to create the final training and test sets. Labels are also assigned (0 for fake, 1 for genuine).

##  Splitting the Dataset

The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

##  Defining the CNN Model

A simple CNN model is defined using the Keras Sequential API. It consists of convolutional layers, max-pooling layers, and dense layers.

## Compiling the Model

The model is compiled with the Adam optimizer and binary cross-entropy loss, as it is a binary classification problem. The accuracy metric is used for evaluation.

## Training the Model

The model is trained on the training set with 10 epochs and a batch size of 32. Validation data is used to monitor the model's performance during training.

## Evaluating the Model

The trained model is evaluated on the test set, and accuracy along with the confusion matrix is printed.

## Saving the Model

The trained model is saved as 'logo_detection_model.h5'.
