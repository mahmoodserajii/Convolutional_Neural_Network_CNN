
# **MNIST Digit Classification with CNN**

## **Overview**
This project contains a complete, end-to-end machine learning pipeline using a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The pipeline demonstrates best practices for data handling, model training, and performance evaluation in a self-contained Python script.

## **Datasets**
* **MNIST** – A dataset of 60,000 training images and 10,000 test images of handwritten digits.

## **Features**
* **Data Loading and Preprocessing:** Handles data loading, normalization, and reshaping.
* **Model Building:** Constructs a sequential CNN model with convolutional, pooling, and dense layers.
* **Model Training:** Trains the model using a validation set and an `EarlyStopping` callback to prevent overfitting.
* **Performance Evaluation:** Reports key metrics including accuracy, precision, recall, and F1-score.
* **Visualization:** Generates plots for training/validation accuracy and loss over epochs.
* **Confusion Matrix Analysis:** Provides a detailed breakdown of model errors.

## **File**
* `CNN.ipynb` – Contains the complete Python script for the entire pipeline.

## **Usage**
1. Ensure all prerequisite libraries are installed.
2. Run the `cnn_mnist_pipeline.py` script.
3. The pipeline will automatically download the dataset, build the model, train it, and display the final evaluation metrics and plots in the console.

