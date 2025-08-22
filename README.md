End-to-End CNN Pipeline for MNIST Digit Classification
Project Description

This project implements a complete, end-to-end machine learning pipeline using a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The pipeline covers all essential steps, from data loading and preprocessing to model training, evaluation, and visualization of results. The model demonstrates a high level of accuracy and is a great example of a CNN's power in image classification tasks.
Prerequisites

To run this pipeline, you will need the following Python libraries installed. You can install them using pip:

    numpy

    pandas

    scikit-learn

    tensorflow

    matplotlib

pip install numpy pandas scikit-learn tensorflow matplotlib

Pipeline Overview

The project is structured into the following logical steps:

    Data Loading: The MNIST dataset is loaded directly from Keras, providing 60,000 training images and 10,000 test images.

    Preprocessing: Image pixel values are normalized to a range of [0, 1] and reshaped to the format required by the CNN model.

    Data Splitting: The training data is split into a smaller training set and a validation set for hyperparameter tuning and preventing overfitting.

    Model Building: A sequential CNN model is constructed using Keras, incorporating convolutional, pooling, and dense layers with Dropout for regularization.

    Training: The model is trained on the training set while monitoring its performance on the validation set using an Early Stopping callback.

    Evaluation: The final, best-performing model is tested on the unseen test set, and key metrics like accuracy, precision, and recall are reported.

    Visualization: Plots of training and validation loss/accuracy are generated to provide insights into the model's learning process.

Results and Analysis

The CNN model achieved outstanding performance on the MNIST dataset.

    Test Accuracy: 0.99

    Weighted F1-Score: 0.99

The high accuracy and F1-score indicate that the model generalizes exceptionally well to new, unseen handwritten digits. The model's performance on a per-class basis is also very strong, with a perfect F1-score of 1.00 for digits 1 and 2.

The confusion matrix highlights the few instances where the model made errors, often confusing visually similar digits:

    Digit 4 was occasionally misclassified as 9.

    Digit 5 was occasionally misclassified as 3 or 8.

    Digit 6 was occasionally misclassified as 5.

These minor misclassifications are expected and do not detract from the overall excellent performance of the model.
