# Predictive Maintenance Model
This work contains code for a predictive maintenance model designed to predict machinery failure. The model uses an ensemble of feature-engineered variables and employs an Artificial Neural Network (ANN) implemented as a feed-forward neural network classifier.

## Contents
Exploratory Data Analysis (EDA)
Data Preprocessing
Training & Building the Model
Measuring Model Accuracy
Saving the Trained Model
Using the Model for Prediction
Results and Evaluation
Exploratory Data Analysis (EDA)
The code begins with exploratory data analysis to ensure data quality before training the model. It includes a correlation heatmap, feature distribution visualizations (histograms and boxplots), and a bar chart analyzing machine failures.

## Data Preprocessing
Data preprocessing steps involve checking for null values, dropping irrelevant columns, and feature engineering to create new variables such as "Temperature Difference" and "Power."

## Training & Building the Model
The model is trained using a Decision Tree Classifier within a pipeline. To address the class imbalance problem, the Synthetic Minority Over-Sampling Technique (SMOTE) is applied during the training phase.

## Measuring Model Accuracy
The accuracy of the model is measured using a confusion matrix, accuracy score, and a detailed classification report. The confusion matrix provides insights into true positives, true negatives, false positives, and false negatives.

## Saving the Trained Model
The trained model is saved using joblib for future use.

## Using the Model for Prediction
A new data point is created, and the trained model is loaded to predict tool wear for the new data.

## Results and Evaluation
The results are printed, including the confusion matrix for the new data and additional information based on the prediction.

## Requirements
Make sure to install the required libraries using the following:
- pip install matplotlib plotly seaborn pandas scikit-learn imbalanced-learn joblib

