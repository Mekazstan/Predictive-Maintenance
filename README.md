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


# ABOUT THE CODE
This code does several things related to predictive maintenance using machine learning. Let me break it down:

- Data Loading and Exploration:
The code starts by loading a dataset named "predictive_maintenance.csv" into a data structure called a DataFrame using the pandas library.
It then displays the first 5 rows of this dataset to give an initial overview of what the data looks like.

- Exploratory Data Analysis (EDA):
The code performs Exploratory Data Analysis to ensure the quality of the data before training the model.
It creates a heatmap showing the correlation between different numeric variables, helping identify relationships in the data.

- Visualizing Feature Distributions:
Next, it visualizes the distribution of specific numeric features in the dataset using histograms and boxplots, making it easier to understand how these features vary.

- Machine Failure Analysis:
It creates a bar chart to visualize the count of machine failures, providing insight into the frequency of failures.

- Descriptive Statistics:
The code displays descriptive statistics of the dataset, giving a summary of key metrics like mean, standard deviation, and quartiles.

- Data Preprocessing:
It checks for any missing values in the dataset and drops columns that are not needed for prediction.
The code then engineers two new features, namely "Temperature Difference" and "Power," from existing features.

- Model Training:
The dataset is split into training and testing sets.
It uses the Synthetic Minority Over-Sampling Technique (SMOTE) to handle imbalanced data during training.
The code employs a Decision Tree Classifier within a pipeline for model training.

- Model Evaluation:
The trained model is evaluated using metrics like confusion matrix, accuracy, and classification report.
These metrics help assess how well the model performs in predicting machine failure.

- Saving the Model:
The trained model is saved for future use, enabling easy deployment without retraining.

- Predicting Tool Wear for New Data:
A new dataset with specific parameter values is created to predict tool wear using the trained model.
The code loads the saved model and makes predictions for the new data.

- Printing Results:
The confusion matrix for the new data is displayed, along with additional information based on the prediction.

In summary, this code is a comprehensive pipeline for building, training, evaluating, and using a machine learning model for predictive maintenance. It involves data exploration, visualization, model training, and result interpretation.
