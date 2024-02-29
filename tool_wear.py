from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.metrics import confusion_matrix

# Load the dataset
df = pd.read_csv("predictive_maintenance.csv")

# Viewing the dataset
df.head(5)

# -----> Performing Exploratory Data Analysis on the dataset to ensure data quality before training(EDA)

# Heatmap visualization of correlation between variables
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr('spearman'), annot=True,cmap='summer')
plt.title('Heatmap Correlation')
plt.show()

# Checking for Null values in the dataset
df.isnull().sum()

# Visualizing the distribution of the given dataset
fig, axes = plt.subplots(2, 5, figsize=[25,15])
j = 0
colors = ['yellow', 'green'] 

for i in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
    sns.histplot(data=df, x=i, kde=True, ax=axes[0,j], hue='Machine failure', palette=colors)
    j+=1
    print('{} skewness = {}'.format(i, round(df[i].skew(), 2)))


# Dropping columns that are not needed for prediction
df = df.drop(['UDI', 'Product ID', 'Type', 'Failure Type'], axis=1)

# Split data into features (X) and target variable (y)
X = df.drop('Target', axis=1)
y = df['Target']

# -----> Training & Building the model

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Training the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# -----> Measuring the accuracy of the model

# Evaluate the model and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(model, 'tool_wear_prediction_model.joblib')

# Using the newly created model to predict tool wear for new data
new_data = pd.DataFrame({
    'Air temperature [K]': [298.8],
    'Process temperature [K]': [308.7],
    'Rotational speed [rpm]': [1497],
    'Torque [Nm]': [46.8],
    'Tool wear [min]': [72]
})

# Load the trained model
loaded_model = joblib.load('tool_wear_prediction_model.joblib')

# Make predictions for the new data
prediction = loaded_model.predict(new_data)
# print("Predicted Tool Wear:", prediction)

# -----> Printing results

# Print confusion matrix for the new data
conf_matrix_new_data = confusion_matrix([1], prediction)
print("\nConfusion Matrix for New Data:")
print(conf_matrix_new_data)

# Print other information based on your prediction
if prediction == [1]:
    print("Tool is more likely to fail!!")
else:
    print("No failure..")


"""_summary_

    'Air temperature [K]': [298.8]
    This represents the air temperature in Kelvin (K) during the industrial process. 
    Kelvin is an absolute temperature scale, and in this case, the air temperature is approximately 298.8 Kelvin.
    
    'Process temperature [K]': [308.7]
    This represents the process temperature in Kelvin (K) during the industrial process. 
    The process temperature is approximately 308.7 Kelvin.
    
    'Rotational speed [rpm]': [1497]
    This represents the rotational speed of a component in the machinery, measured in revolutions per minute (rpm). 
    In this case, the rotational speed is approximately 1497 revolutions per minute.
    
    'Torque [Nm]': [46.8]
    This represents the torque applied to the machinery, measured in Newton-meters (Nm). 
    The torque is approximately 46.8 Newton-meters. Torque (Force that tends to cause rotation)
    
    'Tool wear [min]': [72]
    This represents the cumulative time the tool has been in operation or in use, measured in minutes. 
    The tool has been in use for approximately 72 minutes since the start of its operation.
    
    A confusion matrix provides a more detailed and comprehensive evaluation of the performance of your classification model.
    The confusion matrix provides information on true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). 
    These values can be helpful for understanding where your model is making errors and can guide further improvements or adjustments.

    Accuracy: How often it is correct. (Opposite == Misclassification)
    Recall: When it's actually yes,how often does it predict "YES" (TRUE POSITIVE)
    (FALSE POSITIVE)
    Specificity: (TRUE NEGATIVE)
    Precision: When predicted as yes how often is it correct.
"""