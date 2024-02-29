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
# Calculate the correlation matrix
corr_matrix = dataset.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create the correlation heatmap using seaborn
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)

# Add a title
plt.title('Correlation Heatmap')

# Show the plot
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
