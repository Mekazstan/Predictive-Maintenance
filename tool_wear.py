from matplotlib import pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
# corr_matrix = df.corr()

# Select only numeric columns
numeric_columns = df.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numeric_columns.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create the correlation heatmap using seaborn
heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)

# Add a title
plt.title('Correlation Heatmap')

# Adjust the layout to prevent text cutting off
plt.tight_layout()

# Save the heatmap image with a higher DPI (dots per inch)
heatmap.get_figure().savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# Feature Distributions
# Visualizing the distribution of the numerical features in the dataset using histograms.

# List of variables
variables = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Create subplots with two columns
fig = make_subplots(rows=len(variables), cols=2, subplot_titles=[f'{var} Histogram and Boxplot' for var in variables])

for i, var in enumerate(variables, start=1):
    # Histogram
    histogram = px.histogram(df, x=var, nbins=20, title=f'{var} Histogram')
    histogram.update_layout(showlegend=False)
    
    # Boxplot
    boxplot = px.box(df, x=var, title=f'{var} Boxplot')
    boxplot.update_layout(showlegend=False)

    # Add the histograms and boxplots to the subplots
    fig.add_trace(histogram['data'][0], row=i, col=1)
    fig.add_trace(boxplot['data'][0], row=i, col=2)

    # Save each subplot as an individual image
    # histogram.write_image(f'{var}_histogram.png')
    # boxplot.write_image(f'{var}_boxplot.png')

# Update layout
fig.update_layout(height=400*len(variables), width=800, showlegend=False)

# Show the plot
fig.show()

# Machine Failure Analysis
# To analyze machine failures, we can create a bar chart to visualize the count of failures.
failure_counts = df["Machine failure"].value_counts()
fig = px.bar(failure_counts, x=failure_counts.index, y=failure_counts.values, labels={"x": "Machine failure", "y": "Count"})
fig.show()

# Distinguishing between cases of failure & No-Failure
df['Target'].value_counts()

# Descriptive Statistics
# Use the describe method to get summary statistics of the dataset
styled_data = df.describe().style\
.background_gradient(cmap='coolwarm')\
.set_properties(**{'text-align':'center','border':'1px solid black'})

# display styled data
display(styled_data)

# -----> Data Preprocessing

# Checking for Null values in the dataset
df.isnull().sum()

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
