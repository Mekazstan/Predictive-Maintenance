from matplotlib import pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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

# Feature Engineering
df['Temperature Difference'] = df['Air temperature [K]'] - df['Process temperature [K]']
df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# Split data into features (X) and target variable (y)
X = df.drop(['Target', 'UDI', 'Product ID', 'Type', 'Failure Type'], axis=1)
y = df['Target']

# -----> Training & Building the model

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Decision Tree Classifier within a pipeline
model = make_pipeline(SMOTE(random_state=42), DecisionTreeClassifier(random_state=42))

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Artificial Neural Network": MLPClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    pipeline = make_pipeline(SMOTE(random_state=42), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {"Accuracy": accuracy, "Confusion Matrix": confusion_matrix(y_test, y_pred)}

# Print results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print("Confusion Matrix:")
    print(result["Confusion Matrix"])
    print("\n")

# Save the best model
best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.joblib')

# Predict tool wear using the best model
new_data = pd.DataFrame({
    'Air temperature [K]': [298.8],
    'Process temperature [K]': [308.7],
    'Rotational speed [rpm]': [1497],
    'Torque [Nm]': [46.8],
    'Tool wear [min]': [72],
    'Temperature Difference': [10.1],
    'Power': [1497 * 46.8]
})
loaded_model = joblib.load('best_model.joblib')
prediction = loaded_model.predict(new_data)

# Print prediction
if prediction[0] == 1:
    print("Predicted Tool Wear: The tool is more likely to fail.")
else:
    print("Predicted Tool Wear: No failure is predicted.")

# Get feature importances from the best model (if it's a tree-based model)
if isinstance(best_model, DecisionTreeClassifier) or isinstance(best_model, RandomForestClassifier):
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print feature importances
    print("\nFeature Importances:")
    for feature, importance in sorted_feature_importance:
        print(f"{feature}: {importance:.4f}")

    # Identify the most important feature
    most_important_feature = sorted_feature_importance[0][0]
    print(f"\nMost Important Feature Leading to Machine Failure: {most_important_feature}")
else:
    print("Feature importances are not available for the selected model.")
