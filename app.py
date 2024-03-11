import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix


# Load the trained model
loaded_model = joblib.load('tool_wear_prediction_model.joblib')

# Streamlit App
st.title("Tool Wear Prediction App")

# Sidebar with user input
st.sidebar.header("Enter New Data:")
new_air_temp = st.sidebar.number_input("Air temperature [K]", min_value=0.0, max_value=500.0, step=0.1, value=298.8)
new_process_temp = st.sidebar.number_input("Process temperature [K]", min_value=0.0, max_value=500.0, step=0.1, value=308.7)
new_rotational_speed = st.sidebar.number_input("Rotational speed [rpm]", min_value=0, max_value=5000, step=1, value=1497)
new_torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=500.0, step=0.1, value=46.8)
new_tool_wear = st.sidebar.number_input("Tool wear [min]", min_value=0, max_value=1000, step=1, value=72)

# Feature Engineering
new_temp_diff = new_air_temp - new_process_temp
new_power = new_rotational_speed * new_torque

# Create a DataFrame for the new data
new_data = pd.DataFrame({
    'Air temperature [K]': [new_air_temp],
    'Process temperature [K]': [new_process_temp],
    'Rotational speed [rpm]': [new_rotational_speed],
    'Torque [Nm]': [new_torque],
    'Tool wear [min]': [new_tool_wear],
    'Temperature Difference': [new_temp_diff],
    'Power': [new_power]
})

# Make predictions for the new data
prediction = loaded_model.predict(new_data)

# Display prediction result
st.subheader("Prediction Result:")
if prediction == 1:
    st.error("Tool is more likely to fail!!")
else:
    st.success("No failure detected.")


st.subheader("Additional Information:")
st.write("Confusion Matrix for New Data:")
# Assuming '1' is the positive class and '0' is the negative class
labels = [0, 1]
conf_matrix = confusion_matrix([1], prediction, labels=labels)
st.write(conf_matrix)
