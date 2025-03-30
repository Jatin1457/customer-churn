import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

#Step 1: Load Data

data = pd.read_csv("https://raw.githubusercontent.com/IBM/churn/master/data/customer-churn.csv")

#Step 2: Data Preprocessing

data.drop(["customerID"], axis=1, inplace=True)  # Remove irrelevant column label_encoder = LabelEncoder() data['Churn'] = label_encoder.fit_transform(data['Churn'])

#Convert categorical columns

data = pd.get_dummies(data, drop_first=True)

#Step 3: Train-Test Split

X = data.drop("Churn", axis=1) y = data["Churn"] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Model Training

scaler = StandardScaler() X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42) model.fit(X_train, y_train)

#Step 5: Streamlit App

def main(): st.set_page_config(page_title="Customer Churn Prediction App", layout="wide") st.title("📊 Customer Churn Prediction Dashboard") st.write("Enter customer details to predict churn and explore data insights.")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=500.0)

if st.sidebar.button("Predict Churn", use_container_width=True):
    input_data = np.array([[tenure, monthly_charges, total_charges]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    result = "⚠ Churn" if prediction[0] == 1 else "✅ Not Churn"
    st.sidebar.subheader(f"Prediction: {result}")

# Data Visualization
st.subheader("📈 Data Insights")
col1, col2 = st.columns(2)

with col1:
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data['Churn'], ax=ax, palette="coolwarm")
    st.pyplot(fig)

with col2:
    st.write("### Monthly Charges vs. Churn")
    fig, ax = plt.subplots()
    sns.histplot(data=data, x="MonthlyCharges", hue="Churn", element="step", stat="density", common_norm=False, palette="coolwarm")
    st.pyplot(fig)

st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
st.pyplot(fig)

if name == "main": main()
