import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="ML Exploration", layout="wide")
st.title("Welcome to my ML App")
st.markdown("Upload a dataset and tune hyperparameters to see how they impact the model's performance.")
# Start fresh df for an upload option
df = None
features = []
target = None
# 1 Sidebar create Data Upload options and sample data sets
st.sidebar.header("1. Data Input")
# create drop down options
data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Use Sample Dataset"])
if data_source == "Use Sample Dataset":
    sample_choice = st.sidebar.selectbox("Choose a Sample", ["Titanic (Binary)", "Iris (Multi-class)"]) 
    if sample_choice == "Titanic (Binary)":
        df = sns.load_dataset('titanic')
        # preprocessing the df
        df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        target_default = 'survived'
        #preprocessing second option data set
    elif sample_choice == "Iris (Multi-class)":
        df = sns.load_dataset('iris')
        target_default = 'species'
#
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_default = df.columns[-1] # Default to last column
    else:
        df = None
# Proceed after data set is uploaded
if 'df' in locals() and df is not None:
    st.write(f"### Dataset Preview: {sample_choice if data_source == 'Use Sample Dataset' else 'Uploaded File'}")
    st.dataframe(df.head())
    
# Dynamic selection of features and target
    all_cols = df.columns.tolist()
    target = st.sidebar.selectbox("Select Target Variable", all_cols, index=all_cols.index(target_default))
    features = st.sidebar.multiselect("Select Features", [c for c in all_cols if c != target], default=[c for c in all_cols if c != target][:3])

# Import for preprocessing and training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Create Model Selection 
st.sidebar.header("2. Model Settings")

# User picks the algorithm - (Classifiers)
algorithm = st.sidebar.selectbox(
    "Choose Algorithm", 
    ["Logistic Regression", "Decision Tree", "KNN"]
)

# Initialize model variable
model = None
# Add first model Decision Tree
if algorithm == "Decision Tree":
    # Using max depth as slider (adjustable)
    depth = st.sidebar.slider("Max Depth", 1, 10, 5)
    model = DecisionTreeClassifier(max_depth=depth)
# Add second model choice KNN
elif algorithm == "KNN":
    # Using 'k' as slider (adjustable)
    k_val = st.sidebar.slider("Number of Neighbors (K)", 1, 19, 5, 2)
    model = KNeighborsClassifier(n_neighbors=k_val)
