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


# 2. Create Model Selection & Descriptions for each 
st.sidebar.header("2. Model Settings")

# User picks the algorithm - (Classifiers)
algorithm = st.sidebar.selectbox(
    "Choose Algorithm", 
    ["Logistic Regression 📈", "Decision Tree 🌲", "KNN 👥"]
)
# Initialize model variable
model = None
# Add first model Decision Tree
if algorithm == "Decision Tree 🌲":
    st.sidebar.subheader("About Decision Tree 🌲")
    st.sidebar.write("Flows downward similar to a flow chart structure. Uses'If-Then' rules to split data into categories.")
    
    # Hyperparameter Explanation
    st.sidebar.markdown("**Max Depth:**")
    # Using max depth as slider (adjustable)
    depth = st.sidebar.slider("Select Depth", 1, 10, 5)
    st.sidebar.caption(f"A depth of {depth} means the tree can have up to {depth} levels of questions.")
    
    model = DecisionTreeClassifier(max_depth=depth)
# Add second model choice KNN
elif algorithm == "KNN 👥":
    st.sidebar.subheader("About KNN 👥")
    st.sidebar.write("Classifies data by looking at the 'K' closest labeled data points.")
    
    # Hyperparameter Explanation
    st.sidebar.markdown("**Number of Neighbors (K):**")
     # Using 'k' as slider (adjustable)
    k_val = st.sidebar.slider("Select K", 1, 19, 5, 2)
    st.sidebar.caption(f"The model will look at the {k_val} nearest neighbors to 'vote' on the class.")
    
    model = KNeighborsClassifier(n_neighbors=k_val)
# Add third model choice Logistic Regression
elif algorithm == "Logistic Regression 📈":
    st.sidebar.subheader("About Logistic Regression 📈")
    st.sidebar.write("Used for binary classification (Yes/No). It calculates the probability of an event occurring.")
    
    # Hyperparameter Explanation
    st.sidebar.markdown("**Penalty (Regularization):**")
        #'l2' is the standard, 'none' removes the regularization
    pen = st.sidebar.selectbox("Choose Penalty", ["l2", "none"])
    if pen == "l2":
        st.sidebar.caption("L2 helps prevent overfitting by 'penalizing' large coefficients.")
    else:
        st.sidebar.caption("None allows the model to fit the data exactly as it is.")
        
    model = LogisticRegression(penalty=pen, solver='lbfgs', max_iter=1000)

# Display selection status
if model:
    st.sidebar.success(f"Selected: {algorithm}")
    st.write(f"### Current Model: {algorithm}")
    st.info(f"Adjust the parameters in the sidebar to see how **{algorithm}** changes!")