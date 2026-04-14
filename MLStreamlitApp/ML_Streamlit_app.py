# Imports needed to build ML streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Added plot_tree for Step 6!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler # Recommended for Logistic and KNN
# Create page set ups
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["ML Exploration 🤖", "ML Guide 📚"])
# -------------------------------------------------------------------
# PAGE: ML GUIDE
# -------------------------------------------------------------------
if page == "ML Guide 📚":
    st.title("📚 Machine Learning Guide")
    st.markdown(" ⭐️More Information on how to interpret the models and metrics in this app. ⭐️")
    st.markdown(" ⭐️ Disclaimer: This App only experiments with Classification Algorithms!")
    st.header("🔬 Evaluation Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Accuracy")
        st.write("Overall amount of correct predictions. Great for balanced data, but can be misleading if classes are skewed!")
        
        st.subheader("✅ Precision")
        st.write("This tells us of all positive predictions, how many were actually correct. Guards against **False Positives** (False Alarms).")
        
    with col2:
        st.subheader("✅ Recall")
        st.write("This tells us of all actual positives, how many did the model caught.  Guards against **False Negatives** (Missed cases).")
        
        st.subheader("✅ F1-Score")
        st.write("The 'harmonic mean' of precision and recall. It tells us when one of the two is significantly lower than the other.")

    st.divider()
    
    st.header("🤖 Understanding the Algorithms")
    
    with st.expander("📈 Logistic Regression"):
        st.write("Despite the name, it's for **classification**. It calculates the probability (0 to 1) that a data point belongs to a specific category.")
        
    with st.expander("🌲 Decision Tree"):
        st.write("A flow-chart-like structure that makes decisions based on feature values. It's highly visual and easy to explain to non-technical users.")
        
    with st.expander("👥 K-Nearest Neighbors (KNN)"):
        st.write("The 'tell me who your neighbors are' approach. It classifies a point based on the labels of the points closest to it.")

# -------------------------------------------------------------------
# PAGE: ML EXPLORATION 
# -------------------------------------------------------------------
else:
    st.title("🤖 Welcome to my ML App 🤖")
    st.markdown("Upload a dataset and tune hyperparameters to see how they impact the model's performance.")
    st.markdown(" ⭐️ Disclaimer: This App only experiments with Classification Algorithms!")
    # Start an upload option
    df = None
    features = []
    target = None
    # -------------------------------------------------------------------
    # Step 1: Sidebar create Data Upload options and sample data sets
    # -------------------------------------------------------------------
    st.sidebar.header("1. Data Input")
    # create drop down options
    data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Use Sample Dataset"])
    if data_source == "Use Sample Dataset":
        sample_choice = st.sidebar.selectbox("Choose a Sample", ["Titanic (Binary)", "Iris (Multi-class)"]) 
        if sample_choice == "Titanic (Binary)":
            df = sns.load_dataset('titanic')
            # drop down for Titanic data
            df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
            df['sex'] = df['sex'].map({'male': 0, 'female': 1})
            target_default = 'survived'
            #drop down for Iris data
        elif sample_choice == "Iris (Multi-class)":
            df = sns.load_dataset('iris')
            target_default = 'species'
    #drop down for user uploaded data
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            target_default = df.columns[-1] # Default to last column
        else:
            df = None
# If user has selected data proceed to next step 
    if df is not None:
        st.write(f"### Dataset Preview: {sample_choice if data_source == 'Use Sample Dataset' else 'Uploaded File'}")
        st.dataframe(df.head())
        
        # Dynamic selection of features and target
        all_cols = df.columns.tolist()
        target = st.sidebar.selectbox("Select Target Variable", all_cols, index=all_cols.index(target_default))
        features = st.sidebar.multiselect("Select Features", [c for c in all_cols if c != target], default=[c for c in all_cols if c != target][:3])
    # -------------------------------------------------------------------
        # Step 2: Preprocess the data 
    # -------------------------------------------------------------------
        st.header("Step 2: Preprocess the Data")

        if not features:
            st.warning("Please select at least one feature in the sidebar to continue.")
        else:
            # 1. Clean Missing Values
            initial_count = len(df)
            df_clean = df.dropna(subset=features + [target]) # only drop rows missing in selected columns
            st.write(f"✔ **Cleaned Missing Values:** Removed {initial_count - len(df_clean)} rows.")

            # 2. Define X and y
            X = df_clean[features] 
            y = df_clean[target]

            # 3. Encoding Features
            X = pd.get_dummies(X, drop_first=True)
            st.write("✔ **Categorical Encoding:** Applied 'Drop First' to features to avoid the dummy trap.")

            # 4. Target Encoding
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = y.astype('category').cat.codes
                st.write(f"✔ **Target Encoding:** Converted '{target}' text labels into numeric codes.")

            # 5. Final Previews
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Features (X)")
                st.dataframe(X.head())
            with col2:
                st.write("### Target (y)")
                st.write(y.head())

    # -------------------------------------------------------------------
    # 3. Create Model Selection & Descriptions for each 
    # -------------------------------------------------------------------

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
        depth = st.sidebar.slider("Select Depth", 1, 8, 5)
        st.sidebar.caption(f"A depth of {depth} means the tree can have up to {depth} levels of questions.")
    # added additional Hyperparameter ( Avoid overfitting and complex trees): Min Samples Split
        st.sidebar.markdown("**Min Samples Split:**")
        min_split = st.sidebar.slider("Select Min Samples", 2, 20, 20)
        st.sidebar.caption(f"A node must have at least {min_split} samples to be split into smaller branches.")
        # Updated model with both parameters
        model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split)
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
            #'l2' is the standard, 'None' removes the regularization
        pen = st.sidebar.selectbox("Choose Penalty", ["l2", "None"])
        if pen == "l2":
            st.sidebar.caption("L2 helps prevent overfitting by 'penalizing' large coefficients.")
        else:
            st.sidebar.caption("None allows the model to fit the data exactly as it is.")
            
        model = LogisticRegression(penalty=pen if pen != "None" else None, solver='lbfgs', max_iter=1000)

    # Display selection status
    if model:
        st.sidebar.success(f"Selected: {algorithm}")
        st.write(f"### Current Model: {algorithm}")
        st.info(f"Adjust the parameters in the sidebar to see how **{algorithm}** changes!")
        
    # -------------------------------------------------------------------
    # Step 4: Split, and Train, 
    # -------------------------------------------------------------------
    st.header("Step 3: Train and Evaluate")
    # Ensure data was preprocessed in Step 2 before running
    if 'X' in locals() and 'y' in locals():
        
        # Scaling  used for KNN and Logistic
        use_scaling = False
        if "KNN" in algorithm or "Logistic Regression" in algorithm:
            st.subheader("Model Optimization")
            use_scaling = st.checkbox(
                "Scale Data (StandardScaler)", 
                value=True, 
                help="Normalizes features to have a mean of 0 and a standard deviation of 1."
            )

        #  The "Run" Button so the user can see changes after selecting data and parameters
        if st.button("🚀 Run Model 🚀"):
            st.divider()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Apply Scaling if the user selects this we apply it to train the model
            if use_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                st.info("✔ **Data Scaled:** Features were transformed using StandardScaler.")

            # 3. Training and Predicting
            with st.spinner(f"Training {algorithm}..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # 4. Display Results
            st.metric(label="Overall Accuracy", value=f"{accuracy_score(y_test, y_pred):.2%}")

