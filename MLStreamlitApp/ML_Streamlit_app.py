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
from sklearn.metrics import roc_curve, roc_auc_score
# Create page set ups there will be two 
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["ML Exploration 🤖", "ML Guide 📚"])
# -------------------------------------------------------------------
# PAGE: ML GUIDE - Gives descirptions and info for users
# -------------------------------------------------------------------
if page == "ML Guide 📚":
    # Create title and disclaimer since I am only including classification models!
    # using a disclaimer to preface to users that conitnous targets will NOT work
    st.title("📚 Machine Learning Guide")
    st.markdown(" ⭐️More Information on how to interpret the models and metrics in this app. ⭐️")
    st.markdown(" ⭐️ Disclaimer: This app experiments with Classification Algorithms only! This means you cannot choose a continuous variable as your target (e.g., price or weight).")
    # Include some info on ML such as supervised and unsupervised in a mark down 
    st.markdown("""
                ### 📘 Machine Learning Fundamentals
                #### ⭐️ Supervised vs. Unsupervised Learning:
                * **Supervised Learning:** The model is trained on **labeled data** (input-output pairs). 
                * **Unsupervised Learning:** The model looks for **hidden patterns** or structures in data that has no labels. 
                --- 
                #### ⭐️ Classifiers vs. Continuous (Regression) Models
                Supervised learning is generally split into two types:
                * **Classification:** Predicts a **class** or category. Use this if your goal is to answer "Yes/No" or "Is this a Cat or Dog?"
                * **Continuous (Regression):** Predicts a **specific numerical value**. Use this if your goal is to predict something like a length or price.
                """)
    st.header("🔬 Evaluation Metrics")
    # Create descriptions for metrics 
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
    # Create info sections for each algorithm for User just for the m to see before or after they explore
    st.header("🤖 Understanding the Algorithms")
    # Explain Logistic regression
    with st.expander("📈 Logistic Regression"):
        st.write("This is for **classification**. It calculates the probability (0 to 1) that a data point belongs to a specific category.")
        # Explain Decision Tree
    with st.expander("🌲 Decision Tree"):
        st.write("A flow-chart-like structure that makes decisions based on feature values. It's highly visual and easy to follow as it flows downward")
        # Explain KNN
    with st.expander("👥 K-Nearest Neighbors (KNN)"):
        st.write("This alogorithm classifies a point based on the labels of the points closest to it. KNN is sensitive to the scale of the features, scaling can often lead to a substantial improvement in performance.")

# -------------------------------------------------------------------
# PAGE: ML EXPLORATION 
# -------------------------------------------------------------------
else:
    # Make page name and explanation of what User can modify and what will occur
    st.title("🤖 Welcome to my ML App 🤖")
    st.markdown("Upload a dataset and tune hyperparameters to see how they impact the model's performance.")
    st.markdown(" ⭐️ Disclaimer: This app experiments with Classification Algorithms only! This means you cannot choose a continuous variable as your target (e.g., price or weight).")
    # Start with a empty df so user can select df 
    df = None
    features = []
    target = None
    # -------------------------------------------------------------------
    # Step 1: Sidebar create Data Upload options and sample data sets
    # -------------------------------------------------------------------
    st.sidebar.header("1. Data Input")
    # create drop down options for orgin of Data
    data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Use Sample Dataset"])
    if data_source == "Use Sample Dataset": # Providing two data sets for users to experiment
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
        # Option for user to choose their own data and upload
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            target_default = df.columns[-1] # Default to last column but changable by user
        else: # Means data set has not selected and will not proceed to next steps yet
            df = None
# If user has selected data proceed to next step 
    if df is not None:
        st.write(f"### Dataset Preview: {sample_choice if data_source == 'Use Sample Dataset' else 'Uploaded File'}")
        st.dataframe(df.head())
        
        # Selection of features and target for side bar and user choice
        all_cols = df.columns.tolist()
        target = st.sidebar.selectbox("Select Target Variable", all_cols, index=all_cols.index(target_default))
        features = st.sidebar.multiselect("Select Features", [c for c in all_cols if c != target], default=[c for c in all_cols if c != target][:3])
    # -------------------------------------------------------------------
        # Step 2: Preprocess the data 
    # -------------------------------------------------------------------
        st.header("Step 2: Preprocess the Data")
        # add warning to user if features not yet selected
        if not features:
            st.warning("Please select at least one feature in the sidebar to continue.")
            # Preprocess once data is selected
        else:
            # 1. Clean Missing Values
            initial_count = len(df)
            df_clean = df.dropna(subset=features + [target]) # only drop rows missing in selected columns
            # This will be helpful especially when csv is uploaded by user
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

            # 5. Final Previews changes based on what the user selects
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
        # Explanation of logisiticegression
        st.sidebar.subheader("About Logistic Regression 📈")
        st.sidebar.write("Used for binary classification (Yes/No). It calculates the probability of an event occurring.")
        
        # Hyperparameter 
        st.sidebar.markdown("**Penalty (Regularization):**")
            #'l2' is the standard, 'None' removes the regularization
        pen = st.sidebar.selectbox("Choose Penalty", ["l2", "None"])
        if pen == "l2":
            st.sidebar.caption("L2 helps prevent overfitting by 'penalizing' large coefficients.")
        else: # explanation
            st.sidebar.caption("None allows the model to fit the data exactly as it is.")
            
        model = LogisticRegression(penalty=pen if pen != "None" else None, solver='lbfgs', max_iter=1000)

    # Display selection status of user
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
            # EVALUATIONS
            st.metric(label="Overall Accuracy", value=f"{accuracy_score(y_test, y_pred):.2%}")
            st.info("Accuracy is simply Correct Predictions/Total Predictions")
            # Confusion Matrix will appear for all models
            st.write("### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            # Add labels for presentation
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            st.pyplot(fig_cm)


            # Classification Report
            st.divider()
            st.write("### Classification Report")
            st.code(classification_report(y_test, y_pred))


            # Visualizations for each model 
            if "Decision Tree" in algorithm:
                   st.write("### Decision Tree Logic Visualization")
                   fig_tree, ax_tree = plt.subplots(figsize=(15, 10))
                   plot_tree(model, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=3)
                   st.pyplot(fig_tree)
                   # Provide an explanation to users on how to interpret the visualization
                   st.info("⭐️ **How to read this graph:**")
                   st.markdown("""
                    **Condition (Top Line)** = The "If-Then" rule. If True, go left; if False, go right.
                               
                    **Gini** = The Impurity score. Lower means the node is "purer" (more of one class).
                               
                    **Samples** = The number of training examples that reached this specific node.
            
                    **Value** = The Class Distribution. Shows the count for each category.
            
                    **Class** = The Majority Prediction. The category this box would predict for you.
                    """)
            # Next visualization is for KNN       
            elif "KNN" in algorithm:
                   st.write("### Accuracy vs. Number of Neighbors (k)")
                   k_values = range(1, 21, 2)
                   accuracies = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test) for k in k_values]
                   fig_k, ax_k = plt.subplots()
                   plt.plot(k_values, accuracies, marker='o')
                   st.pyplot(fig_k)
                   # Polish title to include if scaled or unscaled
                   plt.title(f'Effect of k on Accuracy ({"Scaled" if use_scaling else "Unscaled"})')
                   st.write("### 👥 How to Interpret the 'K' Chart")
                   st.markdown("""
                    This graph shows how the model's **Accuracy** changes as you adjust the number of neighbors (**k**). 
                
                    * The Peak: The highest point on the line represents where the model is most accurate. 
                    * Overfitting (Low K): When **k** is very small (like 1), the model is too sensitive to "noise" or outliers in the data.
                    * Underfitting (High K): If **k** is too large, it ignores the local patterns in the data.
                
                 **Try this:** Look for the **'Elbow'** or the highest stable peak, then go back to the sidebar and set your **Select K** slider to that number for the best results.
                    """)
            elif "Logistic Regression" in algorithm:
            # adding ROC curve
            # Check if it's binary classification (like Titanic) ( works best for ROC)
                if len(np.unique(y_test)) == 2:
                # Aesthetic to show ROC
                    st.divider()
                    st.write("### ROC Curve")
                
                        # Get the predicted probabilities for the positive class
                    y_probs = model.predict_proba(X_test)[:, 1]

                        # Calculate FPR, TPR, and thresholds
                    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

                        # Compute the AUC score
                    roc_auc = roc_auc_score(y_test, y_probs)
                    st.write(f"**ROC AUC Score:** {roc_auc:.2f}")
                    # Plot the ROC curve
                    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') 
                    # Polish the graph 
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                
                    st.pyplot(fig_roc)
                    # ROC explanation 
                    st.write("### 📖 How to understand the ROC Curve")
                    st.markdown("""
                    The **ROC (Receiver Operating Characteristic)** curve helps evaluate the model's performance across different classification thresholds:

                    * **ROC Curve** = Plots the True Positive Rate against the False Positive Rate (how many 'false alarms' we had).
                    * **AUC (Area Under the Curve)** = Summarizes the overall ability of the model to tell the two classes apart. 
                
                    **A score of 1.0 is perfect, while 0.5 means the model is just guessing!**
                    """)
            else:
                st.info("Waiting for data preprocessing. Please select features and target to begin.")
            # Encourage users to explore more in final text on page 
            st.subheader("🔎 Keep Exploring!🔎")
            st.markdown("""
                        **Continue to Explore! See how the model shifts, under different parameters**
                        * **Check the Guide:** Head over to the **ML Guide 📚** page in the sidebar to learn more about metrics used and ML!
                        * **Adjust Hyperparameters:** in the sidebar to see how the charts change.
                        """)