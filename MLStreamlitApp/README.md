# 🤖Project Overview🤖: 
The goal of this project is to provide an interactive, web-based platform for supervised machine learning using **Streamlit**. 

- **The Application**: This app allows users to transition from raw data to model evaluation in one interface. Users can upload their own datasets, select from various supervised learning models, and fine-tune hyperparameters in real-time. This tool invites exploration of model behavior and makes performance metrics accessible to users.

### The primary objective of this project is divided into two key goals:

## - 🥇 First Goal: Interactive Model Training & Hyperparameter Tuning
The project aims to encourage model training, exploration, and learning. 
- **User Input**: Users can upload CSV files or choose from provided sample datasets.
- **Tuning**: Users utilize Streamlit widgets (sliders, dropdowns) to adjust parameters such as the 'K' in KNN or the depth of a Decision Tree.

## - 🥈 Second Goal: Visualizations
The project aims to communicate results clearly with evaluation metrics.
- **Evaluation**: The app automatically calculates metrics such as Accuracy, Precision, and Recall.
- **Visuals**: It generates charts to provide a visual story of how well the model is performing on the selected data.

# Instructions📋: 
This app is deployed and can be found here.

To run this project locally, follow these steps using Python:

1. Clone the repository and ensure your data files are in the root directory.
2. Install the required libraries:
```
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```
3. streamlit run the py
   For reference, this app uses the following libraries and imports
   
```
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
```
# App Features: 
The application is split into two distinct sections to guide the user in exploration:

1. **ML Exploration**
This is where the user can adjust hyperparameters and select a model.

Step 1: Data Input & Feature Selection.
    - Users can select the target variable and features to use
    
<img width="206" height="152" alt="Screenshot 2026-04-14 at 9 50 53 PM" src="https://github.com/user-attachments/assets/d2b24eb8-0c17-4a2e-82b7-66caf5533d35" />

Step 2: Automated Preprocessing (Handling NaNs and Encoding).
   
Step 3: Model Selection and Hyperparameter Adjustment

* **Model Variety:** Users can select from three models: KNN, Decision Tree, and Logistic Regression.

<img width="208" height="179" alt="Screenshot 2026-04-14 at 9 55 36 PM" src="https://github.com/user-attachments/assets/07dce714-9664-4a22-8a14-af9f3586b04a" />

* **Dynamic Tuning:** Hyperparameters can be adjusted, which will depend on the model selected:
    1. **KNN 👥** — Users can change the **Number of Neighbors (K)** to see how local data clusters influence classification.
    2. **Decision Tree 🌲** — Users can adjust **Max Depth** and **Min Samples Split** to balance model complexity and prevent the tree from overfitting.
    3. **Logistic Regression 📈** — Users can choose to add **Regularization (Penalty)** to see how penalizing large coefficients changes the model's predictions.

Step 4: Execution and Visual Analysis
This app features:
* **Confusion Matrix:** A heatmapped matrix that visually breaks down correct vs. incorrect predictions.
* **ROC Curve & AUC Score:** For binary classification tasks, the app plots the Receiver Operating Characteristic curve.
* **Classification Report:** A summary table providing Precision, Recall, and F1-Score.

3. **ML Guide 📚**
This is a page that explains key metrics used and provides definitions.
    Includes definitions for 
    - Metrics: Accuracy, Precision, Recall, and F1-Score.
    - Algorithms: breakdowns of how Logistic Regression, Decision Trees, and KNN function.
    -  Summary of steps one can follow when using algorithms on their own 
# Visual Examples: 
## 🌲 Decision Tree:
<img width="1470" height="956" alt="Screenshot 2026-04-14 at 9 58 44 PM" src="https://github.com/user-attachments/assets/efc6b373-a3ca-484c-9fb5-ca93191d12c2" />

## 📈 Logistic Regression & ROC Curve:
<img width="715" height="560" alt="Screenshot 2026-04-14 at 9 08 23 PM" src="https://github.com/user-attachments/assets/bf337176-19d9-4af8-8812-eb61af2150f7" /> 

This is an example from the app using the Titanic dataset.

## KNN:
<img width="732" height="584" alt="Screenshot 2026-04-14 at 9 09 31 PM" src="https://github.com/user-attachments/assets/c6d4def7-e7fc-4829-844d-a323fcf97827" /> 

This is an example from the app using the Titanic dataset.

# References: 
- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Machine Learning Algorithms Cheat Sheet](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms-cheat-sheet/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
