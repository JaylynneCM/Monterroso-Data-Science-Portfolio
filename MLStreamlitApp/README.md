# 🤖Project Overview🤖: 
The goal of this project is to provide an interactive, web-based platform for supervised machine learning using **Streamlit**. 

- **The Application**: This app allows users to transition from raw data to model evaluation in one interface. Users can upload their own datasets, select from various supervised learning models, and fine-tune hyperparameters in real-time. This tool invites exploration of model behavior and makes performance metrics accessible to users.

### The primary objective of this project is divided into two key goals:

## - ⚙️First Goal: Interactive Model Training & Hyperparameter Tuning
The app provides a "sandbox" environment where the user is in control. 
- **User Input**: Users can upload CSV files or choose from provided sample datasets.
- **Tuning**: Users utilize Streamlit widgets (sliders, dropdowns) to adjust parameters such as the 'K' in KNN or the depth of a Decision Tree.

## - 📈Second Goal: Real-Time Performance Visualization
The project aims to communicate results clearly through meaningful feedback.
- **Evaluation**: The app automatically calculates metrics such as Accuracy, Precision, and Recall.
- **Visuals**: It generates dynamic charts to provide a visual story of how well the model is performing on the selected data.

# Instructions📋: 
To run this project locally, follow these steps using Python:

1. Clone the repository and ensure your data files are in the root directory.
2. Install the required libraries:
```
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```
3. streamlit run the py
# App Features: 
The application is split into two distinct sections to guide the user in exploration:

1. ML Exploration 🧪
This is where the user can adjust hyperparameters and select model.

    Step 1: Data Input & Feature Selection.

    Step 2: Automated Preprocessing (Handling NaNs and Encoding).

    Step 3: Model Selection and Hyperparameter adjustment.

    Step 4: Execution and Visual Analysis.

2. ML Guide 📚
This is a page that explains key metrics used and provides definitions.
    Includes definitions for 
    - Metrics: Accuracy, Precision, Recall, and F1-Score.
    - Algorithms: breakdowns of how Logistic Regression, Decision Trees, and KNN function.
# Visual Examples: 
🌲 Decision Tree:
<img width="1252" height="868" alt="Screenshot 2026-04-14 at 5 51 52 PM" src="https://github.com/user-attachments/assets/c1dfc603-d6b1-4864-90e8-4bdde199d0a6" />

📈 Logistic Regression & ROC Curve:

KNN:

Visual Trade-off: Helps users understand the balance between the True Positive Rate and the False Positive Rate.
# References: 
- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Machine Learning Algorithms Cheat Sheet](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms-cheat-sheet/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
