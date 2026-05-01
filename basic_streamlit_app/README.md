# 🖥️ Online Sales Data Explorer
## 📦 Project Overview 📦:
The dataset I am using contains global e-commerce transactions, detailing consumer purchases across various regions and product categories.

Structure: The data includes variables such as Product Category, Units Sold, Total Revenue, and Payment Method, mapped across different geographic regions and dates. 

🥇 Key Goal: Behavioral & Payment Intelligence
The project aims to provide a deep dive into online consumer behavior, such as trends, through interactive data exploration
I created an interactive Streamlit app for exploring consumer behavior in an online sales dataset. This is done through filtering based on region, payment method, and product category.

# Instructions📋:
1. Install streamlit
2. Use this code snippet
````
streamlit run basic_streamlit_app/main.py
````
# Dataset Description📈:
This dataset is adapted from [Kaggle: Online Sales Dataset - Popular Marketplace Data.](https://www.kaggle.com/datasets/shreyanshverma27/online-sales-dataset-popular-marketplace-data)
1. Feature Count: This dataset consists of 9 columns covering transaction details, product info, and geographical data.

2. Temporal Range: The records span from December 31, 2023, to August 26, 2024, allowing for multi-quarter trend analysis.

3. Transaction Volume: Includes unique records with Transaction IDs ranging from 10,001 to 10,240.

# App Features:
 I structured the app in three tabs with ways to access visualizations.
* 🔍 Dataset Exploration (To get an understanding of the dataset)
* 🛒 Customer Purchasing Behavior (Understand customer Regional insights)
* 💳 Payment Method Insights (Understand Payment methods utilized based on Region and Product category)

I used:
* Streamlit: For the interactive interface.
* Pandas: For data manipulation and filtering.
* Matplotlib & Seaborn: For creating data visualizations and charts.


# Visualizations: 

<img width="705" height="359" alt="Screenshot 2026-04-20 at 6 33 24 PM" src="https://github.com/user-attachments/assets/2082ff3b-e323-460e-a238-67181f5a595b" />

<img width="626" height="311" alt="Screenshot 2026-04-20 at 6 34 33 PM" src="https://github.com/user-attachments/assets/ecb9dc0d-0a0f-4cad-ba98-bdd42035ac9b" />

# References

* [Streamlit API Cheatsheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)
* [Matplotlib sheet](https://matplotlib.org/cheatsheets/)
