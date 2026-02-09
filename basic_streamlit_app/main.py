import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
 
 #Title for Project
st.title("💻 Online Sales Analysis 💻")

# Image for Project 
st.image(
    "basic_streamlit_app/Images/shutter-speed-BQ9usyzHx_w-unsplash.jpg",
    width=400
)
col1, col2  = st.columns(2)

with col1:
        
        st.write(' Photo by Shutter Speed on Unsplash')
    
# Load CSV data file
df = pd.read_csv("basic_streamlit_app/data/Online_Sales_Data.csv")
# Brief explanation of Dashboard
st.markdown(
    """
I built this to understand insights of online consumer behavior. 
There are various interactive filters to understand similarities and 
differences based on region, spending category and payment methods to name a few.
"""
)
# -------------------------------------------------------------------
#  FILTERS
# -------------------------------------------------------------------


## TABS
tab_data, tab_behavior, tab_payment = st.tabs(
      [
            "🔍 Dataset Exploration",
        "🛒 Customer Purchasing Behavior",
        "💳 Payment Method Insights"
      ]
)
# -------------------------------------------------------------------
# Tab 1 Exploration of Data set 
# -------------------------------------------------------------------
with tab_data:
    st.title("Exploring the Online Sales Dataset 🔍")
# Brief intro of tab
    st.write(
        """
        This tab provides an overview of the online sales dataset used in this project.
        It allows users to understand the structure and contents of the data before
        exploring customer behavior and payment patterns.
        """
    )
# Data set frame
    st.subheader("Dataset Preview")
    st.dataframe(df)

# Understanding sales category distribution in all regions by bar chart
    st.subheader("Total Units Sold by Category (All Regions)")
    total_sales_by_category = (
        df.groupby("Product Category")["Units Sold"]
        .sum()
        .reset_index()
    )
    st.bar_chart(
        data=total_sales_by_category,
        x="Product Category",
        y="Units Sold"
    )
    # Create the pie chart
    fig, ax = plt.subplots()

    df['Product Category'].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=ax
)

    ax.set_title(
        'Percentage of Observations by Product Category',
        size=13
    )

    st.pyplot(fig)




