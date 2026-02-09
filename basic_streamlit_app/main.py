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
# -------------------------------------------------------------------
# TAB 2: Customer Behavior
# -------------------------------------------------------------------
with tab_behavior:
    st.title("Customer Purchasing Behavior 🛒")
    st.write("This tab explores customer purchasing behavior by examining how product choices and spending patterns vary across regions, payment methods, and month.")

    # Filter by Region
    region = st.selectbox(
        "Select a region",
        df["Region"].unique(),
        key="behavior_region"
    )

    # Payment Filter
    payment_method = st.selectbox(
        "Select a payment method",
        df["Payment Method"].unique(),
        key="behavior_payment"
    )

    # Filter data based on selections
    filtered_df = df[
        (df["Region"] == region) &
        (df["Payment Method"] == payment_method)
    ]

    region_df = df[df["Region"] == region]
   
    # Chart 1: Items sold by category (Region)
    st.subheader(f"Items Sold by Category in {region}")

    category_products = (
        region_df.groupby("Product Category")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_products)

    # Chart: Revenue by category (Region + Payment Method)
    st.subheader(f"Total Revenue by Category in {region} using {payment_method}")
#Filtered df for chart
    category_data = (
        filtered_df.groupby("Product Category")["Total Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
#Barchart displayed
    st.bar_chart(category_data)

    # Add interactive button of Top sold items within top category bought for selected region
    st.markdown("🎉  Top category + top items")
    if st.button("Press me to see the top sold items!", key="top_items_"):
    # organize top product category in region
        top_category = (
            region_df.groupby("Product Category")["Units Sold"]
            .sum()
            .idxmax()
        )   
        st.success(f"⭐️ Top category in {region}: {top_category}") 
    #Top 5 products in the Category of region
        top_items = (
            region_df[region_df["Product Category"] == top_category]
            .groupby("Product Name")["Units Sold"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
         )
        st.write(f"Top 5 products in {top_category}:")
        st.dataframe(top_items)
    else:
         st.write("Click the button to reveal the top category and top sold items for the selected region.")
    # Monthly interactive with all regions
    st.subheader("Monthly Exploration all Regions 📍")
    df["Date"] = pd.to_datetime(df["Date"])

    # Create month  column (all regions)
    df["Month"] = df["Date"].dt.month_name()
# Selection box for month
    month = st.selectbox(
        "Select a month (all regions)",
        sorted(df["Month"].unique()),
        key="month_all_regions"
    )
    month_df = df[df["Month"] == month]

    st.write(f"Top product categories in **{month}** (all regions):")
#Table produced based on month selected from drop down
    month_categories = (
        month_df.groupby("Product Category")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
# Shows top 3 product Categories
    st.dataframe(month_categories.head(3))


# Summarize insights but hidden unless pressed   
    st.markdown("### Overall Consumer Insights")
    # Button creation
    if st.button("Reveal overall consumer insights", key="overall_insights_btn"):
        st.success("Key insights from the full dataset:")

        # Region with most total revenue
        top_region_revenue = (
            df.groupby("Region")["Total Revenue"]
            .sum()
            .idxmax()
        )
        # Most shopped category by units sold
        top_category_units = (
            df.groupby("Product Category")["Units Sold"]
            .sum()
            .idxmax()
        )
        # Category with highest total revenue 
        top_category_revenue = (
            df.groupby("Product Category")["Total Revenue"]
            .sum()
            .idxmax()
        )
        # Month with highest total units sold
        df_dates = df.copy()
        df_dates["Date"] = pd.to_datetime(df_dates["Date"])
        df_dates["Month"] = df_dates["Date"].dt.month_name()

        top_month = (
            df_dates.groupby("Month")["Units Sold"]
            .sum()
            .idxmax()
        )
    # Show summary of Insights
        st.write(
            f"""
            - The region with the **highest overall spending** is **{top_region_revenue}**📈.
            - The category that generated the **highest total revenue** is **{top_category_revenue}** 📈.
            - The **most frequently purchased product category** (by units sold) is **{top_category_units}** 📊.
            - The month with the **highest overall purchasing activity** is **{top_month}**🗓️.
            """
        )
    else:
        st.write("Click the button to view a summary of overall consumer insights.")




