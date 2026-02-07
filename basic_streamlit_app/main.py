import streamlit as st
import pandas as pd
st.title("Hello, Streamlit!")
 
#Title for project
st.title("Online Sales Analysis")


# Load the CSV file
df = pd.read_csv("basic_streamlit_app/data/Online_Sales_Data.csv")

st.write("Here's our data:")
st.dataframe(df)

#Filter by Region
Region = st.selectbox("Select a Region", df["Region"].unique())
filtered_df = df[df["Region"] == Region]

st.write(f"Sales in {Region}:")
st.dataframe(filtered_df)