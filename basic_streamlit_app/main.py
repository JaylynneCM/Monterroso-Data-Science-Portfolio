import streamlit as st
import pandas as pd
 
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

st.write("Here's our data:")
st.dataframe(df)