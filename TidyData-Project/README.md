# Project Overview: 
The dataset I am using contains records of 2008 Olympic medalists, but it is structured in a "wide" format.
- Structure: Each row represents an individual athlete, but the table contains 71 columns where the variables Gender and Sport are combined within the column headers (male_rowing, female_swimming, etc). These columns indicate the "tier" of the medal won (Gold, Silver, or Bronze) as a cell value. Because an athlete only wins a medal in their specific sport, this results with unnecessary NaN (empty) values for many of the cells.

 This is a small preview of the data:
```
       medalist_name male_archery female_archery male_athletics  \
0    Aaron Armstrong          NaN            NaN           gold   
1      Aaron Peirsol          NaN            NaN            NaN   
2   Abdullo Tangriev          NaN            NaN            NaN   
3  Abeer Abdelrahman          NaN            NaN            NaN   
4            Abhinav          NaN            NaN            NaN  
```

### The primary objective of this project is divided into two key goals:

- First Goal: To take an "untidy" dataset and apply Tidy Data Principles.
The tidy data principles are, each variable should have its own column, each observation should have its own row, and each type of unit should form a table. In this project, we addressed specific problems to create a tidy dataset:
    - Multiple variables are stored in one column: Even after melting, "Gender" and "Sport" were combined. I then split these to ensure each variable forms a column

    - Each observation forms a row: I reshaped the data so that every row represents a single observational unit—in here it is an individual medal win.

- Second Goal: To take the tidy dataset and perform an Exploratory Data Analysis (EDA)
By now using the Tidy data I now have a structured data set to explore and generate visualizations and summary statistics (such as the distribution of medals by gender) without complex filtering or steps in the original data set. This allows us to cleanly communicate the data's story and its features.

# Instructions: 
To run this project follow these steps:

1. Download the Dataset: Ensure olympics_08_medalists.csv is in your project directory.

2. Install libraries used in the notebook:

``` 
# Data handling
import pandas as pd  
# Import 2008 Olypics dataset from a CSV file into a pandas DataFrame.
df = pd.read_csv('olympics_08_medalists.csv')
print(df.head())

# Plotting library for statistical data visualization          
import seaborn as sns     

# Plotting library for custom graphs 
import matplotlib.pyplot as plt 
```   


3. Run the Notebook: Open Tidy_Data_Notebook.ipynb and execute the cells in order.

# Dataset Description: 
Outline the source of your data and any pre-processing steps.
Data is adapted from https://edjnet.github.io/OlympicsGoNUTS/2008/
Get to know the initial data through df.head() df.describe() etc. 
### References: 
- [Tidy Data Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Tidy Data Paper by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Visualization Choices](https://www.data-to-viz.com/)
- [Visualization Colors](https://xkcd.com/color/rgb/)
# Visual Examples: 

<img width="265" height="209" alt="Screenshot 2026-03-20 at 1 18 24 PM" src="https://github.com/user-attachments/assets/e7195b7e-45f1-4213-8ed6-6d0d9dcddb09" />

This visualization provides an analysis of medal distributions between male and female athletes, focusing on the five sports with the most medals awarded in the 2008 Olympic Games.

 <img width="266" height="209" alt="Screenshot 2026-03-20 at 1 16 59 PM" src="https://github.com/user-attachments/assets/d72ccb1a-9bba-4a00-9119-f9dccc200b36" />
 
This visualization shows the distribution of Olympic medals across genders, further categorized by medal tier (Gold, Silver, and Bronze).
