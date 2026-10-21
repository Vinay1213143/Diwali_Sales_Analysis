# ============================================================
# DIWALI SALES ANALYSIS
# ============================================================

# Import required Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set a consistent style for Seaborn charts
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. LOAD THE CSV DATA
# ------------------------------------------------------------

# Get the folder where this Python script is saved.
# Keep "Diwali Sales Data.csv" in the same folder as this .py file.
file_path = Path(__file__).parent / r"C:\Users\vinay\OneDrive\Desktop\ThinkQuotient Generative AI\Projects\Python_Diwali_Sales_Analysis\Diwali Sales Data.csv"

# Read the CSV file.
# unicode_escape helps read this dataset if it contains unusual characters.
df = pd.read_csv(file_path, encoding="unicode_escape")

# Display the number of rows and columns
print("Original dataset shape:", df.shape)

# Display the first five rows
print("\nFirst five rows:")
print(df.head())

# Display column names, data types, and non-null counts
print("\nDataset information:")
df.info()

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

# Remove unrelated or blank columns.
# These column names must exist in the CSV file.
df.drop(columns=["Status", "unnamed1"], inplace=True)

# Check the number of missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Remove rows containing missing values
df.dropna(inplace=True)

# Convert Amount to integer data type
df["Amount"] = df["Amount"].astype(int)

# Show the dataset shape after cleaning
print("\nDataset shape after cleaning:", df.shape)

# Display the data type of the Amount column
print("\nAmount column data type:", df["Amount"].dtype)

# Display all column names
print("\nColumn names:")
print(df.columns)

# Rename Marital_Status to Shaadi in a separate DataFrame for demonstration.
# The original df is not changed by this line.
renamed_df = df.rename(columns={"Marital_Status": "Shaadi"})
print("\nColumns after renaming Marital_Status to Shaadi (preview):")
print(renamed_df.columns)

# Show descriptive statistics for numeric columns
print("\nDescriptive statistics for numeric columns:")
print(df.describe())

# Show descriptive statistics for selected columns
print("\nDescriptive statistics for Age, Orders, and Amount:")
print(df[["Age", "Orders", "Amount"]].describe())

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

# -------------------------
# 3.1 Gender
# -------------------------

# Count the number of records for each gender and display labels on bars
plt.figure(figsize=(7, 5))
ax = sns.countplot(x="Gender", data=df)

for bars in ax.containers:
    ax.bar_label(bars)

plt.title("Number of Buyers by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Calculate total sales amount for each gender, highest first
sales_gen = (
    df.groupby("Gender", as_index=False)["Amount"]
    .sum()
    .sort_values(by="Amount", ascending=False)
)

# Plot total sales amount by gender
plt.figure(figsize=(7, 5))
sns.barplot(x="Gender", y="Amount", data=sales_gen)
plt.title("Total Sales Amount by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Amount")
plt.tight_layout()
plt.show()

# -------------------------
# 3.2 Age Group
# -------------------------

# Count buyers by age group and gender
plt.figure(figsize=(9, 5))
ax = sns.countplot(data=df, x="Age Group", hue="Gender")

for bars in ax.containers:
    ax.bar_label(bars)

plt.title("Number of Buyers by Age Group and Gender")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Calculate total sales amount for each age group, highest first
sales_age = (
    df.groupby("Age Group", as_index=False)["Amount"]
    .sum()
    .sort_values(by="Amount", ascending=False)
)

# Plot total sales amount by age group
plt.figure(figsize=(9, 5))
sns.barplot(x="Age Group", y="Amount", data=sales_age)
plt.title("Total Sales Amount by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Amount")
plt.tight_layout()
plt.show()

# -------------------------
# 3.3 State
# -------------------------

# Calculate total orders for each state and select the top 10 states
sales_state_orders = (
    df.groupby("State", as_index=False)["Orders"]
    .sum()
    .sort_values(by="Orders", ascending=False)
    .head(10)
)

# Plot the top 10 states by number of orders
plt.figure(figsize=(15, 5))
sns.barplot(data=sales_state_orders, x="State", y="Orders")
plt.title("Top 10 States by Number of Orders")
plt.xlabel("State")
plt.ylabel("Total Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total sales amount for each state and select the top 10 states
sales_state_amount = (
    df.groupby("State", as_index=False)["Amount"]
    .sum()
    .sort_values(by="Amount", ascending=False)
    .head(10)
)

# Plot the top 10 states by total sales amount
plt.figure(figsize=(15, 5))
sns.barplot(data=sales_state_amount, x="State", y="Amount")
plt.title("Top 10 States by Total Sales Amount")
plt.xlabel("State")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# 3.4 Marital Status
# -------------------------

# Count buyers by marital status and add count labels to bars
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="Marital_Status")

for bars in ax.containers:
    ax.bar_label(bars)

plt.title("Number of Buyers by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Calculate total sales amount by marital status and gender
sales_marital_gender = (
    df.groupby(["Marital_Status", "Gender"], as_index=False)["Amount"]
    .sum()
    .sort_values(by="Amount", ascending=False)
)

# Compare total sales amount by marital status and gender
plt.figure(figsize=(7, 5))
sns.barplot(
    data=sales_marital_gender,
    x="Marital_Status",
    y="Amount",
    hue="Gender"
)
plt.title("Total Sales by Marital Status and Gender")
plt.xlabel("Marital Status")
plt.ylabel("Total Amount")
plt.tight_layout()
plt.show()

# -------------------------
# 3.5 Occupation
# -------------------------

# Count buyers by occupation and display labels on bars
plt.figure(figsize=(20, 5))
ax = sns.countplot(data=df, x="Occupation")

for bars in ax.containers:
    ax.bar_label(bars)

plt.title("Number of Buyers by Occupation")
plt.xlabel("Occupation")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total sales amount for each occupation
sales_occupation = (
    df.groupby("Occupation", as_index=False)["Amount"]
    .sum()
    .sort_values(by="Amount", ascending=False)
)

# Plot total sales amount by occupation
plt.figure(figsize=(20, 5))
sns.barplot(data=sales_occupation, x="Occupation", y="Amount")
plt.title("Total Sales Amount by Occupation")
plt.xlabel("Occupation")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# 3.6 Product Category
# -------------------------

# Count buyers/records by product category and label each bar
plt.figure(figsize=(20, 5))
ax = sns.countplot(data=df, x="Product_Category")

for bars in ax.containers:
    ax.bar_label(bars)

plt.title("Number of Records by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total sales amount for each product category and select top 10
sales_category = (
    df.groupby("Product_Category", as_index=False)["Amount"]
    .sum()
    .sort_values(by="Amount", ascending=False)
    .head(10)
)

# Plot the top 10 product categories by total sales amount
plt.figure(figsize=(20, 5))
sns.barplot(data=sales_category, x="Product_Category", y="Amount")
plt.title("Top Product Categories by Total Sales Amount")
plt.xlabel("Product Category")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# 3.7 Top Products
# -------------------------

# Calculate total orders for each product and select the top 10
sales_product = (
    df.groupby("Product_ID", as_index=False)["Orders"]
    .sum()
    .sort_values(by="Orders", ascending=False)
    .head(10)
)

# Plot the top 10 products by number of orders
plt.figure(figsize=(20, 5))
sns.barplot(data=sales_product, x="Product_ID", y="Orders")
plt.title("Top 10 Products by Number of Orders")
plt.xlabel("Product ID")
plt.ylabel("Total Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Alternative way to find and plot the top 10 most-ordered products
top_10_products = (
    df.groupby("Product_ID")["Orders"]
    .sum()
    .nlargest(10)
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))
top_10_products.plot(kind="bar")
plt.title("Top 10 Most Ordered Products")
plt.xlabel("Product ID")
plt.ylabel("Total Orders")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. CONCLUSION
# ------------------------------------------------------------

print("\nAnalysis completed successfully.")
print("Use the charts above to compare buyers and sales by gender, age group,")
print("state, marital status, occupation, product category, and product ID.")
