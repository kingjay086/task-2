import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("C:/Users/jay30/OneDrive/Documents/myprojects/python/world_bank_data_2025.csv")

# Step 2: Basic Info
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Summary Statistics
print("\n📈 Summary Statistics:")
print(df.describe(include='all'))

# Step 4: Handle missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if not df[col].mode().empty:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Step 5: Visualizations

# 5.1 Histograms for numeric features
df[num_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

# 5.2 Boxplots for numeric features
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5.3 Pairplot for relationships (limit to 5 columns to avoid overload)
sns.pairplot(df[num_cols].iloc[:, :5])
plt.suptitle("Pairplot of Selected Numerical Features", y=1.02)
plt.show()

# 5.4 Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Identify and Remove Outliers (IQR Method for Selected Columns)
selected_cols = num_cols[:5]  # Adjust based on dataset
for col in selected_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

plt.figure(figsize=(15, 6))
sns.boxplot(data=df[selected_cols])
plt.title("Boxplot of Numerical Features (After Outlier Removal)")

# Step 7: Post-Cleaning Summary
print("\n🔍 Summary After Cleaning:")
print(df.describe())

# Final preview
print("\n✅ Final Cleaned Data Sample:")
print(df.head())
