# 🌍 World Bank Data - Exploratory Data Analysis (EDA)

This project focuses on **cleaning**, **preprocessing**, and **exploring** a World Bank dataset to uncover patterns, trends, and relationships between various economic indicators using Python and popular data science libraries.

---

## 📌 Objective

**Understand the data using descriptive statistics and visualizations** to prepare it for further analysis or machine learning models.

---

## 🧰 Tools & Libraries

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

---

## 🗂️ Dataset

- **File**: `world_bank_data_2025.csv`
- **Source**: World Bank (assumed)
- **Contents**: Economic indicators like GDP, inflation, unemployment rate, public debt, etc.

---

## 🧼 Data Cleaning & Preprocessing

1. **Missing Values**:  
   - Numerical features: filled using **median**  
   - Categorical features: filled using **mode**

2. **Encoding**:  
   - Applied **One-Hot Encoding** to categorical columns

3. **Normalization**:  
   - Scaled numerical features using **StandardScaler**

4. **Outlier Removal**:  
   - Used **IQR method** with visual confirmation via boxplots

---

## 📊 Exploratory Data Analysis (EDA)

### 1. **Summary Statistics**
- Generated `.describe()` and `.info()` outputs
- Identified data types and missing/null values

### 2. **Histograms**
- Plotted histograms for all numeric columns to observe data distribution

### 3. **Boxplots**
- Used to detect and verify outliers in numeric data

### 4. **Pairplot**
- Used `seaborn.pairplot()` to visualize relationships between selected key features

### 5. **Correlation Heatmap**
- Identified strong/weak correlations between features using a heatmap

---

## 📌 Key Insights

- **GDP and GNI** show strong correlation
- **Government Revenue/Expense** are positively correlated
- **Inflation vs Interest Rate** shows inverse patterns
- **Anomalies/outliers** observed in GDP, Public Debt, and Income distributions
