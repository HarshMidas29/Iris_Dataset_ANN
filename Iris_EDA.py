# importting Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')


# loading and reading dataset
   
df = pd.read_csv(r'C:\Users\Daham Yakandawala\Documents\Visual Studio 2017\ML_Module\Practicals\Iris_ANN\iris.data.csv')
print(df.head())

# shape of the data
df.shape

#data information 
df.info()

# describing the data
df.describe()

#column to list 
df.columns.tolist()

# check for missing values:
df.isnull().sum()

#checking duplicate values 
df.nunique()

#Scatterplot for two features
sns.scatterplot(x='Sepal Width', y='Petal Width', hue='Class', data=df)
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.title('Relationship Between Sepal Width and Petal Width by Seed Type')
plt.show()


# Count Plot
Petal_Width_counts = df['Petal Width'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(Petal_Width_counts.index, Petal_Width_counts, color='darkmagenta')
plt.title('Count Plot of Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Count')
plt.show()


# Set Seaborn style
sns.set_style("darkgrid")

# Identify numerical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# Plot distribution of each numerical feature
plt.figure(figsize=(14, len(numerical_columns) * 1.5))
for idx, feature in enumerate(numerical_columns, 1):
  plt.subplot(len(numerical_columns), 2, idx)
  sns.histplot(df[feature], kde=True)
  plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
  plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Using Seaborn to create a swarm plot
plt.figure(figsize=(10, 8))

sns.swarmplot(x="Petal Width", y="Sepal Width", data=df, palette='viridis')

plt.title('Swarm Plot for Petal Width and Sepal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Width')
plt.xticks(rotation=45)
plt.show()

# Using Seaborn to create a heatmap

plt.figure(figsize=(15, 10))

sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)

plt.title('Correlation Heatmap')
plt.show()