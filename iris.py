import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns


# Load iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the species column
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

 

# Display the first few rows
print(df.head())

#missing Values

print("\nMissing values:\n", df.isnull().sum())

# visiual relationships between feature

sns.pairplot(df, hue="species", diag_kind="hist")
plt.show()

# outliers boxplotbh 
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="species", y="sepal length (cm)")
plt.show()

# heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns=['species']).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Features Correlation Heatmap")
plt.show()