from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assignment 1
# Import and initially explore the dataset
df = pd.read_csv("student_info.csv")

# Clean the data by handling missing values and duplicates
print("\nMissing values per column:")
print(df.isnull().sum())
df.drop_duplicates(inplace=True)

# Standardize data types for analysis
print("Before standardization: ", df.dtypes)
df.drop(columns=["name", "student_id"], inplace=True)

categorical_cols = [
    "gender",
    "grade_level",
    "parent_education",
    "internet_access",
    "lunch_type",
    "extra_activities",
    "final_result",
]
for col in categorical_cols:
    df[col] = df[col].astype("category")

df["math_score"] = pd.to_numeric(df["math_score"], errors="coerce")
df["reading_score"] = pd.to_numeric(df["reading_score"], errors="coerce")
df["writing_score"] = pd.to_numeric(df["writing_score"], errors="coerce")
df["study_hours"] = pd.to_numeric(df["study_hours"], errors="coerce")
df["attendance_rate"] = pd.to_numeric(df["attendance_rate"], errors="coerce")

print("After standardization: ", df.dtypes)

# Perform EDA to understand the features of the dataset
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.describe(include="all"))

# Identify and visualize relationships between features
# Correlation matrix
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Assignment 2
# Select relevant features for clustering
selected_columns = [
    "math_score",
    "reading_score",
    "writing_score",
]

# Extract the features
X = df[selected_columns]

# Standard scale the data as appropriate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Assignment 3
# Using Elbow method to determine the optional number of clusters. Plot the WCSS with *k* and determine the elbow point (*k*)
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot WCSS vs K
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# Use Silouette score to determine the optimal number of clusters. Plot the Silouette score with *k* and determine the optional *k*
sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    preds = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, preds)
    sil_scores.append(score)

# Plot Silhouette score vs K
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker="s", color="orange")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.grid(True)
plt.tight_layout()
plt.show()

# Implement the K-Means algorithm and visualize the resulting clusters for each optional *k* in above questions
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Reduce dimensionality with PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["Cluster"] = cluster_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=80)
plt.title(f"KMeans Clustering (k={k_optimal}) Visualized with PCA")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
