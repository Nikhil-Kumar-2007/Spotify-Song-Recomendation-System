# Spotify Music Analysis Using Machine Learning Algorithms


## Project Overview

This project focuses on analyzing Spotify music data using various Machine Learning algorithms. The main objective is to explore patterns in music tracks and group similar songs based on their audio features.

The project uses both supervised and unsupervised learning techniques including:

- K-Means Clustering
- K-Nearest Neighbors (KNN)
- Gaussian Mixture Model (GMM)
- DBSCAN

Spotify datasets contain several audio-related features such as danceability, energy, loudness, tempo, valence, acousticness, and more. These features help in understanding music characteristics and enable intelligent clustering and recommendation systems.

---

# Objectives

- Perform Exploratory Data Analysis (EDA) on Spotify data
- Clean and preprocess the dataset
- Apply clustering algorithms to identify similar music groups
- Compare different machine learning models
- Build a recommendation/classification system using KNN
- Visualize clusters and hidden patterns in music data
- Evaluate model performance using appropriate metrics

---

# Dataset Information

The dataset contains Spotify tracks along with their audio features.

## Features Used

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration
- Popularity

These features are used to train and evaluate machine learning models.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- Jupyter Notebook

---

# Exploratory Data Analysis (EDA)

Before applying machine learning algorithms, the dataset was analyzed using various EDA techniques.

## EDA Steps

- Handling missing values
- Removing duplicate records
- Feature selection
- Feature scaling
- Correlation analysis
- Outlier detection
- Data visualization

## Visualizations Used

- Heatmaps
- Histograms
- Pairplots
- Boxplots
- PCA Visualization

EDA helped in understanding relationships between different music features and improving model performance.

---

# Machine Learning Models

## 1. K-Means Clustering

K-Means is an unsupervised learning algorithm used to divide data into multiple clusters based on similarity.

### Process

1. Standardize the data
2. Select optimal number of clusters
3. Train K-Means model
4. Assign songs to clusters
5. Visualize clusters

### Advantages

- Fast and efficient
- Easy to implement
- Works well for large datasets

### Limitations

- Sensitive to outliers
- Requires predefined number of clusters

### Evaluation Metrics

- Silhouette Score
- Inertia Score

---

## 2. K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification and recommendation tasks.

### Process

1. Split dataset into training and testing data
2. Normalize features
3. Train KNN model
4. Predict nearest neighbors

### Applications in Project

- Music recommendation
- Song classification
- Similar track prediction

### Advantages

- Simple and effective
- Good for recommendation systems

### Limitations

- Slow for very large datasets
- Sensitive to feature scaling

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 3. Gaussian Mixture Model (GMM)

GMM is a probabilistic clustering algorithm that assumes data points belong to different Gaussian distributions.

### Features

- Soft clustering
- Probability-based grouping
- Handles overlapping clusters

### Advantages

- More flexible than K-Means
- Captures complex data distributions

### Limitations

- Computationally expensive
- Sensitive to initialization

### Evaluation Metrics

- AIC Score
- BIC Score
- Log Likelihood

---

## 4. DBSCAN Clustering

DBSCAN is a density-based clustering algorithm used to identify dense regions and detect noise in the dataset.

### Features

- Detects outliers
- Handles arbitrary cluster shapes
- Does not require predefined clusters

### Advantages

- Excellent for noisy datasets
- Works well for non-linear clusters

### Limitations

- Parameter tuning can be difficult
- Struggles with varying densities

---

# Dimensionality Reduction

Principal Component Analysis (PCA) was used to reduce high-dimensional data into 2D space for visualization.

## Benefits of PCA

- Simplifies visualization
- Reduces dimensionality
- Improves interpretability

---

# Results and Insights

- K-Means successfully grouped songs with similar audio characteristics
- KNN provided accurate song classification and recommendation
- GMM captured overlapping music patterns effectively
- DBSCAN identified noisy and outlier tracks successfully
- PCA visualization clearly showed separation between clusters

---

# Model Comparison

| Algorithm | Type | Main Purpose |
|---|---|---|
| K-Means | Unsupervised | Cluster similar songs |
| KNN | Supervised | Recommendation & classification |
| GMM | Unsupervised | Probabilistic clustering |
| DBSCAN | Unsupervised | Noise detection & clustering |

---

# Future Improvements

- Integrate Spotify API
- Build a real-time recommendation system
- Deploy project using Streamlit or Flask
- Add Deep Learning models
- Improve recommendation accuracy
- Use hybrid recommendation systems

---

# Learning Outcomes

This project helped in understanding:

- Data preprocessing
- Feature engineering
- Clustering algorithms
- Classification techniques
- Dimensionality reduction
- Data visualization
- Machine Learning evaluation metrics

---

# Conclusion

This project demonstrates how machine learning algorithms can be applied to Spotify music data for clustering, classification, and recommendation purposes. By comparing K-Means, KNN, GMM, and DBSCAN, the project provides insights into different approaches for analyzing music patterns and building intelligent recommendation systems.

---

# Deployment Link
https://spotify-song-recomendation-169.streamlit.app/

---

