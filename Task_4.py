import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import minimize

# Load your dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Extract FICO scores and default status
fico_scores = df["fico_score"].to_numpy()
defaults = df["default"].to_numpy()

# Number of buckets
num_buckets = 10





# Approach 1: MSE-Based Bucketing (Using K-Means Clustering)
def mse_bucketization(scores, num_buckets):
    scores = np.array(scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_buckets, random_state=42, n_init=10)
    kmeans.fit(scores)
    bucket_labels = kmeans.predict(scores)

    # Get bucket boundaries
    boundaries = np.sort(kmeans.cluster_centers_.flatten())
    return boundaries, bucket_labels


# Approach 2: Log-Likelihood Optimization (Dynamic Programming)
def log_likelihood_boundaries(fico_scores, defaults, num_buckets):
    """
    Optimizes bucket boundaries using log-likelihood maximization.
    """

    def likelihood_function(boundaries):
        boundaries = np.sort(boundaries)
        boundaries = [fico_scores.min()] + list(boundaries) + [fico_scores.max()]
        likelihood = 0

        for i in range(len(boundaries) - 1):
            mask = (fico_scores >= boundaries[i]) & (fico_scores < boundaries[i + 1])
            ni = np.sum(mask)  # Number of samples in bucket
            ki = np.sum(defaults[mask])  # Number of defaults

            # Avoid division by zero
            if ni == 0:
                continue  # Skip empty buckets

            # Ensure pi is in a valid range
            pi = max(min(ki / ni, 1 - 1e-6), 1e-6)  # Prevents log(0) or log(1)

            likelihood += ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

        return -likelihood  # Minimize negative log-likelihood

    # Initial guess for bucket boundaries
    initial_boundaries = np.linspace(fico_scores.min(), fico_scores.max(), num_buckets - 1)[1:-1]

    # Optimize bucket boundaries
    result = minimize(likelihood_function, initial_boundaries, method="Powell")

    optimized_boundaries = np.sort(result.x)
    return [fico_scores.min()] + list(optimized_boundaries) + [fico_scores.max()]


# Apply MSE-based method
mse_boundaries, mse_labels = mse_bucketization(fico_scores, num_buckets)

# Apply Log-Likelihood optimization
ll_boundaries = log_likelihood_boundaries(fico_scores, defaults, num_buckets)


# Assign ratings based on buckets
def assign_ratings(fico_scores, boundaries):
    ratings = np.zeros_like(fico_scores)
    for i, score in enumerate(fico_scores):
        ratings[i] = np.searchsorted(boundaries, score, side="right")
    return ratings


fico_ratings_mse = assign_ratings(fico_scores, mse_boundaries)
fico_ratings_ll = assign_ratings(fico_scores, ll_boundaries)

# Print Results
print("MSE-Based Buckets:", mse_boundaries)
print("Log-Likelihood Buckets:", ll_boundaries)
