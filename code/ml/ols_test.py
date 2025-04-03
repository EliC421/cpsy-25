import pandas as pd

def load_data(x_path, y_path):
    """
    Load X and Y data from CSV files.
    Assumes the first row contains headers.

    Parameters:
      x_path : str, path to X data CSV
      y_path : str, path to Y data CSV

    Returns:
      X : numpy array
      Y : numpy array
    """
    X_df = pd.read_csv(x_path)
    Y_df = pd.read_csv(y_path)

    # Optionally convert True/False to 1/0 if needed
    X_df = X_df.replace({True: 1, False: 0})
    print('NUMPY VERSION OF X (DEBUG):' + '\n' + str(X_df.to_numpy()))
    return X_df.to_numpy(), Y_df.to_numpy()

# Set file paths
x_path = "../../data/final/x.csv"  # ⬅️ Replace this with your actual file path
y_path = "../../data/final/y.csv"  # ⬅️ Replace this with your actual file path

# Load data
X, Y = load_data(x_path, y_path)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)
print(model.score(X, Y))  # R^2

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def multivariate_r2_permutation_test(X, Y, n_permutations=1000, random_state=42):
    np.random.seed(random_state)
    n_targets = Y.shape[1]

    # Fit on original data
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    # R² for each target
    real_r2 = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(n_targets)]

    # Collect permutation R²s
    permuted_r2s = np.zeros((n_permutations, n_targets))
    for b in range(n_permutations):
        for i in range(n_targets):
            Y_perm = np.random.permutation(Y[:, i])
            model.fit(X, Y_perm)
            Y_perm_pred = model.predict(X)
            permuted_r2s[b, i] = r2_score(Y_perm, Y_perm_pred)

    return np.array(real_r2), permuted_r2s


real_r2, perm_r2s = multivariate_r2_permutation_test(X, Y, n_permutations=1000)

for i, r2 in enumerate(real_r2):
    p_value = np.mean(perm_r2s[:, i] >= r2)
    print(f"Target {i}: R² = {r2:.4f}, p-value = {p_value:.4f}")
