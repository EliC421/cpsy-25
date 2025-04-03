import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def compute_feature_importance(X, Y, feature_names=None, model_type='rf', standardize=True, random_state=42):
    """
    Compute feature importance for multivariate targets using a specified model.
    
    Parameters:
        X (np.ndarray): Predictor matrix (n_samples x n_features)
        Y (np.ndarray): Response matrix (n_samples x n_targets)
        feature_names (list): Optional list of feature names
        model_type (str): 'rf' for Random Forest, 'ridge' for Ridge Regression
        standardize (bool): Whether to standardize X
        random_state (int): Random seed
        
    Returns:
        importance_df (pd.DataFrame): Feature importances averaged across targets
    """
    n_targets = Y.shape[1]
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    importance_matrix = []

    for i in range(n_targets):
        y = Y[:, i]
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=random_state)
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0, random_state=random_state)
        else:
            raise ValueError("Unsupported model_type. Use 'rf' or 'ridge'.")

        model.fit(X, y)

        if model_type == 'rf':
            importance = model.feature_importances_
        else:  # Ridge
            importance = np.abs(model.coef_)

        importance_matrix.append(importance)

    importance_matrix = np.array(importance_matrix)  # (n_targets x n_features)
    mean_importance = importance_matrix.mean(axis=0)
    std_importance = importance_matrix.std(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Importance": mean_importance,
        "Std Dev": std_importance
    }).sort_values(by="Mean Importance", ascending=False)

    return importance_df

def plot_feature_importance(importance_df, top_n=15):
    df = importance_df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(df["Feature"][::-1], df["Mean Importance"][::-1], xerr=df["Std Dev"][::-1])
    plt.xlabel("Mean Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    X = pd.read_csv("../../data/final/x.csv").to_numpy()
    Y = pd.read_csv("../../data/final/y.csv").to_numpy()
    feature_names = pd.read_csv("../../data/final/x.csv").columns.tolist()

    importance_df = compute_feature_importance(X, Y, feature_names, model_type='rf')  # or 'ridge'
    print(importance_df.head(10))

    plot_feature_importance(importance_df)
