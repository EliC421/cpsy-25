import numpy as np
import pymc3 as pm
import theano.tensor as tt
from scipy import sparse

def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Generate low-rank data for testing."""
    X = np.random.randn(num, dimX)
    # Create a low-rank coefficient matrix: W = A * B^T
    W = np.dot(np.random.randn(dimX, rrank), np.random.randn(rrank, dimY))
    Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
    return X, Y

class BayesianReducedRankRegressor(object):
    """
    Bayesian Reduced Rank Regressor with Monotonic Effects for Ordinal Predictors.
    
    This model assumes a regression of the form:
        Y ~ N(X_mod A B^T, σ²)
    where the coefficient matrix is factorized as C = A B^T.
    
    Ordinal predictors in X (specified via ordinal_info) are transformed using a
    monotonic effect. For an ordinal predictor taking values in {0, ..., D}, the
    transformation is defined as:
    
        c_mo(x, ζ) = ∑_{i=1}^{x} ζ_i,
    
    where ζ is a simplex (i.e. ζ_i ≥ 0 and ∑ ζ_i = 1) with a Dirichlet(1) prior.
    
    Priors:
      - Each element of A ~ Laplace(0, b=1) to promote sparsity.
      - Each element of B ~ Laplace(0, b=1) to promote sparsity (allowing negative values).
      - For each ordinal predictor, ζ ~ Dirichlet(ones), which centers the effect around a linear trend.
      - Observation noise is fixed (default σ = 0.908).
      
    Parameters:
      X            : Predictor matrix (numpy array).
      Y            : Response matrix (numpy array).
      rank         : Rank constraint (latent dimensionality).
      noise        : Fixed observation noise standard deviation.
      ordinal_info : dict mapping column index (int) to the number of ordinal categories D.
                     For an ordinal predictor taking values in {0, …, D} a monotonic
                     transformation is applied.
    """
    def __init__(self, X, Y, rank, noise=0.908, ordinal_info=None):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        # Ensure inputs are 2D.
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)
        self.rank = rank
        self.noise = noise
        self.ordinal_info = ordinal_info  # e.g., {col_index: D}
        self.n_samples, self.num_features = self.X.shape
        self.num_targets = self.Y.shape[1]
        self.model = None
        self.trace = None

    def fit(self, samples=1000, tune=1000, random_seed=42):
        """
        Build the Bayesian model and sample from the posterior.
        
        Parameters:
          samples     : Number of posterior samples to draw.
          tune        : Number of tuning (burn-in) steps.
          random_seed : Seed for reproducibility.
        """
        with pm.Model() as model:
            # Set up the predictor data as a shared variable.
            X_shared = pm.Data('X_shared', self.X)
            
            # If ordinal predictors are specified, apply the monotonic transformation.
            # X_mod will replace ordinal columns with their transformed values.
            X_mod = X_shared
            if self.ordinal_info is not None:
                for col, D in self.ordinal_info.items():
                    # ζ is a simplex of length D for the ordinal predictor at column col.
                    zeta = pm.Dirichlet(f'zeta_{col}', a=np.ones(D), shape=(D,))
                    # Compute cumulative sum of ζ.
                    zeta_cum = tt.extra_ops.cumsum(zeta)
                    # Extract the ordinal column and cast to int32.
                    x_col = tt.cast(X_shared[:, col], 'int32')
                    # For each entry: if 0 then 0, else the cumulative sum at index (x-1).
                    transformed = tt.switch(tt.eq(x_col, 0), 0.0, zeta_cum[x_col - 1])
                    # Replace column col in X_mod with the transformed values.
                    X_mod = tt.set_subtensor(X_mod[:, col], transformed)
            
            # Priors for A and B (using Laplace for sparsity, allowing negativity).
            A = pm.Laplace('A', mu=0, b=1, shape=(self.num_features, self.rank))
            B = pm.Laplace('B', mu=0, b=1, shape=(self.num_targets, self.rank))
            
            # Expected value: μ = X_mod * A * B^T.
            mu = tt.dot(X_mod, tt.dot(A, B.T))
            
            # Likelihood with fixed Gaussian noise.
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=self.noise, observed=self.Y)
            
            # Sample from the posterior.
            self.trace = pm.sample(samples, tune=tune, random_seed=random_seed, progressbar=True)
            self.model = model

    def predict(self, X_new):
        """
        Predict responses for new data X_new using the posterior means.
        
        For ordinal predictors, the monotonic transformation is applied using the
        posterior mean of the ζ parameters.
        
        Parameters:
          X_new: New predictor data (2D array).
          
        Returns:
          Predicted responses computed as: X_new_transformed * (A_mean * B_mean^T)
        """
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        # Copy new data so we can modify ordinal columns.
        X_new_trans = X_new.copy()
        if self.ordinal_info is not None:
            for col, D in self.ordinal_info.items():
                # Obtain the posterior mean for ζ for this ordinal predictor.
                zeta_mean = self.trace.get_values(f'zeta_{col}', combine=True).mean(axis=0)
                zeta_cum_mean = np.cumsum(zeta_mean)
                # For each value in the ordinal column, transform according to:
                # if x==0 then 0, else cumulative sum at index (x-1).
                transformed_col = np.array([0 if int(x)==0 else zeta_cum_mean[int(x)-1] for x in X_new_trans[:, col]])
                X_new_trans[:, col] = transformed_col
        # Get posterior means for A and B.
        A_mean = self.trace['A'].mean(axis=0)
        B_mean = self.trace['B'].mean(axis=0)
        return np.dot(X_new_trans, np.dot(A_mean, B_mean.T))

    def __str__(self):
        return f'Bayesian Reduced Rank Regressor with Monotonic Effects (rank = {self.rank})'









