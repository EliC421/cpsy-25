#!C:\Users\jedim\miniforge3\envs\pymc_env
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
from pytensor.gradient import disconnected_grad
import uuid
import arviz as az

class RRR:
    """
    Reduced Rank Regression (RRR) with Bayesian inference using PyMC.
    
    The regression model is:
    
        Y ~ Normal( (X_mod dot A) dot B^T, sigma )
    
    where:
      - X_mod is X with ordinal columns transformed via a monotonic effect.
      - For each ordinal predictor (column index j in ordinal_info), 
          ζ_j ~ Dirichlet( a=ones* (1/D) )  [with D given in ordinal_info[j]]
        and the transformation for an observed value x is:
          cmo(x, ζ_j) = sum_{i=1}^{x} ζ_{j,i}   (with 0 if x==0)
      - For continuous predictors, the raw X values are used.
      - A (shape: p x rank) has Laplace(0,1) priors.
      - B (shape: q x rank) has LogNormal(mu_B, sigma_B) priors.
      - sigma (error SD) defaults to 0.908 but can be changed.
    """
    def __init__(self, rank, noise=0.25, ordinal_info=None, mu_B=0, sigma_B=1):
        """
        Parameters:
          rank         : int, the reduced rank (k) constraint, must be <= min(n, p).
          noise        : float, the observation noise (default 0.908).
          ordinal_info : dict mapping ordinal predictor column indices to maximum level D.
                         For example, {2: 5} means column 2 is ordinal with levels {0,1,...,5}.
          mu_B         : mean parameter for B's LogNormal prior.
          sigma_B      : sigma parameter for B's LogNormal prior.
        """
        self.rank = rank
        self.noise = noise
        self.ordinal_info = ordinal_info if ordinal_info is not None else {}
        self.mu_B = mu_B
        self.sigma_B = sigma_B
        self.model = None
        self.trace = None
        self._model_id = uuid.uuid4().hex  # unique identifier for each model build

    def build_model(self, X, Y):
        if self.model is not None:
            return self.model
        """
        Constructs the PyMC model using shared data.
        
        Parameters:
          X : numpy array of shape (n, p), predictor matrix.
          Y : numpy array of shape (n, q), response matrix.
          
        Returns:
          model : the built PyMC model.
        """
        n, p = X.shape
        q = Y.shape[1]
        
        with pm.Model() as model:
            # Use shared data so that new data can be set later for prediction.
            X_shared = pm.Data('X_shared', X)
            
            # Build modified X: for ordinal columns, transform via monotonic effect.
            X_mod_cols = []
            for j in range(p):
                if j in self.ordinal_info:
                    D = self.ordinal_info[j]  # maximum level for ordinal predictor j
                    # Latent simplex for ordinal predictor j.
                    zeta = pm.Dirichlet(f'zeta_{j}', a=np.ones(D) * (1.0 / D), shape=(D,))
                    # Get the jth column (cast to int); assume observed values are integers (0,...,D).
                    x_col = pt.cast(X_shared[:, j], 'int32')
                    
                    # Define a scalar monotonic transformation.
                    def monotonic_transform(x_val, cumsum_zeta):
                        # if x_val > 0, return cumulative sum at index x_val-1, else 0.
                        return pt.switch(pt.gt(x_val, 0), cumsum_zeta[x_val - 1], 0.)
                    
                    # Apply the transformation to each element of x_col using a scan.
                    var_name = f'zeta_{j}_{self._model_id}'
                    zeta = pm.Dirichlet(var_name, a=np.ones(D) * (1.0 / D), shape=(D,))
                    # Detach the cumulative sum from the graph:
                    zeta_cumsum = disconnected_grad(pt.cumsum(zeta))
                    transformed = pt.switch(pt.gt(x_col, 0), pt.take(zeta_cumsum, x_col - 1), 0.)
                    X_mod_cols.append(transformed)
                else:
                    # Continuous predictor: use the original value.
                    X_mod_cols.append(X_shared[:, j])
            # Stack the columns back together to form the modified predictor matrix.
            X_mod = pt.stack(X_mod_cols, axis=1)
            
            # Coefficient matrix A: for all predictors (p x rank) with Laplace prior.
            A = pm.Laplace('A', mu=0, b=1, shape=(p, self.rank))
            # Compute latent representation (n x rank).
            latent = pt.dot(X_mod, A)
            
            # Factor matrix B: (q x rank) with a LogNormal prior.
            B = pm.Lognormal('B', mu=self.mu_B, sigma=self.sigma_B, shape=(q, self.rank))
            # Predicted mean: (n x q)
            mu_Y = pt.dot(latent, B.T)
            
            # Likelihood for responses.
            Y_obs = pm.Normal('Y_obs', mu=mu_Y, sigma=self.noise, observed=Y)
            
        self.model = model
        return model

    def train(self, X, Y, draws=1000, tune=1000, **kwargs):
        """
        Build the model with data and run MCMC sampling.
        
        Parameters:
          X      : numpy array, predictor matrix.
          Y      : numpy array, response matrix.
          draws  : number of posterior samples to draw.
          tune   : number of tuning steps.
          kwargs : any additional keyword arguments to pm.sample.
          
        Returns:
          trace : the sampled posterior trace.
        """
        model = self.build_model(X, Y)
        with model:
            self.trace = pm.sample(draws=draws, tune=tune, **kwargs)
        return self.trace

    def predict(self, X_new, posterior_predictive=True, **kwargs):
        """
        Generate predictions for new data.
        
        Parameters:
          X_new               : numpy array, new predictor data (n_new x p).
          posterior_predictive: if True, return samples from the posterior predictive;
                                otherwise, return the mean prediction computed from posterior means.
          kwargs              : additional arguments passed to pm.sample_posterior_predictive.
                                
        Returns:
          If posterior_predictive=True: an array of posterior predictive samples for Y.
          Otherwise: the mean prediction (n_new x q).
        """
        if self.model is None:
            raise ValueError("Model has not been built/trained yet.")
            
        with self.model:
            # Update shared data
            pm.set_data({'X_shared': X_new})
            
            if posterior_predictive:
                # Use 'draws' instead of 'samples'
                ppc = pm.sample_posterior_predictive(self.trace)
                print("Keys in posterior predictive:", ppc.keys())  # Debugging step
                return ppc.posterior_predictive["Y_obs"].values
            else:
                # Compute the posterior mean prediction
                A_mean = self.trace.posterior['A'].mean(dim=("chain", "draw"))
                B_mean = self.trace.posterior['B'].mean(dim=("chain", "draw"))
                
                X_new_mod = X_new.copy()
                for j in self.ordinal_info:
                    D = self.ordinal_info[j]
                    zeta_samples = self.trace.posterior[f'zeta_{j}']
                    zeta_mean = zeta_samples.mean(dim=("chain", "draw"))
                    cumsum_zeta = np.cumsum(zeta_mean, axis=-1)
                    X_new_mod[:, j] = np.array([cumsum_zeta[int(x) - 1] if x > 0 else 0 for x in X_new[:, j]])
                    
            latent_mean = X_new_mod @ A_mean.values  # Convert to NumPy
            mu_Y_mean = latent_mean @ B_mean.values.T  # Convert to NumPy
            return mu_Y_mean

# Generate ideal data and test RRR class
def generate_test_data(n=200, p=5, q=3, ordinal_col=2, max_level=5):
    """
    Generate synthetic test data.
    
    Parameters:
      n           : int, number of observations.
      p           : int, number of predictors.
      q           : int, number of response variables.
      ordinal_col : int, index of the ordinal predictor column.
      max_level   : int, maximum ordinal level (i.e., D in the RRR model).
      
    Returns:
      X : numpy array of shape (n, p)
      Y : numpy array of shape (n, q)
    """
    X = np.empty((n, p))
    for j in range(p):
        if j == ordinal_col:
            # Generate integer levels 0,...,max_level
            X[:, j] = np.random.randint(0, max_level + 1, size=n)
        else:
            # Continuous predictors sampled from a standard normal distribution
            X[:, j] = np.random.randn(n)
    
    # Create a low-rank structure for Y.
    rank = 2
    # True latent coefficients for predictors and responses.
    A_true = np.random.randn(p, rank)
    B_true = np.random.randn(q, rank)
    
    # For the ordinal column, mimic a monotonic transformation.
    # (For testing we use a simple scaling: x -> x / max_level if x > 0, else 0)
    X_mod = X.copy()
    X_mod[:, ordinal_col] = np.where(X[:, ordinal_col] > 0, X[:, ordinal_col] / max_level, 0)
    
    # Generate latent structure and add a small noise.
    latent = np.dot(X_mod, A_true)
    noise_std = 0.1
    Y = np.dot(latent, B_true.T) + np.random.randn(n, q) * noise_std
    
    return X, Y

def main():
    # Seed for reproducibility.
    np.random.seed(42)
    
    # Generate test data.
    n, p, q = 200, 5, 3
    ordinal_info = {2: 5}  # Column index 2 is ordinal with levels 0,...,5
    X, Y = generate_test_data(n=n, p=p, q=q, ordinal_col=2, max_level=5)
    
    # Instantiate the RRR model.
    # Note: Here we set noise to a low value (e.g., 0.1) to reflect the data generating process.
    model = RRR(rank=2, noise=0.1, ordinal_info=ordinal_info, mu_B=0, sigma_B=1)
    
    # Train the model with a modest number of draws for testing.
    trace = model.train(X, Y, draws=300, tune=300, target_accept=0.9)
    
    # Generate predictions using posterior predictive sampling.
    ppc_samples = model.predict(X, posterior_predictive=True, samples=100)
    print("Posterior predictive sample shape:", ppc_samples.shape)
    
    # Also compute the mean prediction using posterior mean parameters.
    mean_prediction = model.predict(X, posterior_predictive=False)
    print("Mean prediction shape:", mean_prediction.shape)
    print("First 5 mean predictions:\n", mean_prediction[:5])
    print("First 5 Response Variables:\n", Y[:5])
    
if __name__ == "__main__":
    main()
