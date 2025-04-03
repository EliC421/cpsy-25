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
          ζ_j ~ Dirichlet( a=ones * (1/D) )  [with D given in ordinal_info[j]]
        and the transformation for an observed value x is:
          cmo(x, ζ_j) = sum_{i=1}^{x} ζ_{j,i}   (with 0 if x==0)
      - For continuous predictors, the raw X values are used.
      - A (shape: p x rank) has Laplace(0,1) priors (reparameterized below).
      - B (shape: q x rank) has LogNormal(mu_B, sigma_B) priors (non-centered).
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
        n, p = X.shape
        q = Y.shape[1]
        
        with pm.Model() as model:
            X_shared = pm.Data('X_shared', X)
            X_mod_cols = []
            self.zeta_names = {}  # Dictionary to store zeta variable names
            
            for j in range(p):
                if j in self.ordinal_info:
                    D = self.ordinal_info[j]
                    # Create a unique variable name and store it
                    var_name = f'zeta_{j}_{self._model_id}'
                    self.zeta_names[j] = var_name
                    zeta = pm.Dirichlet(var_name, a=np.ones(D) * (1.0 / D), shape=(D,))
                    zeta_cumsum = disconnected_grad(pt.cumsum(zeta))
                    x_col = pt.cast(X_shared[:, j], 'int32')
                    transformed = pt.switch(pt.gt(x_col, 0), pt.take(zeta_cumsum, x_col - 1), 0.)
                    X_mod_cols.append(transformed)
                else:
                    X_mod_cols.append(X_shared[:, j])
            X_mod = pt.stack(X_mod_cols, axis=1)
            
            # --- Non-centered parameterization for coefficient matrix A ---
            # Instead of directly sampling A ~ Laplace(0,1), we use:
            #    A = sqrt(u_A/2) * A_tilde,  with u_A ~ Exponential(1) and A_tilde ~ Normal(0,1)
            b = pm.HalfCauchy('b', beta=10)
            u_A = pm.Exponential('u_A', lam=0.1, shape=(p, self.rank))
            A_tilde = pm.Normal('A_tilde', mu=0, sigma=1, shape=(p, self.rank))
            A = pm.Deterministic('A', b * A_tilde * pt.sqrt(u_A / 2))
            
            # Latent representation: (n x rank)
            latent = pt.dot(X_mod, A)
            
            # --- Non-centered parameterization for factor matrix B ---
            # Instead of sampling B ~ LogNormal(mu_B, sigma_B) directly,
            # sample B_tilde ~ Normal(0,1) and set B = exp(mu_B + sigma_B * B_tilde)
            B_tilde = pm.Normal('B_tilde', mu=0, sigma=1, shape=(q, self.rank))
            B = pm.Deterministic('B', pt.exp(self.mu_B + self.sigma_B * B_tilde))
            
            # Predicted mean: (n x q)
            mu_Y = pt.dot(latent, B.T)
            
            # Likelihood for responses.
            Y_obs = pm.Normal('Y_obs', mu=mu_Y, sigma=self.noise, observed=Y)
            
        self.model = model
        return model

    def train(self, X, Y, draws=2000, tune=1000, **kwargs):
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
            self.trace = pm.sample(draws=draws, tune=tune, chains=2, **kwargs)

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
                    var_name = self.zeta_names[j]
                    zeta_samples = self.trace.posterior[var_name]
                    zeta_mean = zeta_samples.mean(dim=("chain", "draw"))
                    cumsum_zeta = np.cumsum(zeta_mean, axis=-1)
                    X_new_mod[:, j] = np.array([cumsum_zeta[int(x) - 1] if x > 0 else 0 for x in X_new[:, j]])
                    
            latent_mean = X_new_mod @ A_mean.values  # Convert to NumPy
            mu_Y_mean = latent_mean @ B_mean.values.T  # Convert to NumPy
            return mu_Y_mean

import numpy as np
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

def main():
    import matplotlib.pyplot as plt
    # Set file paths
    x_path = "../../data/final/x.csv"  # ⬅️ Replace this with your actual file path
    y_path = "../../data/final/y.csv"  # ⬅️ Replace this with your actual file path

    # Load data
    X, Y = load_data(x_path, y_path)

    # TODO: Define ordinal columns and their max levels, e.g.:
    # If column 2 is ordinal with values from 0 to 5:
    ordinal_info = {1: 10}

    # Instantiate the RRR model
    model = RRR(rank=1, noise=0.2, ordinal_info=ordinal_info, mu_B=0, sigma_B=1)

    # Train the model
    trace = model.train(X, Y, draws=2000, tune=1500, target_accept=0.90)
    print(az.summary(trace, var_names=["A", "B", "b"], round_to=4, stat_focus="mean").to_string())
    print(az.summary(trace, var_names=["A_tilde", "B_tilde", "u_A"], round_to=3).to_string())
    az.plot_trace(trace, var_names=["b", "A", "B"])
    plt.tight_layout()
    plt.show()

    # Posterior predictive sampling
    ppc_samples = model.predict(X, posterior_predictive=True, samples=100)
    print("Posterior predictive sample shape:", ppc_samples.shape)

    # Posterior mean predictions
    mean_prediction = model.predict(X, posterior_predictive=False)
    print("Mean prediction shape:", mean_prediction.shape)
    print("Last 5 mean predictions:\n", mean_prediction[-5:])
    print("Last 5 Response Variables:\n", Y[-5:])
    from sklearn.metrics import mean_absolute_error

    # Calculate MAE
    mae = mean_absolute_error(Y, mean_prediction)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

if __name__ == "__main__":
    main()
