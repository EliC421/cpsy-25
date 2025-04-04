#!C:\Users\jedim\miniforge3\envs\pymc_env
import numpy as np
from math import sqrt
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
from pytensor.gradient import disconnected_grad
import uuid
import arviz as az
import matplotlib.pyplot as plt




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
            X_mod_quad = pm.Data("X_mod_quad", np.column_stack([X, X[:,2]**2]))
            n, p = X_mod_quad.shape
             



            # --- Non-centered parameterization for coefficient matrix A ---
            # Instead of directly sampling A ~ Laplace(0,1), we use:
            #    A = sqrt(u_A/2) * A_tilde,  with u_A ~ Exponential(1) and A_tilde ~ Normal(0,1)
            #b = pm.HalfNormal('b', sigma=0.3) 
            #u_A = pm.Exponential('u_A', lam=3.0, shape=(p, self.rank))
            #uA_log = pm.Normal("uA_log", mu=0, sigma=1, shape=(p, self.rank))
            uA_log = pm.Normal("uA_log", mu=1, sigma=0.5, shape=(1, self.rank))
            u_A = pm.Deterministic("u_A", pt.exp(uA_log))
            A_tilde = pm.Normal('A_tilde', mu=0, sigma=0.3, shape=(p, self.rank))
            A = pm.Deterministic('A', A_tilde * pt.sqrt(u_A / 2))
            #A = pm.Normal("A", mu=0, sigma=0.3, shape=(p, self.rank))
            # Latent representation: (n x rank)
            latent = pt.dot(X_mod_quad, A)
            
            # --- Non-centered parameterization for factor matrix B with normalization ---
            # Sample B_tilde ~ Normal to allow both +/-
            B_tilde = pm.Normal('B_tilde', mu=0, sigma=0.3, shape=(q, self.rank))

            # Normalize each column to unit L2 norm
            epsilon = 1e-6
            B_normalized = B_tilde / (pt.sqrt(pt.sum(B_tilde**2, axis=0, keepdims=True)) + epsilon)

            # Use as deterministic variable
            B = pm.Deterministic('B', B_normalized)
            
    
            # Likelihood for responses.
            sigma = pm.HalfCauchy('sigma', beta=2)
            Y_obs = pm.Normal('Y_obs', mu=mu_Y, sigma=sigma, observed=Y)
        print(X_shared)
            
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
            prior_pred = pm.sample_prior_predictive(samples=500)
            self.trace = pm.sample(
            draws=1000, 
            tune=2000, 
            target_accept=0.99,
            chains=2, 
            cores=4,
            return_inferencedata=True
        )
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Access the prior predictive samples
        mu_samples = prior_pred.prior["mu_Y"]  # shape: (500, n, q)
        print(mu_samples.shape)

        # Pick a single output dimension (e.g., first target dimension)
        target_dim = 0
        n_samples_to_plot = min(20, mu_samples.shape[0])  # don't exceed available samples

        plt.figure(figsize=(10, 6))
        for i in range(n_samples_to_plot):
            y_vals = mu_samples[i, :, target_dim]  # shape: (n_obs,)
            x_vals = np.arange(y_vals.shape[0])        # shape: (n_obs,)
            plt.plot(x_vals, y_vals, alpha=0.3)

        plt.title(f"Prior Predictive Samples of mu_Y (dim {target_dim})")
        plt.xlabel("Observation Index")
        plt.ylabel("Predicted Value")
        plt.tight_layout()
        plt.show()

        return self.trace

    @staticmethod
    def transform_quad(X):
        # Example transformation: add a quadratic term for column 2.
        # (Adjust this if you need to transform more columns.)
        return np.column_stack([X, X[:, 2]**2])

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
            # Update shared data for both X_shared (if used for ordinal transforms) 
            # and the new quadratic features stored in X_mod_quad.
            pm.set_data({
                'X_shared': X_new,  
                'X_mod_quad': self.transform_quad(X_new)
            })
            
            if posterior_predictive:
                # Sample from the posterior predictive
                ppc = pm.sample_posterior_predictive(self.trace)
                print("Keys in posterior predictive:", ppc.keys())  # Debug step
                az.plot_ppc(ppc)
                plt.show()
                az.plot_trace(self.trace)
                plt.show()
                az.plot_energy(self.trace)
                az.summary(self.trace, var_names=["sigma"])
                plt.show()
                az.plot_pair(self.trace, var_names=["A"])
                plt.show()
                return ppc.posterior_predictive["Y_obs"].values
            else:
                # Compute predictions using the posterior means.
                A_mean = self.trace.posterior['A'].mean(dim=("chain", "draw"))
                B_mean = self.trace.posterior['B'].mean(dim=("chain", "draw"))
                
                # Start with transformed X_new (quadratic version).
                X_new_mod = self.transform_quad(X_new).copy()
                
                # Process ordinal columns if any.
                for j in self.ordinal_info:
                    D = self.ordinal_info[j]
                    var_name = self.zeta_names[j]
                    zeta_samples = self.trace.posterior[var_name]
                    zeta_mean = zeta_samples.mean(dim=("chain", "draw"))
                    cumsum_zeta = np.cumsum(zeta_mean, axis=-1)
                    X_new_mod[:, j] = np.array([cumsum_zeta[int(x) - 1] if x > 0 else 0 for x in X_new[:, j]])
                
                # Compute latent representation and predicted mean.
                latent_mean = X_new_mod @ A_mean.values
                mu_Y_mean = latent_mean @ B_mean.values.T
                
                # Optionally compute R² if provided in kwargs.
                if "Y_true" in kwargs and "Y_all" in kwargs:
                    r2 = self.compute_r2(kwargs["Y_true"], mu_Y_mean, kwargs["Y_all"])
                    print(f"R²: {r2:.3f}")
                
                return mu_Y_mean
    def compute_r2(self, Y_true, Y_pred, Y_all):
        """
        Compute R^2 = 1 - MSE / Var(Y_all)
        Y_true: Ground truth for test set
        Y_pred: Predicted mean (same shape)
        Y_all : Full dataset targets (used for variance normalization)
        """
        mse = np.mean((Y_true - Y_pred) ** 2)
        var_total = np.var(Y_all)
        return 1 - mse / var_total

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
    # Set file paths
    x_path = "../../data/final/x_final_scaled.csv"  # ⬅️ Replace this with your actual file path
    y_path = "../../data/final/y_final.csv"  # ⬅️ Replace this with your actual file path

    # Load data
    X, Y = load_data(x_path, y_path)

    # TODO: Define ordinal columns and their max levels, e.g.:
    # If column 2 is ordinal with values from 0 to 5:
    #ordinal_col = 1
    #X[:, ordinal_col] = np.minimum(X[:, ordinal_col], 6).astype(int)
    #ordinal_info = {ordinal_col: 6}  # D = 7 levels: 0 to 7
    # Instantiate the RRR model
    #model = RRR(rank=1, noise=0.4, ordinal_info=ordinal_info, mu_B=0, sigma_B=1)
    model = RRR(rank=2, noise=0.5, ordinal_info=None, mu_B=0, sigma_B=1)


    # Train the model
    trace = model.train(X, Y, draws=2000, tune=1500, target_accept=0.95)

    # Print the numerical summary
    print(az.summary(trace, var_names=["A", "B"], round_to=4, stat_focus="mean").to_string())

    # Plot the traces for key parameters
    az.plot_trace(trace, var_names=["A", "B"])
    plt.show()

    # Plot the posterior densities
    az.plot_posterior(trace, var_names=["A"])
    plt.show()

    # Posterior predictive check: this plots observed vs. simulated data
    plt.show()

    # Posterior predictive sampling
    ppc_samples = model.predict(X, posterior_predictive=True, samples=100)
    print("Posterior predictive sample shape:", ppc_samples.shape)

    # Posterior mean predictions
    mean_prediction = model.predict(X, posterior_predictive=False)
    print("Mean prediction shape:", mean_prediction.shape)
    print("Last 5 mean predictions:\n", mean_prediction[-5:])
    print("Last 5 Response Variables:\n", Y[-5:])
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Calculate MAE
    mae = mean_absolute_error(Y, mean_prediction)
    mse = mean_squared_error(Y, mean_prediction)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")



if __name__ == "__main__":
    main()
