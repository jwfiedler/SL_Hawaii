#%%
import spotpy
import numpy as np
from spotpy.parameter import Uniform
from scipy.optimize import approx_fprime
from pathlib import Path
import multiprocessing
import random
import sys

def initialize_random(seed):
    """
    Initialize the random seed for each multiprocessing process.
    """
    np.random.seed(seed)

seed = 42
np.random.seed(seed)
random.seed(seed)


def GEV_S_T_Cv_Nodal(icromo, w, T_vector, Y, serieCV):
    """
    Translated version of the Fortran functn function.
    
    Parameters:
    w         : Array of optimization parameters.
    T_vector  : Time vector (e.g., years or time points).
    Y         : Observed data (e.g., sea level).
    serieCV   : Covariate data (e.g., climate indices).
    icromo    : Array of binary flags for seasonal/trend components.

    Returns:
    amut      : Location parameters (array).
    apsit     : Scale parameters (array).
    xit       : Shape parameters (array).
    """
    #number of data points
    ndata = len(T_vector)
    
    # Constants
    api = np.pi # pi
    ak = 2 * api # 2pi
    ak18 = 2 * api / 18.61  # 18.61-year nodal cycle

    # Extract primary GEV parameters from x
    b0 = w['b0']  # Location parameter (base)
    a0 = w['a0']  # Scale parameter (base)
    g0 = w['g0']  # Shape parameter
    k = 3

    # Initialize seasonal/trend/covariate parameters
    b1, b2, b3, b4, b5, b6 = 0, 0, 0, 0, 0, 0
    bLT, bCI, aCI, bN1, bN2  = 0, 0, 0, 0 ,0

    # Annual cycle
    if icromo[0] == 1:
        b1 = w['b1']
        b2 = w['b2']
    
    # Semiannual cycle
    if icromo[1] == 1:
        b3 = w['b3']
        b4 = w['b4']
    
    # Higher frequency cycle
    if icromo[2] == 1:
        b5 = w['b5']
        b6 = w['b6']
    
    # Long-term trend
    if icromo[3] == 1:
        bLT = w['bLT']
    
    # Covariate effect on location parameter
    if icromo[4] == 1:
        bCI = w['bCI']
    
    # Covariate effect on scale parameter
    if icromo[5] == 1:
        aCI = w['cCI']

    # Nodal cycle
    if icromo[6] == 1:
        bN1 = w['bN1']
        bN2 = w['bN2']

    # Initialize arrays for location, scale, and shape parameters
    amut = np.zeros(ndata)
    apsit = np.zeros(ndata)
    xit = np.full(ndata, g0)  # Shape parameter constant across all data points

    T = T_vector
    
    amut = (b0 * np.exp(bLT * T) +
        b1 * np.cos(ak * T) + b2 * np.sin(ak * T) +
        b3 * np.cos(2 * ak * T) + b4 * np.sin(2 * ak * T) +
        b5 * np.cos(4 * ak * T) + b6 * np.sin(4 * ak * T) +
        bN1 * np.cos(ak18 * T) + bN2 * np.sin(ak18 * T) +
        bCI * serieCV)

    eps = 1e-8
    apsit = np.maximum(a0 + aCI * serieCV, eps) # prevent scale parameter from being zero or negative

    # Factor used in the log-likelihood calculation
    factor = 1.0 + xit * (Y - amut) / apsit

    # Check for invalid values (factor <= 0 or scale <= 0)
    invalid = np.any(factor <= 0) or np.any(apsit <= 0)
    if invalid:
        return np.nan, np.nan, np.nan
    
    return amut, apsit, xit


def numerical_hessian(params, objective_function, epsilon=1e-4):
        """
        Computes the Hessian matrix (second-order partial derivatives) numerically.

        Parameters:
        - params: Parameter values (numpy array)
        - objective_function: The objective function to optimize
        - epsilon: Small perturbation for finite differences

        Returns:
        - Hessian matrix
        """
        n_params = len(params)
        hessian = np.zeros((n_params, n_params))

        # Compute the gradient at the best parameter values
        gradient = approx_fprime(params, objective_function, epsilon)

        # Loop to compute the second derivatives using finite differences
        for i in range(n_params):
            perturbed_params = np.copy(params)
            perturbed_params[i] += epsilon
            perturbed_gradient = approx_fprime(perturbed_params, objective_function, epsilon)

            # Compute second derivative
            hessian[:, i] = (perturbed_gradient - gradient) / epsilon

        return hessian

def get_log_likelihood(Y, amut, apsit, xit):
    """
    Log-likelihood calculation, following the Fortran implementation.
    
    Parameters:
    - Y: Observed data (e.g., sea level)
    - amut: Location parameters (array)
    - apsit: Scale parameters (array)
    - xit: Shape parameters (array)
    
    Returns:
    - log_likelihood: Calculated log-likelihood value
    """
    ndata = len(Y)
    al = 0  # Initialize log-likelihood accumulator

    inv_xit = 1.0 / xit

    # Case when g0 is non-zero (away from zero)
    if abs(np.mean(xit)) > 0.001:
        factor = 1.0 + xit * (Y - amut) / apsit
        
        if np.any(factor <= 0):
            return -1e6  # Penalize invalid values

        log_likelihood = (
            - np.sum(np.log(apsit))  # Sum of log(apsit)
            - np.sum((1.0 + inv_xit) * np.log(factor))  # Sum for the first part
            - np.sum(factor ** (-inv_xit))  # Sum for the second part
        )
    
    # Case when shape is approximately zero (Gumbel distribution)
    else:
        log_likelihood = (
            - np.sum(np.log(apsit))  # Sum of log(apsit)
            - np.sum((Y - amut) / apsit)  # Sum for the linear term
            - np.sum(np.exp(-(Y - amut) / apsit))  # Sum for the exponential term
        )

    return log_likelihood


class gev_spot_setup:

    def __init__(self, T_vector, Y, serieCV, icromo):
        """
        Initialize the GEV model setup with time vector, observed data, covariates, and icromo flags.
        """
        self.T_vector = T_vector         # Time vector (e.g., years)
        self.Y = Y                       # Observed data (e.g., sea level)
        self.serieCV = serieCV           # Covariate data (e.g., climate indices)
        self.icromo = icromo             # Binary flags to activate different model components
        
    def parameters(self):
        """
        Dynamically define the parameters based on `icromo` flags.
        """
        param_list = [
            Uniform('b0', low=0.01, high=7.5, optguess=0.001),
            Uniform('a0', low=0.001, high=2.5, optguess=0.001),
            Uniform('g0', low=-0.35, high=0.15, optguess=-0.001)
        ]

        if self.icromo[0] == 1:
            param_list.extend([
                Uniform('b1', low=-2.001, high=2.001, optguess=0.001),
                Uniform('b2', low=-2.001, high=2.001, optguess=0.001)
            ])
        if self.icromo[1] == 1:
            param_list.extend([
                Uniform('b3', low=-0.5, high=0.5, optguess=0.001),
                Uniform('b4', low=-0.5, high=0.5, optguess=0.001)
            ])
        if self.icromo[2] == 1:
            param_list.extend([
                Uniform('b5', low=-0.25, high=0.25, optguess=0.001),
                Uniform('b6', low=-0.25, high=0.25, optguess=0.001)
            ])
        if self.icromo[3] == 1:
            param_list.append(Uniform('bLT', low=-0.3, high=0.3, optguess=0.001))
        if self.icromo[4] == 1:
            param_list.append(Uniform('bCI', low=-0.5, high=0.5, optguess=0.001))
        if self.icromo[5] == 1:
            param_list.append(Uniform('cCI', low=-0.2, high=0.2, optguess=0.001))
        if self.icromo[6] == 1:
            param_list.extend([
                Uniform('bN1', low=-0.2, high=0.2, optguess=0.001),
                Uniform('bN2', low=-0.2, high=0.2, optguess=0.001)
            ])
        return spotpy.parameter.generate(param_list)


    def simulation(self, w):
        """
        The simulation function runs the GEV model given a set of parameters.
        SPOTPY passes the sampled parameters `w` into this function. 
        Ensure that parameters corresponding to inactive components (where icromo == 0) are set to zero.
        """
        ndata = len(self.T_vector)

        # set up parameters based on icromo flags
        active_params = self.parameters()
        param_values = {}
        k=0

        # check if w is a dict or an array
        if isinstance(w, dict):
            for param in active_params:
                param_values[param.name] = w[param.name]
        else:
            # Required base parameters
            param_values['b0'] = w[0]
            param_values['a0'] = w[1]
            param_values['g0'] = w[2]
            k += 3

            if self.icromo[0] == 1:
                param_values['b1'] = w[k]
                param_values['b2'] = w[k+1]
                k += 2

            if self.icromo[1] == 1:
                param_values['b3'] = w[k]
                param_values['b4'] = w[k+1]
                k += 2

            if self.icromo[2] == 1:
                param_values['b5'] = w[k]
                param_values['b6'] = w[k+1]
                k += 2

            if self.icromo[3] == 1:
                param_values['bLT'] = w[k]
                k += 1

            if self.icromo[4] == 1:
                param_values['bCI'] = w[k]
                k += 1

            if self.icromo[5] == 1:
                param_values['cCI'] = w[k]
                k += 1

            if self.icromo[6] == 1:
                param_values['bN1'] = w[k]
                param_values['bN2'] = w[k+1]
                k += 2


        amut, apsit, xit =  GEV_S_T_Cv_Nodal(self.icromo, param_values, self.T_vector, self.Y, self.serieCV)
        
        if np.any(np.isnan(amut)) or np.any(np.isnan(apsit)) or np.any(np.isnan(xit)) or np.any(np.isinf(amut)) or np.any(np.isinf(apsit)) or np.any(np.isinf(xit)):
            return np.full(ndata, np.nan), np.full(ndata,np.nan), np.full(ndata,np.nan)  # Return a large value for invalid cases
        
        return amut, apsit, xit

    def evaluation(self):
        """
        This function returns the observed data, which SPOTPY uses for comparison.
        """
        return self.Y

    def objectivefunction(self, simulation, evaluation):
        """
        Compute the log-likelihood using the GEV model simulation and observed data.
        """
        # Check for invalid values in the simulation
        amut, apsit, xit = simulation
        if np.any(np.isnan(amut)) or np.any(np.isnan(apsit)) or np.any(np.isnan(xit)) or np.any(np.isinf(amut)) or np.any(np.isinf(apsit)) or np.any(np.isinf(xit)):
            return 1e3 # Return a large value for invalid cases

        # Check for very large values
        if np.any(amut>=1e6) or np.any(apsit>=1e6) or np.any(xit>=1e6):
            return 1e3

        # Compute the log-likelihood
        log_likelihood = get_log_likelihood(evaluation, amut, apsit, xit)
        # log_likelihood = np.sum(gev.logpdf(evaluation, c=xit, loc=amut, scale=apsit))

        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            return 1e3

        # Return the negative log-likelihood as the objective function to maximize
        return -log_likelihood

#%%
def run_GEVt_model(icromo, run_dir, parallel= True):
    """
    Run the GEV model using the SCE-UA optimization algorithm.
    icromo: Array of binary flags for seasonal/trend components.
    run_dir: Directory to temporarily read and save the results.
    """
    T = np.loadtxt(Path(run_dir) / 'T.txt')
    Y = np.loadtxt(Path(run_dir) / 'Y.txt')
    CI = np.loadtxt(Path(run_dir) / 'CI.txt')


    # attempt to set random seed for reproducibility
    seed = 955
    np.random.seed(seed)
    random.seed(seed)

    # Make icromo 7 components long
    if len(icromo) < 7:
        icromo = np.concatenate([icromo, np.zeros(7 - len(icromo))])

    setup = gev_spot_setup(T, Y, CI, icromo)
    if parallel:
        np.random.seed(seed)
        sampler = spotpy.algorithms.sceua(setup, dbname='GEVsceua', dbformat='ram', parallel='mpi', save_sim=False )
    else:
        np.random.seed(seed)
        sampler = spotpy.algorithms.sceua(setup, dbname='GEVsceua', dbformat='ram', parallel='seq',save_sim=False)
  
    # Select number of maximum repetitions and complexes
    n = int(len(setup.parameters()))
    rep = int( -2000 + 1500 * n ) #number of iterations
    ngs = int (np.ceil(n / 4)) #number of complexes

    # Run the sampler
    sampler.sample(rep, ngs=ngs, kstop=3, pcento=0.001, peps=1e-5)

    results = sampler.getdata()

    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    best_model_run = results[bestindex]
    params = [word for word in best_model_run.dtype.names if word.startswith("par")]
    best_param_values = list(best_model_run[params])
    param_names = [word[3:] for word in params]

    best_params = dict(zip(param_names, best_param_values))

    

    # Define the objective function to optimize
    def objective_function(params):
        """
        Objective function to minimize.
        """
        # Compute the negative log-likelihood (as defined in the setup)
        return setup.objectivefunction(setup.simulation(params), setup.evaluation())

    
    mio = numerical_hessian(best_param_values, objective_function)
    # save mio to file
    np.savetxt(Path(run_dir) / 'mio.txt', mio)

    # save param values to file, starting with bestobjf, then best_param_values
    with open(Path(run_dir) / 'best.txt', 'w') as f:
        f.write(str(-bestobjf))
        f.write('\n')
        for item in best_param_values:
            f.write("%s\n" % item)

    return -bestobjf, best_params


#%%
# icromo = [1,1,0,1] 
# run_dir = '/Users/juliafiedler/Documents/Repositories/SL_Hawaii/SL_Hawaii/notebooks/nonstationaryGEV'
# best_params = run_GEVt_model(icromo, run_dir)
# best_params


