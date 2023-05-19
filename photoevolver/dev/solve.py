
import numpy as np
from scipy import stats

def solve_structure_errors(
        params : dict, # = {'mass': mean, ...}
        errors : dict, # = {'mass': error or (lower,upper), ...}

        core_model     : callable,
        envelope_model : callable,
        model_kwargs   : dict,
        samples :int = 1000 
    ) -> dict:
    """
    Propagates errors via the Monte Carlo sampling method.

    Parameters
    ----------
    params  : dict, Mean value of the planet parameters.
            Must contain 'mass' and 'radius'.
    errors  : dict, The errors associate with each planet parameter.
            Must have the same keys as `params`. If the key is not
            defined, it is assumed to have an error of 0.
            Errors can be given as a single value (symmetric),
                `errors = { 'mass': 0.1 }`
            or a list of two values (lower, upper) (asymmetric),
                `errors = { 'mass': [0.1, 0.15] }`
    core_model
            : callable, Model that returns the core radius given its mass.
    envelope_model
            : callable, Model that returns the envelope thickness given
            the envelope mass fraction.
    model_kwargs
            : dict, keyword arguments for the models.
    samples : int, number of samples for the Monte Carlo method.
    """

    def sym(mean, sigma, num):
        return np.random.normal(mean, sigma, num)

    def asym(mean, lsigma, rsigma, num):
        # alpha (rsigma - lsigma) / 2
        lpeak = 1/(np.sqrt(2*np.pi)/lsigma)
        rpeak = 1/(np.sqrt(2*np.pi)/lsigma)

        lhs = np.random.normal(mean, lsigma, num * int(rpeak/lpeak))
        rhs = np.random.normal(mean, rsigma, num)
        
        # lhs[lhs > mean] = rhs[lhs > mean] 
        # return lhs
        return np.concatenate([lhs, rhs])
        

    # Default error is 0
    for k,v in params.items():
        errors.setdefault(k, 0.0)

    # Generate distributions
    dist = {}
    for k,mean in params.items():
        err = errors[k]
        if isinstance(err, float): # Gaussian
            dist[k] = sym(mean, err, samples)
        else: # Asymmetric
            dist[k] = asym(mean, err[0], err[1], samples)

    return dist
     
    