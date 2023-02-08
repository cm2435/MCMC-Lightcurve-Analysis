import pandas as pd
import numpy as np 
from typing import Optional, List

import pymc3 as pm
import aesara_theano_fallback.tensor as tt

import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
import exoplanet as xo 

def find_average_orbital_flux(luminosity, semimajor, eccentricity):
    #Convert all items to si 
    luminosity = luminosity * 3.846e26
    semimajor = semimajor * 1.495978707e11

    F = luminosity / ((4 * np.pi * semimajor**2) * (np.sqrt(1 - eccentricity**2)))
    return F
    

def fold_lightcurve(time, flux, error, period, verbose: bool = False):
    """
    Folds the lightcurve given a period.
    time: input time (same unit as period)
    flux: input flux
    error: input error
    period: period to be folded to, needs to same unit as time (i.e. days)
    returns: phase, folded flux, folded error
    """
    #Create a pandats dataframe from the 
    data = pd.DataFrame({'time': time, 'flux': flux, 'error': error})
    #create the phase 
    data['phase'] = data.apply(lambda x: ((x.time/ period) - np.floor(x.time / period)), axis=1)
    if verbose: 
        print(data.head(10))

    #Creates the out phase, flux and error
    phase_long = np.concatenate((data['phase'], data['phase'] + 1.0, data['phase'] + 2.0))
    flux_long = np.concatenate((flux, flux, flux))
    err_long = np.concatenate((error, error, error))
    
    return(data['time'], phase_long, flux_long, err_long)


def model_curve(x, d, transit_b, transit_e) -> float: 
    """
    Fit a qu
    """
    m = (16 * (1-d) / (transit_e - transit_b)**4) * (x - (transit_e+transit_b) / 2)**4 + d
    return m 

def chisquared_reduced(x, y, error, ymodel):

    chisquare = np.sum((y-ymodel)**2/error**2)
    reduced_chisquared = chisquare / (len(x) - 3 -1) # 3 degrees of freedom for the quartic fit hence -3 - 1 
    return reduced_chisquared