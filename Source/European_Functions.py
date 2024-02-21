import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import pandas as pd
from scipy.optimize import bisect
from scipy.optimize import root_scalar
from scipy.stats import linregress
import seaborn as sns
from collections import deque


def Simulate_Stock_Price(S_0,sigma, r, T, m,N):
    """This function returns N GBM, each evaluated at time T/m,...,T with a starting value of S_0

    Parameters
    ----------
    S_0 : float
        Initial value of the geometric Brownian motion
    sigma : float
        Volatility of the geometric Brownian motion
    r : float
        Interest rate of the geometric Brownian motion
    T : float
        Time horizon of the geometric Brownian motion 
    m : int
        Number of time steps
    N : int
        Number of GBM to generate

    Returns
    -------
    S : np.ndarray
        N GBM evaluated at time T/m,...,T (N_features, m_features)
    """
    simulations = np.ndarray((N, m))
    expXt = st.lognorm.rvs(s=np.sqrt(sigma ** 2 * T / m), loc=0, scale=np.exp((r - 0.5 * sigma ** 2) * T / m), size=m*N)
    expXt_reshaped = expXt.reshape((N, m), order='F')  # Reshape the array to m rows and N columns
    simulations = np.cumprod(np.concatenate([np.ones((N, 1)) * S_0, expXt_reshaped], axis=1), axis=1)
    return simulations[:, 1:]

def I(sigma, S_0 = 100 , K = 120, r = 0.05, T = 0.2):
    """Closed form solution for the option price of an european put option
    Parameters
    ----------
    sigma : float
        volatility of the geometric brownian motion
    S_0 : float
        Initial value of the geometric brownian motion
    K : float
        strike price of the put option
    r : float
        interest rate of the put option
    sigma : float
        volatility of the geometric brownian motion
    T : float
        time horizon of the put option

    Returns
    -------
    I : float
        Option price"""
    
    w = (np.log(K/S_0) - (r-0.5*sigma**2)*T) / ( sigma * np.sqrt(T) )
    I = np.exp(-r*T) * K * st.norm.cdf(w) - S_0 * st.norm.cdf(w - sigma*np.sqrt(T))
    return I

def RM_European(n, N, rho, K, S0, T, r, I_market,sigma_0 = 1):
    """This function returns the estimated volatility for the option price to equal to market price using the Robbins-Monro algorithm
    Parameters
    ----------
    n : int
        Number of iterations
    N : int    
        Number of MonteCarlo simulation to approximate J
    rho : float
        Agressivness of the algorithm
    K : float   
        Strike price of the put option
    S0 : float
        Initial value of the geometric Brownian motion
    T : float   
        Time horizon of the geometric Brownian motion
    r : float
        Interest rate of the geometric Brownian motion
    I_market : float    
        Market price of the option
    sigma_0 : float
        Initial guess of the volatility

    Returns
    -------
    sigma_estim : np.ndarray
        Estimated volatility at each iteration  (n_features, )
    """
    alpha_0 = 2/(K+S0) # Fixed constant in the algorithm

    sigma_estim = np.empty((n,))
    sigma_estim[0] = sigma_0
    
    for i in range(1, n):
        alpha_n = alpha_0 / i ** rho
        sigma_cur = sigma_estim[i - 1]

        # Compute Jhat
        stock_prices = Simulate_Stock_Price(S0, sigma_cur, r, T, m=1, N=N)
        payoffs = np.exp(-r*T) * np.maximum(K - stock_prices, 0)
        Jhat = np.mean(payoffs) - I_market

        sigma_estim[i] = sigma_cur - alpha_n * Jhat

    return sigma_estim