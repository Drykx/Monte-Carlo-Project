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
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

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

def RM_Asian_mean(n, N, rho, K, S0, T, r, I_market, m, sigma_0):
    """This function returns the sequence sigma_n and J_n for the option price to equal to market price using a variant of the Robbins-Monro algorithm
        after n iterations
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
    m : int
        Number of time steps GBM
    sigma_0 : float
        Initial guess of the volatility
    Returns
    -------
    sigma_estim : np.ndarray
        Estimated volatility at each iteration  (n_features, )
    mean_it : np.ndarray
        Estimated valued of J at each iteration  (n_features, )
    """

    alpha_0 = 2/(K+S0)
    tol = 1 

    sigma_estim = np.empty((n,))
    mean_it = np.empty((n,))

    sigma_estim[0] = sigma_0
    
    simulations = Simulate_Stock_Price(S0, sigma_0 , r, T, m, N)
    avg_stock_prices = np.mean(simulations, axis=1)
    Z = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0) - I_market
    mean_it[0] = np.mean(Z)


    for i in range(1, n):
        alpha_n = alpha_0 / i ** rho
        sigma_cur = sigma_estim[i - 1]

        # Calculate Jhat
        simulations = Simulate_Stock_Price(S0, sigma_cur, r, T, m, N)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(payoffs) - I_market
        
        sigma_estim[i] = sigma_cur - alpha_n * Jhat
        mean_it[i] = Jhat


    return sigma_estim,mean_it

def Sign_changing(K, S0, T, r, I_market, m, sigma_0,Max_iteration = 10**5):
    """ Aggressive MC algorithm to have a better guess for sigma_0"""
    alpha_0 = 2/(K+S0)
    rho_0 = 0.5
    N = 1000

    sigma = sigma_0
    discount_factor = np.exp(-r*T)

    #Initalize Jcurr
    simulations = Simulate_Stock_Price(S0, sigma, r, T, m, N)
    avg_stock_prices = np.mean(simulations, axis=1)
    payoffs = discount_factor * np.maximum(K - avg_stock_prices, 0)
    Jhat_curr = np.mean(payoffs) - I_market 
    Jhat = Jhat_curr
    i = 1

    while(Jhat * Jhat_curr >= 0):

        Jhat_curr = Jhat

        alpha_n = alpha_0 / i ** rho_0

        # Calculate Jhat
        simulations = Simulate_Stock_Price(S0, sigma, r, T, m, N)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = discount_factor * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(payoffs) - I_market

        sigma -= alpha_n * Jhat

        if i == Max_iteration:
            return sigma,i
        
        i += 1

    return sigma,i

def function_r_tilde(K, S0, T, r, I_market, m, sigma_0,M=10):
    """ Initial guess of sigma_* with sign_changing + MC of 10 values obtained + output 0-9-quantile"""
    H = np.zeros(shape=(M,2))

    for k in range(M):
        sigma, i = Sign_changing(K, S0, T, r, I_market, m, sigma_0)
        H[k,0] = sigma
        H[k,1] = i

    sigma_suboptimal = np.mean(H[:,0])
    # Mean of the path
    mu_path = (1+np.exp(r*T) )/2 * S0

    def intersection(x):
        return st.lognorm.ppf(0.90, scale = np.exp(np.log(mu_path) + (x-0.5*sigma_suboptimal**2) * T), s = np.sqrt( sigma_suboptimal ** 2 * T))-K

    r_tilde = bisect(lambda sigma: intersection(sigma), -10, 10)

    return r_tilde

# Calculate likelihood ratio
def w(S_T, S_0, sigma, r, r_tilde, T):
    return (S_T/S_0)**((r - r_tilde)/sigma**2) * np.exp((r+r_tilde-sigma**2)*(-r+r_tilde)*T/(2*sigma**2))


def RM_Asian(n, N, rho, K, S0, T, r, I_market, m, sigma_0):
    """This function returns the sequence of sigmas for the option price to equal the market price using the Robbin-Monro algorithm
        after n iterations
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
    m : int
        Number of time steps GBM
    sigma_0 : float
        Initial guess of the volatility
    Returns
    -------
    sigma_estim : np.ndarray
        Estimated volatility at each iteration  (n_features, )
    """
    alpha_0 = 2/(K+S0)

    sigma_estim = np.empty((n,))
    sigma_estim[0] = sigma_0
    
    for i in range(1, n):
        alpha_n = alpha_0 / i ** rho
        sigma_cur = sigma_estim[i - 1]

        # Calculate Jhat
        simulations = Simulate_Stock_Price(S0, sigma_cur, r, T, m, N)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(payoffs) - I_market
        
        sigma_estim[i] = sigma_cur - alpha_n * Jhat

    return sigma_estim

def RM_Asian_with_Stopping(N, rho, K, S0, T, r, I_market, m,sigma_0,Max_iteration = 10**6,window_size=1000,tol1=10**(-3),tol2=10**(-2)):
    """This function returns the sequence sigma_n and J_n for the option price to equal to market price using a variant of the Robbins-Monro algorithm
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
    m : int
        Number of time steps GBM
    sigma_0 : float
        Initial guess of the volatility
    Returns
    -------
    sigma_estim : np.ndarray
        Estimated volatility at each iteration  (n_features, )
    mean_it : np.ndarray
        Estimated valued of J at each iteration  (n_features, )
    """

    if rho > 0.5:
        sigma_0,i = Sign_changing(K, S0, T, r, I_market, m, sigma_0)
    else:
        i=1

    alpha_0 = 2/(K+S0)
    sigma_cur = sigma_0
    sigma_estim = deque([sigma_0])
    J_estim = deque([])
        
    while i < Max_iteration:
        alpha_n = alpha_0 / i ** rho

        # Calculate Jhat
        simulations = Simulate_Stock_Price(S0, sigma_cur, r, T, m, N)
        avg_stock_prices = np.mean(simulations, axis=1)
        Z = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0) - I_market
        Jhat, est_std = np.mean(Z), np.std(Z)
        sigma_cur = sigma_cur - alpha_n * Jhat
        sigma_estim.append(sigma_cur)
        J_estim.append(Jhat)

        # Stopping Criterion
        if i == window_size:
            mean_value = np.mean(J_estim)
        if i > window_size:
            removed_value = J_estim.popleft()
            mean_value = mean_value + (Jhat - removed_value) / window_size
            if abs(mean_value) < tol1 and abs(Jhat) < tol2:
                break
        i += 1
        
    print(f'Stopped at iteration n = {i}, where sigma_n = {sigma_cur}')
    return sigma_estim,i

def RM_Asian_with_IS_and_Stopping(N, rho, K, S0, T, r, r_tilde, I_market, m,sigma_0, Max_iteration = 10**6,window_size=1000,tol1=10**(-3),tol2=10**(-2)):
    if rho > 0.5:
        sigma_0,i = Sign_changing(K, S0, T, r, I_market, m, sigma_0)
    else:
        i=1

    alpha_0 = 2/(K+S0)
    sigma_cur = sigma_0
    sigma_estim = deque([sigma_0])
    J_estim = deque([])

    while i < Max_iteration:
        alpha_n = alpha_0 / i ** rho

        # Calculate Jhat
        simulations = Simulate_Stock_Price(S0, sigma_cur, r_tilde, T, m, N)
        likelihood_ratios = w(simulations[:, -1], S0, sigma_cur, r, r_tilde, T)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(np.multiply(payoffs, likelihood_ratios)) - I_market
        sigma_cur -= alpha_n * Jhat
        sigma_estim.append(sigma_cur)
        J_estim.append(Jhat)

        # Stopping Criterion
        if i == window_size:
            mean_value = np.mean(J_estim)
        if i > window_size:
            removed_value = J_estim.popleft()
            mean_value = mean_value + (Jhat - removed_value) / window_size
            if abs(mean_value) < tol1 and abs(Jhat) < tol2:
                break
        if i == Max_iteration:
            return sigma_estim,i       
        i += 1

    print(f'Stopped at iteration n = {i}, where sigma_n = {sigma_cur}')
    return sigma_estim, i

def RM_Asian_with_IS(n, N, rho, K, S0, T, r, r_tilde, I_market, m, sigma_0):
    if rho > 0.5:
        sigma_0,i = Sign_changing(K, S0, T, r, I_market, m, sigma_0)
    else:
        i=1

    alpha_0 = 2/(K+S0)
    sigma_estim = np.empty((n,))
    sigma_cur = sigma_0
    
    while i < n:
        alpha_n = alpha_0 / i ** rho

        # Calculate Jhat with IS
        simulations = Simulate_Stock_Price(S0, sigma_cur, r_tilde, T, m, N)
        likelihood_ratios = w(simulations[:, -1], S0, sigma_cur, r, r_tilde, T)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(np.multiply(payoffs, likelihood_ratios)) - I_market

        sigma_cur -= alpha_n * Jhat
        sigma_estim[i] = sigma_cur
        i += 1

    return sigma_estim

def get_r_opt(sigmas, S0, K, r, T, m, N):
    def g(eta, sigma, simulations):
        likelihood_ratios = w(simulations[:, -1], S0, sigma, r, eta, T)
        
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs_squared = (np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0))**2
        variance = np.mean(np.multiply(payoffs_squared, likelihood_ratios))
        return variance
    
    result = np.empty((len(sigmas),))
    for i, sigma in enumerate(sigmas):
        simulations = Simulate_Stock_Price(S0, sigma, r, T, m, N)
        result[i] = minimize_scalar(g, args=(sigma,simulations)).x

    return interp1d(sigmas, result, kind='cubic')

class VarianceAnalyzer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
            return self.calculate_var()
        return 0

    def calculate_var(self):
        if len(self.values) < 2:
            return None  # Variance is undefined with less than two values
        return np.var(self.values, ddof=1)  # ddof=1 for unbiased variance estimation

def RM_Asian_with_IS_opt_and_stopping(N, rho, K, S0, T, r, I_market, m, sigma_0, r_opt, Max_iteration = 10**6,window_size=1000,tol1=10**(-3),tol2=10**(-2)):
    if rho > 0.5:
        sigma_0,i = Sign_changing(K, S0, T, r, I_market, m, sigma_0)
    else:
        i=1
        
    alpha_0 = 2/(K+S0)

    sigma_cur = sigma_0
    sigma_estim = deque([sigma_0])
    J_estim = deque([])
    while i < Max_iteration:
        alpha_n = alpha_0 / i ** rho

        # Calculate Jhat
        r_tilde = r_opt(sigma_cur)
        simulations = Simulate_Stock_Price(S0, sigma_cur, r_tilde, T, m, N)
        likelihood_ratios = w(simulations[:, -1], S0, sigma_cur, r, r_tilde, T)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(np.multiply(payoffs, likelihood_ratios)) - I_market
        sigma_cur = sigma_cur - alpha_n * Jhat
        sigma_estim.append(sigma_cur)
        J_estim.append(Jhat)

        # Stopping Criterion
        if i == window_size:
            mean_value = np.mean(J_estim)
        if i > window_size:
            removed_value = J_estim.popleft()
            mean_value = mean_value + (Jhat - removed_value) / window_size
            if abs(mean_value) < tol1 and abs(Jhat) < tol2:
                break
        i += 1
    print(f'Stopped at iteration n = {i}, where sigma_n = {sigma_cur}')
    return sigma_estim, i

def RM_Asian_with_IS_opt(n, N, rho, K, S0, T, r, I_market, m,r_opt):
    if rho > 0.5:
        sigma_0,i = Sign_changing(K, S0, T, r, I_market, m, sigma_0)
    else:
        i=1

    sigma_0 = 1
    alpha_0 = 2/(K+S0)

    sigma_estim = np.empty((n,))
    sigma_cur = sigma_0
    
    while i < n:
        alpha_n = alpha_0 / i ** rho

        # Calculate Jhat with IS
        r_tilde = r_opt(sigma_cur)
        simulations = Simulate_Stock_Price(S0, sigma_cur, r_tilde, T, m, N)
        likelihood_ratios = w(simulations[:, -1], S0, sigma_cur, r, r_tilde, T)
        avg_stock_prices = np.mean(simulations, axis=1)
        payoffs = np.exp(-r*T) * np.maximum(K - avg_stock_prices, 0)
        Jhat = np.mean(np.multiply(payoffs, likelihood_ratios)) - I_market

        sigma_cur = sigma_cur - alpha_n * Jhat
        sigma_estim[i] = sigma_cur
        i +=1

    return sigma_estim
