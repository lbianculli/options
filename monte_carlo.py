class BlackScholes:
    def __init__(self, s, k, sigma, t, price, r, q=0):  # if new IV works, can i change to 0?
        self.s = s  # have to set these as _fprime can take only one arg. is there a better way?
        self.k = k
        self.sigma = sigma
        self.t = t
        self.price = price
        self.r = r
        self.q = q
        
    def update(s, k, sigma, t, price, r, q=0):
        """ update params to test with new option values """
        self.s = s
        self.k = k
        self.sigma = sigma
        self.t = t
        self.price = price
        self.q = q
        
    def _get_norm_cdf(self, d1):  # is this the same thing as scipy function?
        return (1 / sqrt(2*pi)) * exp(-d1**2 / 2)
    
    def call(self, s, k, sigma, t, r, q=0):
        """ returns the call price of an option according to BS """
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        
        return norm.cdf(d1)*s*exp(-q*t) - norm.cdf(d2)*k*exp(-r*t)
        
    def put(self, s, k, sigma, t, r, q=0):
        """ returns the put  price of an option according to BS """
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        
        return  k*exp(-r*t)*norm.cdf(-d2) - s*exp(-q*t)*norm.cdf(-d1)
    
    def greeks(self, s, k, sigma, t, price, r, q, option="calls"):
        """ return dict of greeks for the option """
        greeks = {}
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        norm_cdf = self._get_norm_cdf(d1)
        T = 365.242199 #  T = days in a year -- calendar OR trading days. Why would it not be trading days?
        
        if option == "call":
            greeks["delta"] = exp(-q*t) * norm.cdf(d1)
            greeks["theta"] = (-(s*sigma*exp(-q*t) / 2*sqrt(t)*norm_cdf) - r*k*exp(-r*t)*norm.cdf(d2) + q*s*exp(-q*t)*norm.cdf(d1)) / T
            greeks["rho"] = (k*t*exp(-r*t)*norm.cdf(d2)) / 100
            
        else:
            greeks["delta"] = exp(-q*t) * (norm.cdf(d1) - 1)
            greeks["theta"] = (-(s*sigma*exp(-q*t) / 2*sqrt(t)*norm_cdf) + r*k*exp(-r*t)*norm.cdf(-d2) - q*s*exp(-q*t)*norm.cdf(-d1)) / T
            greeks["rho"] = -(k*t*exp(-r*t)*norm.cdf(-d2)) / 100
        
        greeks["gamma"] = exp(-q*t) / (s * sigma * sqrt(t)) * norm_cdf
        greeks["vega"] = (s*exp(-q*t)*sqrt(t) * norm_cdf) / 100
        
        return greeks
        
    def _get_d1(self, s, k, sigma, t, r, q=1e-10): 
        """ calculate d1 for BS formula """
        return (log(s/k)+(r - q + sigma**2/2)*t) / (sigma*sqrt(t))
        
    def _get_d2(self, d1, sigma, t): 
        """ calculate d2 for BS formula """
        return d1 - sigma * sqrt(t)

    
    def ivr(self):
        """" calculate IV rank of the option """
        raise NotImplementedError
        
    def ivp(self):
        """" calculate IV% of the option """
        raise NotImplementedError
        
        
    def ivx(self, window=30):
        """" calculate IV% of the option """
        raise NotImplementedError 
        
    def pop(self):
        """" calculate PoP of the option """
        raise NotImplementedError 
        
    def p50(self):
        """" calculate p50 of the option """
        raise NotImplementedError 
        
    def _get_rate(self, option="call"):
        """ get rf rate given treasury rate and time to expiration """
        raise NotImplementedError
    
    def get_implied_vol(self, s, k, t, price, r, q=0, max_iter=100, tol=1e-3, option="call"):
        """Guess the implied volatility."""
        
        # Set base case
        known_min = 0
        known_max = 10.0
        try:
            iv_guess = (
                sqrt(2 * pi / t) * (price / k)
            )
        except TypeError:
            print("TypeError in IV calculation. Returning NaN")
            return np.nan
            
        if option == "call":
            opt_val = self.call(s, k, iv_guess, t, r, q)
        if option == "put":
            opt_val = self.put(s, k, iv_guess, t, r, q)
            
        price_diff = opt_val - price

        # iterate until we can minimize difference between guess and actual price
        iterations = 0
        while abs(price_diff) > tol:
            if price_diff > 0:  # if our guess is higher than the actual
                known_max = iv_guess
                iv_guess = (known_min + known_max) / 2
            else:
                known_min = iv_guess
                iv_guess = (known_min + known_max) / 2

            if option == "call":
                opt_val = self.call(s, k, iv_guess, t, r, q)
            if option == "put":
                opt_val = self.put(s, k, iv_guess, t, r, q)
                
            price_diff = opt_val - price

            if iv_guess < 0.001:
                return 0

            iterations += 1
            if iterations > max_iter:
                print(f"Warning: Reached maximum number of iterations for "
                      + f"implied volatility guess for strike {k}. "
                      + f"Returning 0...")
                return 0

        return iv_guess
    
    
    def mc_price(self, s0, k, sigma, t, r, q, n_paths=10000, seed=None, method="euler"):  # how to incorporate q??
        """ simulate price of option using monte carlo """
        np.random.seed(seed)
        t0 = time()
        T = 1.0
        rf = 0.0156
        h = math.ceil(t * 21/30)  # steps should be business days till expiration. need to be int
        dt = T / h
        
        if method == "euler":
            # Simulating I paths with h time steps. The last piece is the stoachastic term
            # think of the left side - up until the sqrt(dt) term - as the params that specify the path and the noise term as adding a path with some noise.
            s = s0 * np.exp(np.cumsum((r - q - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * np.random.standard_normal((h + 1, n_paths)), axis=0))  
            s[0] = s0  # set the starting value of each path

            # Calculating the Monte Carlo estimator
            price = math.exp(-r-q * T) * sum(np.maximum(s[-1] - k, 0)) / i

        
#         if method == "milstein":
            
        

        print('The Option Value is: ', price)  # The European Option Value is:  6.57
        print('The Execution Time is: ', time() - t0) 
        
        
        

import matplotlib.pyplot as plt
import numpy as np
import math


class GeometricBrownianMotion:
    def __init__(self, initial_price, drift, volatility, dt, T):
        self.current_price = initial_price
        self.initial_price = initial_price
def generate_asset_price(s, sigma, r, t, q):  # would like to understand better
    return s * exp((r - q - 0.5 * sigma**2) * t + sigma*sqrt(t) * gauss(0, 1.0))

def mc_asset_price(s, sigma, r, t, q, n_paths=10000, option="call"):
    """
    use monte carlo simulation with n_paths to estimate price of an option
    s: current spot price
    k: strike price
    sigma: volatility
    r: risk-free rate
    t: time-to-expiration (in years)
    q: annualized dividend yield
    """
    future_prices = []
    for i in range(n_paths):  
        future_prices.append(generate_asset_price(s, sigma, r, t, q))

    return np.array(future_prices)


def mc_option_price(s, k, sigma, r, t, q, n_paths=10000, option="call"):
    """
    use monte carlo simulation with n_paths to estimate price of an option
    s: current spot price
    k: strike price
    sigma: volatility
    r: risk-free rate
    t: time-to-expiration (in years)
    q: annualized dividend yield
    """
    payoffs = []
    for i in range(n_paths):  
        s_T = generate_asset_price(s, sigma, r, t, q)
        if option == "call":
            payoff = max(0, s_T - k)
        else:
            payoff = max(0, k - s_T)

        payoffs.append(payoff)

    discount_factor = exp(-r-q*t)
    option_price = discount_factor * (sum(payoffs) / float(n_paths))

    return option_price

# TODO: Plotting
