
class BlackScholes:
    def __init__(self, s, k, sigma, t, price, r, q=1e-10):  # if new IV works, can i change to 0?
        self.s = s  # have to set these as _fprime can take only one arg. is there a better way?
        self.k = k
        self.sigma = sigma
        self.t = t
        self.price = price
        self.r = r
        self.q = q
        
    def _get_norm_cdf(self, d1):  # is this the same thing as scipy function?
        return (1 / sqrt(2*pi)) * exp(-d1**2 / 2)
    
    def call(self, s, k, sigma, t, r, q=1e-10):
        """ returns the call price of an option according to BS """
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        
        return norm.cdf(d1)*s*exp(-q*t) - norm.cdf(d2)*k*exp(-r*t)
        
    def put(self, s, k, sigma, t, r, q=1e-10):
        """ returns the put  price of an option according to BS """
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        
        return  k*exp(-r*t)*norm.cdf(-d2) - s*exp(-q*t)*norm.cdf(-d1)
    
    def delta(self, s, k, sigma, t, r, q=1e-10, option="call"):
        d1 = self._get_d1(s, k, sigma, t, r, q)
        if option == "call":
            return exp(-q*t) * norm.cdf(d1)
        else:
            return exp(-q*t) * (norm.cdf(d1) - 1)
        
    def gamma(self, s, k, sigma, t, r, q=1e-10):
        # note that gamma does not vary based on call or put
        d1 = self._get_d1(s, k, sigma, t, r, q)
        norm_cdf = self._get_norm_cdf(d1)
        
        return exp(-q*t) / (s * sigma * sqrt(t)) * norm_cdf
        
    def vega(self, s, k, sigma, t, r, q=1e-10):
        """ returns vega of underlying according to params. """ 
        d1 = self._get_d1(s, k, sigma, t, r, q)
        norm_cdf = self.get_norm_cdf(d1)
        
        return (s*exp(-q*t)*sqrt(t) * norm_cdf) / 100
        
        
    def theta(self, s, k, sigma, t, r, q=1e-10, option="call"):
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        norm_cdf = self._get_norm_cdf(d1)
        T = 365.242199 #  T = days in a year -- calendar OR trading days. Why would it not be trading days?
        
        if option == "call":
            return (-(s*sigma*exp(-q*t) / 2*sqrt(t)*norm_cdf) - r*k*exp(-r*t)*norm.cdf(d2) + q*s*exp(-q*t)*norm.cdf(d1)) / T
        else:
            return (-(s*sigma*exp(-q*t) / 2*sqrt(t)*norm_cdf) + r*k*exp(-r*t)*norm.cdf(-d2) - q*s*exp(-q*t)*norm.cdf(-d1)) / T

            
    def rho(self, s, k, sigma, t, r, q=1e-10, option="call"):
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        if option == "call":
            return (k*t*exp(-r*t)*norm.cdf(d2)) / 100
        else:
            return -(k*t*exp(-r*t)*norm.cdf(-d2)) / 100
        
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
    
    def get_implied_vol(self, s, k, t, price, r, q=1e-10, max_iter=100, tol=1e-3, option="call"):
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
    
    
