# get ticker history and annualized vol
def ticker_history(ticker, window=21, start=None, end=None):  # UNDERSTAND ANN. VOL :)
    ticker_obj = yf.Ticker(ticker)
    if start is None:
        start = dt.datetime.strftime(dt.datetime.today(), "%Y-%m-%d")
    if end is None:
        end = dt.datetime.strftime(dt.datetime.today(), "%Y-%m-%d")
        
    data = ticker_obj.history(period='1d', start=start, end=end)
    data["returns"] = data["Close"] / data["Close"].shift(-1)
    data["rolling_std"] = data["returns"].rolling(window).std()
    data["annualized_vol"] = data["rolling_std"] * 252 ** 0.5 

    return data.dropna()

tmp = ticker_history("SPY", window=21, start="2019-01-01", end="2021-02-23")
tmp.tail()


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def iron_condor(ticker="SPY", exp_date=None, n_contracts=1, wing_spread=10, short_delta=.16):
    """ 
    pull data to give iron condor summary. 
    returns expiration date chain and management date chain
    date must be in format m-d-y 
    """
    # would like to use datetime to set to 45 day iron condor if None
    # e.g. get closes to 45 day and 21 day (for management) and use those as 'defaults'
    # Assumes shorts are ATM for now, can improve later. 
    # need to give max loss, net credit, cash equivalent, anything else?
    chains = {}
    today = dt.datetime.today()
    if type(exp_date) == str:
        try:
            exp_date = dt.datetime(exp_date)
        except ValueError as e:
            print("exp_date not in the right format. try m-d-y")
            
    if exp_date is None:
        exp_date = today + dt.timedelta(45)  # default at 45 days out
        
    exp_dates = options.get_expiration_dates(ticker)
    for date in exp_dates:  # get list of expiration dates
        date_str = date.replace(",", "")
        month, day, year = date_str.split(" ")
        date_str = f"{month[:3]}-{day}-{year[-2:]}"
        date = dt.datetime.strptime(date_str, "%b-%d-%y")
        chains[date] = options.get_options_chain(ticker, date)

    exp_date_chain = chains[nearest(chains.keys(), exp_date)]  # return closest one to 45 days out
    
    manage_date = today + dt.timedelta(21)
    manage_date_chain = chains[nearest(chains.keys(), manage_date)]
        
#     return exp_date_chain, manage_date_chain
    ########
    call_chain["delta_diff"] = call_chain["delta"] - short_delta
    put_chain["delta_diff"] = put_chain["delta"] + short_delta
    short_call_row =  call_chain[call_chain["delta_diff"].min()] # not sure if this works exactly
    short_put_row = put_chain[put_chain["delta_diff"].min()]
    
    long_call_row = call_chain.loc[call_chain["strike"] == short_call_row["strike"] + wing_spread]] 
    long_put_row = put_chain.loc[put_chain["strike"] == short_put_row["strike"] - wing_spread]] 
    
    credit = short_call_row["price"] + short_put_row["price"]
    debit = long_call_row["price"] + long_put_row["price"]
    max_gain = (credit - debit) * 100 * n_contracts
    
    max_loss = max(call_spread, put_spread) - debit  # again, hopefully ~ 30% of width
    
    print(f"pct of width is: {round(max_gain / wing_spread, 2)}")

    # need to get spot price somewhere
    # spot_price = ...
    upper_breakeven = spot_price + (credit - debit)
    lower_breakeven = spot_price - (credit - debit)
    
    #########
    
  ### Black Scholes:
# Note: t is % of year
# MAKE SURE TO CHECK THESE. VALIDATE IF POSSIBLE
# Would like to get historical vol. PoP/P50 as well? IVR, IVX, IV%

class BlackScholes:
    def __init__(self, s, k, sigma, t, price, r, q=1e-10):  # if new IV works, can i change to 0?
        self.s = s  # have to set these as _fprime can take only one arg. is there a better way?
        self.k = k
        self.t = t
        self.q = q
        self.r = r
        self.price = price
        
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
        
    def _get_d2(self, d1, t, r, q=1e-10): 
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
            
        
#     def _fprime(self, sigma): 
#         """ 
#         f_prime is the function that computes the jacobian of the solver func with derivs across the rows 
#         Currently needs to have one param -- sigma
#         """
#         d1 = self._get_d1(self.s, self.k, sigma, self.t, self.r, self.q)
#         return s*sqrt(t)*norm.pdf(d1)*exp(-self.r*t)
    
#     def BS(self, s, k, t, sigma, r, q, option="call"):
#             if option == 'call':
#                 return self.call(s, k, sigma, t, r, q)
#             elif option == 'put':
#                 return self.put(s, k, sigma, t, r, q)
        
#     def implied_vol(self, option="call"):
#         """ use scipy quadratic solver for implied volatility """
#         iv_func = lambda x: self.BS(self.s, self.k, x, self.t, self.r, self.q, option) - self.price  # x = sigma
#         iv = fsolve(iv_func, self.solver_default, fprime=self._fprime, xtol=1e-3)
        
#         return iv[0]
    
    
    def get_implied_volatility(self, max_iter=100, tol=1e-3):
        """Guess the implied volatility."""
        
        # Set base case
        known_min = 0
        known_max = 10.0
        try:
            iv_guess = (
                math.sqrt(2 * math.pi / self.t) * (self.price / self.k)
            )
        except TypeError:
            print("TypeError in IV calculation. Returning NaN")
            return np.nan
            
        if option == "call":
            opt_val = self.call(self.s, self.k, iv_guess, self.t, self.r, self.q)
        if option == "put":
            opt_val = self.put(self.s, self.k, iv_guess, self.t, self.r, self.q)
            
        price_diff = opt_val - self.price

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
                opt_val = self.call(self.s, self.k, iv_guess, self.t, self.r, self.q)
            if option == "put":
                opt_val = self.put(self.s, self.k, iv_guess, self.t, self.r, self.q)
                
            price_diff = opt_val - self.price

            if iv_guess < 0.001:
                return 0

            iterations += 1
            if iterations > max_iter:
                print(f"Warning: Reached maximum number of iterations for "
                      + f"implied volatility guess for strike {self.k}. "
                      + f"Returning 0...")
                return 0

        return iv_guess
   
MbieG7gMB13pGPnFUNOsAtvUzv14
