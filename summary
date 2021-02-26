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

def iron_condor(ticker="SPY", exp_date=None, wing_spread=10, short_delta=.16):
    """ 
    pull data to give iron condor summary. 
    returns expiration date chain and management date chain
    date must be in format m-d-y 
    """
    # would like to use datetime to set to 45 day iron condor if None
    # e.g. get closes to 45 day and 21 day (for management) and use those as 'defaults'
    # Assumes shorts are ATM for now, can improve later. 
    # need to give max loss, net credit, cash equivalent, anything else?
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
    
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
        
    return exp_date_chain, manage_date_chain
    
    
    
    
  
  ### Black Scholes:
# Note: t is % of year
# MAKE SURE TO CHECK THESE. VALIDATE IF POSSIBLE
# Would like to get historical vol. PoP/P50 as well? IVR, IVX, IV%

class BlackScholes:
    def __init__(self, s, k, sigma, t, price, r, q=1e-10, solver_default=.20):
        self.s = s  # have to set these as _fprime can take only one arg. is there a better way?
        self.k = k
        self.t = t
        self.q = q
        self.r = r
        self.price = price
        self.solver_default = solver_default
    
    def _get_norm_cdf(self, d1):  # is this the same thing as scipy function?
        return (1 / sqrt(2*pi)) * exp(-d1**2 / 2)
    
    def call(self, s, k, sigma, t, r, q=1e-10):
        """ returns the call price of an option according to BS """
        d1 = self._get_d1(s, k, sigma, t, r, q)
        d2 = self._get_d2(d1, sigma, t)
        
        return norm.cdf(d1)*s*exp(-q*t) - norm.cdf(d2)*k*exp(-r * t)
        
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
        T = 252  #  T = days in a year -- calendar OR trading days. Why would it not be trading days?
        
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
            
        
    def _fprime(self, sigma): 
        """ 
        f_prime is the function that computes the jacobian of the solver func with derivs across the rows 
        Currently needs to have one param -- sigma
        """
        d1 = self._get_d1(self.s, self.k, sigma, self.t, self.r, self.q)
        return s*sqrt(t)*norm.pdf(d1)*exp(-self.r*t)
    
    def BS(self, s, k, t, sigma, r, q, option="call"):
            if option == 'call':
                return self.call(s, k, sigma, t, r, q)
            elif option == 'put':
                return self.put(s, k, sigma, t, r, q)
        
    def implied_vol(self, option="call"):
        """ use scipy quadratic solver for implied volatility """
        iv_func = lambda x: self.BS(self.s, self.k, x, self.t, self.r, self.q, option) - self.price  # x = sigma
        iv = fsolve(iv_func, self.solver_default, fprime=self._fprime, xtol=1e-6)
        
        return iv[0]
    
# mess around with numbers
s = 382.33
k = 360
sigma = .3103
t =  25/365  # MAKE SURE THIS IS RIGHT
price = 26.2
rf = 0.014
q = 0.0178

bs_obj = BlackScholes(s=s, k=k, sigma=sigma, t=t, price=price, r=rf, q=q)
bs_obj.BS(s=s, k=k, sigma=sigma, t=t, r=rf, q=q, option="call")  # why does this not work if q=0????
bs_obj.call(s, k, sigma, t, rf, q)

bs_obj.implied_vol() 

bs_obj.delta(s, k, sigma, t, rf, q)
