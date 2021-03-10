class TradierAPI:
    def __init__(self, key):
        self.key = key
        self.root_url = "https://sandbox.tradier.com/v1/markets/
        
    def get_strike_list(self, ticker, exp_date):
        strike_list_response = requests.get(self.root_url + '/options/strikes?',
            params={'symbol': ticker, 'expiration': exp_date},
            headers={'Authorization': self.api_key, 'Accept': 'application/json'}
        )
        strikes_json = strike_list_response.json()
        strikes = strikes_json['strikes']['strike']
        print("List of available strike prices: ")
        print(strikes)

        return strikes
    
    def option_history(self, ticker, start_date, end_date):
        """ get historical data for an option """
        dates_response = requests.get(self.root_url + '/options/expirations?',
            params={'symbol': ticker},
            headers={'Authorization': self.key, 'Accept': 'application/json'}
        )        
