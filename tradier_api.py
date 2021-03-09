def get_strike_list(ticker, expiry, api_key):
    strike_list_response = requests.get(root_url + '/options/strikes?',
        params={'symbol': ticker, 'expiration': expiry},
        headers={'Authorization': api_key, 'Accept': 'application/json'}
    )
    strikes_json = strike_list_response.json()
    strikeList = strikes_json['strikes']['strike']
    print("List of available strike prices: ")
    print(strikeList)
    
    return strikeList
