import pandas as pd
import numpy as np
import requests
import datetime as dt
import json
from bs4 import BeautifulSoup

tickers = ["QQQ", "SPY", "EEM", "IWM", "DIA", "GLD", "TLT", "EWW", "FXE", "XOP", "FXI", 
           "SMH", "XLE", "XLV", "GWX", "USO", "XLI", "XLF", "XRT", "UVX"]


def get_iv_data(tickers):
    """ get IV, IVR, IVP for tickers in a list """
    ivs = []
    ivrs = []
    ivps = []
    pcs = []
    for ticker in tickers:
        try:
            url = f"https://volafy.net/equity/{ticker}" # for IVP
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content, 'lxml')
            p = soup.find_all("b")
            
            assert len(p) > 0
            
            for i in range(len(soup.find_all("b"))):
                if p[i].get_text().split(" ")[-1] == "(IV)":
                    ivs.append(float(p[i+1].get_text()[:-1]))
                if p[i].get_text().split(" ")[-1] == "(IVR)":
                    ivrs.append(float(p[i+1].get_text()))
                if p[i].get_text().split(" ")[-1] == "(IVP)":
                    ivps.append(float(p[i+1].get_text()))
                    
                if "Put/Call-Ratio" in p[i].get_text():
                    pcs.append(p[i].get_text().split(":")[-1])
                    
            if "Put/Call-Ratio" not in soup.get_text():
                print(f"Put/Call not available for {ticker}")
                pcs.append(np.nan)

        except AssertionError as e:
            print(f"Data not available for {ticker}")
            tickers.remove(ticker)
                        
    iv_screen = pd.DataFrame(columns=tickers, index=["iv", "ivr","ivp", "put_call_ratio"], data=[ivs, ivrs, ivps, pcs]).T
    return iv_screen.sort_values("ivp", ascending=False)
