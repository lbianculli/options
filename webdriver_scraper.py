def option_chain(ticker, date):
    """ get option chain of ticker. Date should be in the format Y-M-D """
    # setup chrome driver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options) 

    # GET URL of ticker and navigate to home page
    home_page = f"https://www.nasdaq.com/market-activity/funds-and-etfs/{ticker}/option-chain"
    driver.get(home_page)
    
    # interact with buttons to get correct date/page
    month_button = 
    
    
    dom = driver.page_source
    driver.implicitly_wait(10)  # waits up to 10 seconds for dom retrieval
    driver.quit()
    print("Driver closed")
