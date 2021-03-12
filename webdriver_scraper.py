month_map = {
    "1": ,
    

def option_chain(ticker, date):
    """ get option chain of ticker. Date should be in the format Y-M-D """
    # setup chrome driver
    chrome_options = Options()
#     chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options) 
    
    # get closest date for option chain and other dates to assist in navigating to option chain
    month_beginning = date[:-2] + "01
    dt_obj = dt.datetime.strptime("
    mydate.strftime("%B")


    # GET URL of ticker and navigate to home page
    home_page = f"https://www.nasdaq.com/market-activity/funds-and-etfs/{ticker}/option-chain"
    driver.get(home_page)
    
    # interact with buttons to get correct date/page
    select = Select(driver.find_element_by_xpath("/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div/div/div[1]/div/div[1]/div[1]/button"))
    select.select_by_visible_text('Banana')    
    
    dom = driver.page_source
    driver.implicitly_wait(10)  # waits up to 10 seconds for dom retrieval
    driver.quit()
    print("Driver closed")
