# Libraries used
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from twelvedata import TDClient
from dotenv import load_dotenv
import os
# Initialize client - apikey parameter is required


class Asset:
    '''
    The Asset class helps managing single assets.

    To initialize it you have to input the following attributes:
    - type: (String) the asset type (stock, ETF, index, commodity, crypto, currency...)
    - ticker: (String) the ticker of the asset 
    - full_name: (String) full name of the asset

    NOTE: Use Asset.load() function to initialize the following attributes
    The following attributes are computed automatically using the assets list:
    - df: pandas dataframe of the asset (monthly)
    - isin: (String) ETF isin (if ETF, else None)
    - index_name: (String) ETF underlying index (if ETF, else None)
    - ter: (float) ETF ter (if ETF, else None)
    '''
    def __init__(self, type, ticker, full_name):
        self.type = type
        self.ticker = ticker
        self.full_name = full_name
        self.df = None
        self.isin = None
        self.index_name = None
        self.ter = None

    def apply_ter(self, ter):
        montly_ter_pct = (ter/12)/100

        columns = self.df[self.ticker]
        
        new_df = columns.apply(lambda x: x - montly_ter_pct)

        self.df[self.ticker] = new_df

    def load_etf_isin(self):
        with open('./very_long_html.txt', 'r') as file:
            data = file.read()

        # NOTA: spesso non trova l'etf a causa di parentesi o altre piccole differenze con justEtf, creare una funzione di ricerca
        #       con gerarchica basata su parole chiave (World, S&P 500, ...) piuttosto che cercare con .find()
        index = data.find(self.full_name.upper(),0)

        # se non trova nulla ritorna None
        if index == -1:
            return None

        i=0
        isin=""
        # getting the etf isin by reading from the fourth " symbol up to the fifth " symbol it encounters on the very_long_html file
        while (i<5):
            index+=1
            letter=data[index]
            if letter == '"':
                i+=1
            if (i>=4) & (letter!='"'):
                isin+=letter

        self.isin = isin
    
    def load_index_name(self):
        if self.isin == None:
            return 

        url="https://www.justetf.com/it/etf-profile.html?isin=" + self.isin

        # not showing browser GUI (makes code much faster)
        options = Options()
        options.add_argument("--headless")
        browser = webdriver.Chrome(options=options)

        browser.get(url)

        # get html of justetf page and look for index name
        html=browser.page_source
        index = html.find("replica l'indice",0) + 16

        index_name=""
        letter=''
        # the index name is found before the first . symbol in the text
        while letter!='.':
            index += 1
            letter = html[index]
            if letter != '.':
                index_name+=letter
            if letter == '&':
                index += 4

        self.index_name = index_name
    
    def load_ter(self):
        if self.isin == None:
            return 

        with open('./very_long_html.txt', 'r') as file:
            data = file.read() # replace'\n', ''

        index = data.find(self.isin,0)

        i=0
        ter=""
        while (i<5):
            index+=1
            letter=data[index]
            if letter == '"':
                i+=1
            if (i>=4) & (letter!='"') & (letter!='%'):
                ter+=letter

        self.ter = ter
    
    def load_df(self):
        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        td = TDClient(apikey=API_KEY)
        # twelve data tickers don't include the name of the exchange (eg: VUAA.MI would simply be VUAA)
        if "." in self.ticker:
            ticker = self.ticker[0:self.ticker.rfind(".")]
        else:
            ticker = self.ticker

        df = td.time_series(
            symbol=ticker,
            interval="1month",
            #start_date="2019-01-01",
            #end_date="2020-02-02",
            timezone="America/New_York"
        )

        # Returns pandas.DataFrame
        self.df = df.as_pandas()

    def load(self):
        if self.type == 'ETF':
            self.load_etf_isin()
            self.load_index_name()
            self.load_ter()
        self.load_df()
    
    def info(self):
        print("Full name: ", self.full_name)
        print("Ticker: ", self.ticker)
        print("Type: ", self.type)
        print("Ter: ", self.ter, "%")
        print("Index name: ", self.index_name)
        print("Isin: ", self.isin)
        print("Dataframe: \n", self.df)