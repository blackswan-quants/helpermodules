# Libraries used
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class Asset:
    def __init__(self, type, ticker, full_name, df, index_name, ter):
        self.type = type
        self.ticker = ticker
        self.df = df
        self.full_name = full_name
        self.index_name = index_name
        self.ter = ter

    def apply_ter(self, ter):
        montly_ter_pct = (ter/12)/100

        columns = self.df[self.ticker]
        
        new_df = columns.apply(lambda x: x - montly_ter_pct)

        self.df[self.ticker] = new_df

    def update_etf_isin(self):
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
    
    def update_index_name(self):
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
            index+=1
            letter=html[index]
            if (letter!='.'):
                index_name+=letter

        self.index_name = index_name
    
    def update_ter(self):
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
    