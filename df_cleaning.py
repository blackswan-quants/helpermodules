

from datetime import timedelta, datetime
import os
import pandas as pd
from twelvedata import TDClient
import re
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time

class DataFrameHelper:
    """
    A class for downloading and processing historical stock price data using the Twelve Data API.

    Parameters:
        filename (str): Name of the pickle file to save or load DataFrame.
        link (str): URL link to a Wikipedia page containing stock exchange information.
        interval (str): Time self.frequency of historical data to load (e.g., '1min', '1day', '1W').
        self.frequency (str): self.frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
        years (int, optional): Number of years of historical data to load (default: None).
        months (int, optional): Number of months of historical data to load (default: None).

    Methods:
        getdata():
            Loads a DataFrame of stock price data from a pickle file if it exists, otherwise creates a new DataFrame.
            Returns:
                pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.

        get_stockex_tickers():
            Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
            Returns:
                List[str]: List of ticker symbols extracted from the specified Wikipedia page.

        loaded_df():
            Downloads historical stock price data for the specified time window and tickers using the Twelve Data API.
            Returns:
                pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.
    """

    def __init__(self, filename, link, frequency, years=None, months=None):
        self.filename = filename
        self.link = link
        self.frequency = frequency
        self.tickers = []
        self.years = years
        self.months = months

    def getdata(self):
        """
        Load a DataFrame of stock price data from a pickle file if it exists, otherwise create a new DataFrame.
        Returns:
            pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.
        """
        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"
        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.dataframe = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.dataframe.columns.tolist()
            return self.dataframe
        else:
            self.tickers = self.get_stockex_tickers()
            self.dataframe = self.loaded_df()

        return None


    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
        Returns:
            List[str]: List of ticker symbols extracted from the specified Wikipedia page.
        """
        tables = pd.read_html(self.link)
        df = tables[4]
        df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
                axis=1, inplace=True)
        tickers = df['Ticker'].values.tolist()
        return tickers
    
    def fetch_twelve_data(self, start_date : datetime, end_date: datetime):
        
        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        td = TDClient(apikey=API_KEY)

        dataframes = []
        generator = Timestamping(start_date=start_date, end_date=end_date)
        
        batchsize = 5000
        boundaries = []
        try:
            boundary_start = next(generator)
            boundary_end = boundary_start 
            while True: 
                for _ in range(batchsize -1):
                    try:
                        boundary_end = next(generator)
                    except StopIteration:
                        boundaries.append((boundary_start, boundary_end))
                        raise StopIteration
                boundaries.append((boundary_start, boundary_end))
                boundary_start = next(generator)   
        except StopIteration:
            pass
        
        #divide tickers into batches
        def divide_tickers_inbatches(tickers):
            return [tickers[i:i+55] for i in range(0, len(tickers), 55)]
        
        ticker_batches = divide_tickers_inbatches(tickers=self.tickers) 
        
        for i, ticker_list in enumerate(ticker_batches):
                print(f'Processing batch {i+1}/{len(ticker_batches)}')
                for ticker in ticker_list:
                    
                    ticker_dataframes = []

                    for j, (call_start, call_end) in enumerate(boundaries):
                        print(f'Fetching data for {ticker} - Call {j+1}/{len(boundaries)}')
                        try:
                            dataframe = td.time_series(
                                symbol=ticker,
                                interval=self.frequency,
                                start_date=call_start, 
                                #FIXME: the point is that i need a function that keeps track of the last day and hour of every batch of 5k, 
                                #so that the next call starts with that timestamp 
                                end_date=call_end,
                                outputsize=batchsize,
                                timezone="America/New_York",
                            ).as_pandas()
                            ticker_dataframes.append(dataframe)
                        except Exception as e:
                            print(f"Error fetching data for {ticker} - Call {j+1}/{len(boundaries)}: {e}")
                    # Concatenate all dataframes for the ticker
                    if ticker_dataframes:
                        dataframes.append(pd.concat(ticker_dataframes, ignore_index=True))
                print('Please wait 60 seconds.')
                time.sleep(60)
        # Concatenate all dataframes for all tickers
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            print('No data retrieved.')
            return None
    
    def loaded_df(self):
        """
        Downloads historical stock price data for the specified time window and tickers using the Twelve Data API.
        Returns:
            pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.
        """
        if self.years is not None and self.months is None:
            time_window_months = self.years * 12
        elif self.months is not None and self.years is None:
            time_window_months = self.months
        else:
            raise ValueError("Exactly one of 'years' or 'months' should be provided.")

        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(months=time_window_months)

        stocks_df = self.fetch_twelve_data(start_date=start_date, end_date=end_date)
        return stocks_df

    def clean_df(self, percentage):
        """
        Cleans the DataFrame by dropping stocks with NaN values exceeding the given percentage threshold.
        The cleaned DataFrame is pickled after the operation.

        Parameters:
        self
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
        
        Returns:
        None
        """
        if percentage > 1:
            percentage = percentage / 100


        for ticker in self.tickers:
            nan_values = self.dataframe[ticker].isnull().values.any()
            if nan_values:
                count_nan = self.dataframe[ticker].isnull().sum()
                if count_nan > (len(self.dataframe) * percentage):
                    self.dataframe.drop(ticker, axis=1, inplace=True)

        self.dataframe.fillna(method='ffill', inplace=True)
        #FIXME: fml this doesn't work if i have consecutive days
        PickleHelper(obj=self.dataframe).pickle_dump(filename='cleaned_nasdaq_dataframe')

class Timestamping:
    def __init__(self, start_date: datetime, end_date: datetime, frequency_minutes=1):
        #define market trading hours (from 9:45 to 15:15, considering Simone's hypothesis)
        self.market_open_hour = 9
        self.market_open_minute = 45
        self.market_close_hour = 15
        self.market_close_minute = 15
        #initial assumption: unless stated, starts at the start of the trading day  
        self.current = start_date.replace(hour= self.market_open_hour ,minute=self.market_open_minute -1)
        self.end = end_date
        self.frequency = frequency_minutes


    def __iter__(self):
        return self

    def __next__(self) -> datetime:
        # TODO: Increment current to the next stop:
        # - Add 1 minute (later we'll add the frequency here)
        self.current += timedelta(minutes=1)
        # if it's end of day:
        #   - Add 1 day
        if self.current.minute > self.market_close_minute and self.current.hour >= self.market_close_hour:
            self.current += timedelta(days=1)
            self.current = self.current.replace(hour=self.market_open_hour, minute=self.market_open_minute)
        #   if it's end of Friday:
        #     - make it Monday
        if self.current.weekday() == 5:
            self.current += timedelta(days=2)
        if self.current.weekday() == 6:
            self.current += timedelta(days=1)

        if self.current > self.end:
            raise StopIteration

        return self.current

