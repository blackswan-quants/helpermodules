import numpy as np
from datetime import timedelta, datetime
import os
import pandas as pd
from twelvedata import TDClient
import yfinance as yf  # added for yfinance
import re
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time
from pytickersymbols import PyTickerSymbols

class IndexData_Retrieval:
    """
    A class for downloading and processing historical stock price data using either the Twelve Data API or Yahoo Finance.

    Parameters:
        filename (str): Name of the pickle file to save or load df.
        index (str): Name of the stock index (e.g., 'S&P 500').
        interval (str): Time self.frequency of historical data to load (e.g., '1min', '1day', '1W').
        self.frequency (str): self.frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
        years (int, optional): Number of years of historical data to load (default: None).
        months (int, optional): Number of months of historical data to load (default: None).
        use_yfinance (bool, optional): If True, uses yfinance for data retrieval, otherwise uses Twelve Data API.

    Methods:
        getdata():
            Loads a dataframe of stock price data from a pickle file if it exists, otherwise creates a new dataframe.
            Returns:
                pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.

        get_stockex_tickers():
            Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
            Returns:
                List[str]: List of ticker symbols extracted from the specified Wikipedia page.

        fetch_data(start_date, end_date):
            Download historical stock prices for the specified time window and data source.
            Returns:
                pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.

        loaded_df():
            Downloads historical stock price data for the specified time window up to the current date and tickers using the selected data source.
            Returns:
                pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.

        clean_df(percentage):
            Cleans the dataframe by dropping stocks with NaN values exceeding the given percentage threshold.
            The cleaned dataframe is pickled after the operation.
            Returns:
                None
    """
    def __init__(self, filename, index, frequency, years=None, months=None, use_yfinance=False):
        self.filename = filename
        self.index = index
        self.df = pd.DataFrame()
        self.frequency = frequency
        self.tickers = []  # Placeholder for stock tickers
        self.years = years
        self.months = months
        self.use_yfinance = use_yfinance

    def getdata(self):
        # Append .pkl extension to filename if missing
        if not re.search("^.*\\.pkl$", self.filename):
            self.filename += ".pkl"
        file_path = "./pickle_files/" + self.filename

        # Check if the pickle file exists to load previously saved data
        if os.path.isfile(file_path):
            # Load data from pickle if it exists and set ticker columns
            self.df = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.df.columns.tolist()
            return self.df
        else:
            # Get tickers if no saved data, and load a new DataFrame
            self.tickers = self.get_stockex_tickers()
            self.df = self.loaded_df()

        return None


    def get_stockex_tickers(self):
        """
        Get list of the indexes' tickers using PyTickerSymbols
        Returns:
            list: List of ticker symbols
        """
        stock_data = PyTickerSymbols()
        tickers = stock_data.get_stocks_by_index(self.index)
        return [stock['symbol'] for stock in tickers]
    
    def fetch_data(self, start_date: datetime, end_date: datetime, use_yfinance=False):
        """
        Fetches historical data for multiple tickers within a specified date range from either the Twelve Data API or Yahoo Finance and stores
        the results in a DataFrame.
        This method allows the user to choose the data source by setting the 'use_yfinance' parameter.

        Parameters:
        -----------
        start_date : datetime
            The start date for fetching data.
        end_date : datetime
            The end date for fetching data.
        use_yfinance : bool, optional
            If True, uses yfinance for data retrieval; otherwise, uses the Twelve Data API (default is False).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the historical data for all specified tickers.
        """
        if use_yfinance:
            # Use yfinance to fetch data
            # Valid intervals of frequencies
            valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m',
                           '1h', '1d', '5d', '1wk', '1mo', '3mo']
            if self.frequency not in valid_intervals:
                raise ValueError(f"Frequency '{self.frequency}' not valid for yfinance.")
            
            # Download data from Yahoo Finance with selected settings
            data = yf.download(
                tickers=self.tickers,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=self.frequency,
                group_by='ticker',
                auto_adjust=False,  # Set to False since we'll manually select 'Adj Close'
                prepost=False,
                threads=True,
                proxy=None
            )
            # Select only the 'Adj Close' columns for each ticker
            data = data.xs('Adj Close', level=1, axis=1)

            if data.empty:
                print("No data retrieved from Yahoo Finance.")
                return None
            # Adjust columns if only one ticker was retrieved
            if len(self.tickers) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, self.tickers], names=['Attributes', 'Ticker'])
            return data
          
        else:
            # Initialize Twelve Data API client with API key from environment
            load_dotenv()
            API_KEY = os.getenv('API_KEY')
            td = TDClient(apikey=API_KEY)
            # Create a DataFrame with all ticker columns, filled initially with NaN
            dataframes = pd.DataFrame(np.nan, columns=self.tickers, index=[d for d in Timestamping(start_date, end_date)])
            
            #divide tickers into batches
            def divide_tickers_inbatches(tickers):
                """
                Divides the tickers list into batches of 55.
                Parameters:
                -----------
                tickers : list
                    The list of ticker symbols to be divided.
                Returns:
                --------
                list
                    A list of ticker batches, each containing up to 55 tickers.
                """
                return [tickers[i:i+55] for i in range(0, len(tickers), 55)]

            ticker_batches = divide_tickers_inbatches(tickers=self.tickers)

            # Generate date boundaries for batching if necessary (limit 5000 per batch)
            generator = Timestamping(start_date=start_date, end_date=end_date, frequency_minutes=self.frequency)
            boundaries = []
            timestamps = list(generator)
            if len(timestamps) <= 5000:
                # If data points are <= 5000, no need for batching
                boundaries = [(start_date, end_date)]
            else:
                # Split data into 5000-long boundaries for each API call
                boundary_start = timestamps[0]
                for i in range(0, len(timestamps), 5000):
                    boundary_end = timestamps[min(i + 4999, len(timestamps) - 1)]
                    boundaries.append((boundary_start, boundary_end))
                    boundary_start = timestamps[min(i + 5000, len(timestamps) - 1)]

            # Fetch data for each batch of tickers and boundaries
            for i, ticker_list in enumerate(divide_tickers_inbatches(self.tickers)):
                print(f'Processing batch {i + 1}/{len(ticker_batches)}')
                for ticker in ticker_list:
                    if len(boundaries) == 1:
                        # Single batch if within 5000 limit
                        call_start, call_end = boundaries[0]
                        print(f'Fetching single batch data for {ticker}')
                        try:
                            df = td.time_series(
                                symbol=ticker,
                                interval=f"{self.frequency}m",
                                start_date=call_start,
                                end_date=call_end,
                                outputsize=5000,
                                timezone="America/New_York",
                            ).as_pandas()
                            for index, value in df['close'].items():
                                dataframes.loc[index, ticker] = value
                        except Exception as e:
                            print(f"Error fetching data for {ticker}: {e}")
                    else:
                        # Loop for multi-boundary data retrieval when limit exceeded
                        for j, (call_start, call_end) in enumerate(boundaries):
                            print(f'Fetching data for {ticker} - Call {j + 1}/{len(boundaries)}')
                            try:
                                df = td.time_series(
                                    symbol=ticker,
                                    interval=f"{self.frequency}m",
                                    start_date=call_start,
                                    end_date=call_end,
                                    outputsize=5000,
                                    timezone="America/New_York",
                                ).as_pandas()
                                for index, value in df['close'].items():
                                    dataframes.loc[index, ticker] = value
                            except Exception as e:
                                print(f"Error fetching data for {ticker} - Call {j + 1}/{len(boundaries)}: {e}")
                if len(ticker_batches) == 55:
                    print('Please wait 60 seconds.')
                    time.sleep(60)  # API limit management
            return dataframes


    def loaded_df(self):
        # Calculate months of data to retrieve based on either years or months
        if self.years is not None and self.months is None:
            time_window_months = self.years * 12
        elif self.months is not None and self.years is None:
            time_window_months = self.months
        else:
            raise ValueError("Specify either 'years' or 'months', not both.")
        
        # Define the time window from end_date and start_date
        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - pd.DateOffset(months=time_window_months)
        # Fetch data within the time window
        stocks_df = self.fetch_data(start_date=start_date, end_date=end_date)
        if stocks_df is not None:
            PickleHelper(obj=stocks_df).pickle_dump(filename=self.filename)
            return stocks_df
        else:
            print("Unable to retrieve data.")
            return None

    def clean_df(self, percentage):
        # Ensure percentage is in decimal form
        if percentage > 1:
            percentage = percentage / 100
        # Drop tickers with NaN above specified threshold
        for ticker in self.tickers:
            count_nan = self.df[ticker].isnull().sum()
            if count_nan > (len(self.df) * percentage):
                self.df.drop(ticker, axis=1, inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        #FIXME: fml this doesn't work if i have consecutive days
        PickleHelper(obj=self.df).pickle_dump(filename=f'cleaned_{self.filename}')

class Timestamping:
    def __init__(self, start_date: datetime, end_date: datetime, frequency_minutes=1):
        # Set market open/close times
        self.market_open_hour = 9
        self.market_open_minute = 45
        self.market_close_hour = 15
        self.market_close_minute = 15
        # Initialize starting point and end date for iteration
        self.current = start_date.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        self.end = end_date
        self.frequency = frequency_minutes

    def __iter__(self):
        return self

    def __next__(self) -> datetime:
        # Move forward by frequency, check if within market hours
        self.current += timedelta(minutes=self.frequency)
        if self.current.minute > self.market_close_minute and self.current.hour >= self.market_close_hour:
            # Advance to next day at market open if past close
            self.current += timedelta(days=1)
            self.current = self.current.replace(hour=self.market_open_hour, minute=self.market_open_minute)
        if self.current.weekday() == 5:
            self.current += timedelta(days=2)
        if self.current.weekday() == 6:
            self.current += timedelta(days=1)
        if self.current > self.end:
            raise StopIteration
        return self.current
