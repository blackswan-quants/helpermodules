import numpy as np
from datetime import timedelta, datetime
import os
import pandas as pd
from twelvedata import TDClient
import yfinance as yf # import yfinance 
import re
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time
from pytickersymbols import PyTickerSymbols

class IndexData_Retrieval:
    """
    A class for downloading and processing historical stock price data using yfinance.

    Parameters:
        filename (str): Name of the pickle file to save or load df.
        index (str): Name of the stock index (e.g., 'S&P 500').
        interval (str): Time self.frequency of historical data to load (e.g., '1min', '1day', '1W').
        self.frequency (str): self.frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
        years (int, optional): Number of years of historical data to load (default: None).
        months (int, optional): Number of months of historical data to load (default: None).

    Methods:
        getdata():
            Loads a df of stock price data from a pickle file if it exists, otherwise creates a new df.
            Returns:
                pandas.df or None: df containing stock price data if loaded successfully, otherwise None.

        get_stockex_tickers():
            Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
            Returns:
                List[str]: List of ticker symbols extracted from the specified Wikipedia page.
        
        fetch_yahoo_dat():


        loaded_df():
            Downloads historical stock price data for the specified time window and tickers using yfinance.
            Returns:
                pandas.df or None: df containing downloaded stock price data if successful, otherwise None.
    """
    def __init__(self, filename, index, frequency, years=None, months=None, use_yfinance=False):
        self.filename = filename
        self.index = index
        self.df = pd.DataFrame()
        self.frequency = frequency
        self.tickers = []
        self.years = years
        self.months = months
        self.use_yfinance = use_yfinance

    def getdata(self):
        """
        Load a df of stock price data from a pickle file if it exists, otherwise create a new df.
        Returns:
            pandas.df or None: df containing stock price data if loaded successfully, otherwise None.
        """
        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"
        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.df = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.df.columns.tolist()
            return self.df
        else:
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
                raise ValueError(
                    f"Frequency not valid '{self.frequency}'. Valid frequencies are: {valid_intervals}")

            # Limitations for intraday data
            max_periods = {
                '1m': timedelta(days=7),
                '2m': timedelta(days=60),
                '5m': timedelta(days=60),
                '15m': timedelta(days=60),
                '30m': timedelta(days=60),
                '60m': timedelta(days=730),
                '90m': timedelta(days=730),
                '1h': timedelta(days=730),
            }

            intraday_intervals = ['1m', '2m', '5m', '15m',
                                '30m', '60m', '90m', '1h']
            if self.frequency in intraday_intervals:
                max_period = max_periods.get(
                    self.frequency, timedelta(days=60))
                if end_date - start_date > max_period:
                    start_date = end_date - max_period
                    print(
                        f"Start date set to {start_date.strftime('%Y-%m-%d')} due to limitations for the interval '{self.frequency}'.")

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
                print("No data retrieved from Yahoo Finance. Check the tickers and the date range.")
                return None

            # If only one ticker is requested, data columns are not multi-indexed
            if len(self.tickers) == 1:
                # Only one ticker
                data.columns = pd.MultiIndex.from_product(
                    [data.columns, self.tickers], names=['Attributes', 'Ticker'])
            else:
                # Data already has multi-indexed columns
                pass  # No action needed

            return data

        else:
            # Use Twelve Data API to fetch data
            load_dotenv()
            API_KEY = os.getenv('API_KEY')
            td = TDClient(apikey=API_KEY)
            # Initialize the final DataFrame, setting columns to be names of tickers
            dataframes = pd.DataFrame(np.nan, columns=self.tickers, index=[d for d in Timestamping(start_date, end_date)])

            # Dividing the date range into batches
            generator = Timestamping(start_date=start_date, end_date=end_date)
            batchsize = 5000
            boundaries = []
            try:
                boundary_start = next(generator)
                boundary_end = boundary_start
                while True:
                    for _ in range(batchsize - 1):
                        try:
                            boundary_end = next(generator)
                        except StopIteration:
                            boundaries.append((boundary_start, boundary_end))
                            raise StopIteration
                    boundaries.append((boundary_start, boundary_end))
                    boundary_start = next(generator)
            except StopIteration:
                pass

            # Divide tickers into batches
            def divide_tickers_in_batches(tickers):
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
                return [tickers[i:i + 55] for i in range(0, len(tickers), 55)]

            ticker_batches = divide_tickers_in_batches(tickers=self.tickers)

            for i, ticker_list in enumerate(ticker_batches):
                print(f'Processing batch {i + 1}/{len(ticker_batches)}')
                for ticker in ticker_list:
                    for j, (call_start, call_end) in enumerate(boundaries):
                        print(f'Fetching data for {ticker} - Call {j + 1}/{len(boundaries)}')
                        try:
                            df = td.time_series(
                                symbol=ticker,
                                interval=self.frequency,
                                start_date=call_start,
                                end_date=call_end,
                                outputsize=batchsize,
                                timezone="America/New_York",
                            ).as_pandas()
                            print(len(df))
                            for index, value in df['close'].items():
                                dataframes.loc[index, ticker] = value

                        except Exception as e:
                            print(f"Error fetching data for {ticker} - Call {j + 1}/{len(boundaries)}: {e}")
                if len(ticker_batches) == 55:
                    print('Please wait 60 seconds.')
                    time.sleep(60)
            return dataframes

    def loaded_df(self):
        """
        Downloads historical stock price data for the specified time window and tickers using the selected data source.
        Returns:
            pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.
        """
        if self.years is not None and self.months is None:
            time_window_months = self.years * 12
        elif self.months is not None and self.years is None:
            time_window_months = self.months
        else:
            raise ValueError("Exactly one of 'years' or 'months' should be provided.")

        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - pd.DateOffset(months=time_window_months)

        # Decide whether to use yfinance or Twelve Data API
        stocks_df = self.fetch_data(start_date=start_date, end_date=end_date, use_yfinance=self.use_yfinance)
        if stocks_df is not None:
            PickleHelper(obj=stocks_df).pickle_dump(filename=self.filename)
            return stocks_df
        else:
            print("Unable to retrieve data.")
            return None

    def clean_df(self, percentage):
        """
        Cleans the df by dropping stocks with NaN values exceeding the given percentage threshold.
        The cleaned df is pickled after the operation.

        Parameters:
        self
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
        
        Returns:
        None
        """
        if percentage > 1:
            percentage = percentage / 100

        #FIXME: this is not working, it's giving out an empty df
        for ticker in self.tickers:
            nan_values = self.df[ticker].isnull().values.any()
            if nan_values:
                count_nan = self.df[ticker].isnull().sum()
                if count_nan > (len(self.df) * percentage):
                    self.df.drop(ticker, axis=1, inplace=True)

        self.df.fillna(method='ffill', inplace=True)
        #FIXME: fml this doesn't work if i have consecutive days
        PickleHelper(obj=self.df).pickle_dump(filename=f'cleaned_{self.filename}')

class Timestamping:
    """
    A class to generate timestamps at specified minute intervals within market trading hours,
    iterating from a start date to an end date. 

    The market trading hours are defined as 9:45 AM to 3:15 PM (based on Simone's hypothesis).
    The generator skips weekends and ensures the timestamps are within trading hours.

    Attributes:
    -----------
    start_date : datetime
        The start date for the timestamp generation.
    end_date : datetime
        The end date for the timestamp generation.
    frequency_minutes : int, optional
        The frequency in minutes for generating timestamps. Default is 1 minute.

    Methods:
    --------
    __iter__():
        Returns the iterator object itself.
    __next__() -> datetime:
        Returns the next timestamp in the sequence.
    """
    def __init__(self, start_date: datetime, end_date: datetime, frequency_minutes=1):
        """
        Initializes the Timestamping class with start date, end date, and frequency in minutes.

        Parameters:
        -----------
        start_date : datetime
            The start date for the timestamp generation.
        end_date : datetime
            The end date for the timestamp generation.
        frequency_minutes : int, optional
            The frequency in minutes for generating timestamps. Default is 1 minute.
        """
        #define market trading hours (from 9:45 to 15:15, considering Simone's hypothesis)
        self.market_open_hour = 9
        self.market_open_minute = 45
        self.market_close_hour = 15
        self.market_close_minute = 15
        #initial assumption: unless stated, starts at the start of the trading day  
        self.current = start_date.replace(hour= self.market_open_hour ,minute=self.market_open_minute -1, second=0, microsecond=0)
        self.end = end_date
        self.frequency = frequency_minutes


    def __iter__(self):
        return self

    def __next__(self) -> datetime:
        """
        Returns the next timestamp in the sequence.

        Increments the current timestamp by the specified frequency (in minutes).
        Adjusts for end-of-day, weekends, and ensures timestamps fall within market trading hours.

        Raises:
        -------
        StopIteration:
            When the current timestamp exceeds the end date.
        
        Returns:
        --------
        datetime
            The next timestamp in the sequence.
        """
        #Increment current to the next stop:
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

