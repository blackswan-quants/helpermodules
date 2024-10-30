
import numpy as np
from datetime import timedelta, datetime
import os
import pandas as pd
from twelvedata import TDClient
import re
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time

class IndexData_Retrieval:
    """
    A class for downloading and processing historical stock price data using the Twelve Data API.

    Parameters:
        filename (str): Name of the pickle file to save or load data from.
        link (str): URL to a Wikipedia page with stock exchange ticker information.
        frequency (str): Frequency of the historical data (e.g., '1d', '1wk').
        years (int, optional): Number of years of historical data to load.
        months (int, optional): Number of months of historical data to load.
        yfinance (bool, optional): Flag to determine if yfinance should be used for data retrieval (default: True).

    Methods:
        load():
            Attempts to load data from a pickle file; if unsuccessful, fetches new data from the API.

            Returns:
                pd.DataFrame or None: Returns a DataFrame with stock price data if successfully loaded or retrieved;
                returns None if neither option succeeds.

        get_stockex_tickers():
            Retrieves ticker symbols from a specified Wikipedia page and stores them in `self.tickers`.

            Returns:
                None: This method updates the `self.tickers` attribute with a list of tickers from the webpage.

        loaded_df():
            Downloads historical data for the tickers over a specified time window.

            Returns:
                pd.DataFrame or None: Returns a DataFrame containing the downloaded stock price data for the specified
                tickers and time period; returns None if data retrieval fails.

        clean_df(percentage):
            Cleans the DataFrame by removing columns with NaN values exceeding a specified threshold.

            Parameters:
                percentage (float): The threshold percentage of NaN values for removing a ticker's data. For example,
                if `percentage` is 5, tickers with more than 5% missing data are removed.

            Returns:
                None: The method updates `self.dataframe` with cleaned data, removing any columns with excessive NaNs.
    """

    def __init__(self, filename, link, frequency, years=None, months=None,yfinance=True):
        self.filename = filename
        self.link = link
        self.df = pd.DataFrame()
        self.frequency = frequency
        self.tickers = []
        self.years = years
        self.months = months
        self.yfinance = yfinance

    def getdata(self):
        """
        Loads a DataFrame of stock price data from a pickle file if it exists; otherwise, fetches data from API.

        The function first attempts to load stock data from a pickle file, which contains historical price data.
        If no file is found, it fetches ticker symbols from the specified Wikipedia page, retrieves new stock data
        from the API, and saves it to a pickle file.

        Returns:
            pandas.DataFrame: DataFrame containing stock price data if successfully loaded or retrieved.
            If neither option succeeds, returns None.
        """
        # Ensure filename ends with '.pkl'
        if not self.filename.endswith(".pkl"):
            self.filename += ".pkl"

        # Construct file path using os.path.join for compatibility
        file_path = os.path.join("pickle_files", self.filename)

        # Attempt to load from existing pickle file
        if os.path.isfile(file_path):
            try:
                self.df = PickleHelper.pickle_load(self.filename).obj
                #self.tickers = self.df.columns.tolist() /// Would work with 12data, not with yfinance
                self.tickers = self.df.columns.get_level_values(1).tolist()
                print(f"Loaded data from {self.filename}")
                return self.df
            except Exception as e:
                print(f"Error loading pickle file '{self.filename}': {e}")

        # If file not found, attempt to fetch new data
        try:
            self.get_stockex_tickers()
            if not self.tickers:
                print("No tickers found. Unable to retrieve data.")
                return None  # Exit if no tickers are found

            self.df = self.loaded_df()
            if self.df is not None:
                print("New data fetched and loaded successfully.")
            else:
                print("Failed to fetch new data.")

            return self.df
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None


    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
        Sets the `self.tickers` attribute with the extracted ticker symbols.
        """
        try:
            # Read all tables on the Wikipedia page
            tables = pd.read_html(self.link)
            print(f"Found {len(tables)} tables on the page.")

            # Find the first table with at least one column and select only the first column
            for table in tables:
                self.tickers = table.loc[:, "Symbol"].dropna().tolist()  # Get the first column and drop any NaN values
                print(f"Retrieved {len(self.tickers)} tickers from the first column of the table.")
                return  # Exit after finding the first valid table

            print("No valid table found with ticker symbols.")
        except Exception as e:
            print(f"Error retrieving tickers from {self.link}: {e}")
            self.tickers = []  # Reset tickers if there's an error
    
    def fetch_twelve_data(self, start_date : datetime, end_date: datetime):
        """
        Fetches historical data for multiple tickers within a specified date range from the Twelve Data API and stores the results in a DataFrame.
        This function divides the date range into manageable batches, retrieves the data from the Twelve Data API, and handles rate limits by batching tickers and pausing between batches.
        The data retrieval follows New York trading hours (9:45 AM to 3:15 PM EST).

        Parameters:
        -----------
        start_date : datetime
            The start date for fetching data.
        end_date : datetime
            The end date for fetching data.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the historical data for all specified tickers.
        """

        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        td = TDClient(apikey=API_KEY)
        #initializing the final df, setting columns to be name of tickers
        dataframes = pd.DataFrame(np.nan, columns = self.tickers, index = [d for d in Timestamping(start_date, end_date)])
        
        #dividing the 5k batches and keeping track of the
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
        
        for i, ticker_list in enumerate(ticker_batches):
                print(f'Processing batch {i+1}/{len(ticker_batches)}')
                for ticker in ticker_list:
                
                    for j, (call_start, call_end) in enumerate(boundaries):
                        print(f'Fetching data for {ticker} - Call {j+1}/{len(boundaries)}')
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
                                dataframes.loc[index, ticker] =  value
                            
                        except Exception as e:
                            print(f"Error fetching data for {ticker} - Call {j+1}/{len(boundaries)}: {e}")
                if len(ticker_batches) == 55: 
                    print('Please wait 60 seconds.')
                    time.sleep(60)
        return dataframes
    
    def loaded_df(self):
        """
        Downloads historical stock price data for the specified time window and tickers using either the Twelve Data API
        or yfinance, based on the preferred method.

        Args:
            use_yfinance (bool): If True, uses yfinance to download data instead of the Twelve Data API.

        Returns:
            pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.
        """
        if (self.years is None) == (self.months is None):  # both or neither
            raise ValueError("Specify exactly one of 'years' or 'months'.")

        time_window_months = self.years * 12 if self.years else self.months
        end_date = datetime.now() - timedelta(days = 30)  #why has this been set to 30 days?
        start_date = end_date - pd.DateOffset(months=time_window_months)

        if not isinstance(self.yfinance, bool):
            raise TypeError(f"Expected 'yfinance' to be a boolean, but got {type(self.yfinance).__name__}")

        if self.yfinance:
            # Using yfinance to download data
            try:
                data = yf.download(self.tickers, start=start_date_str, end=end_date_str, interval=self.frequency)
                # Adjusting format if multiple tickers are used
                if len(self.tickers) == 1:
                    data.columns = [self.tickers[0]]  # Rename column if only one ticker is downloaded
                return data
            except Exception as e:
                print(f"Error downloading data with yfinance: {e}")
                return None

        #Using 12 data

        stocks_df = self.fetch_twelve_data(start_date=start_date, end_date=end_date)
        PickleHelper(obj=stocks_df).pickle_dump(filename='nasdaq_dataframe')
        return stocks_df

    def clean_df(self, percentage):
        """
        Cleans the DataFrame by removing columns (tickers) with NaN values exceeding a specified threshold.

        Parameters:
        ----------
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).

        Returns:
        -------
        None: The method updates `self.df` with cleaned data, removing any columns with excessive NaNs.
        """
        # Convert percentage if given as an integer greater than 1
        threshold = percentage / 100 if percentage > 1 else percentage

        #Currently only working with yfinance (cannot test with 12data because of lack of API key)

        # Drop tickers with NaN values exceeding the threshold
        for ticker in self.tickers:
            # Check if the ticker exists as a first-level column in the DataFrame
            if ticker in self.df.columns.get_level_values(1):
                # Calculate the percentage of NaN values for that ticker across all fields
                if self.df.xs(ticker,axis=1,level=1).isna().sum().sum() > (len(self.df) * threshold):
                    self.df.drop(columns=[ticker], level=1, inplace=True)
            else:
                print(f"Warning: Ticker '{ticker}' not found in DataFrame columns.")

        # Fill remaining NaN values using forward fill
        self.df.ffill(inplace=True)

        # Drop any remaining columns with NaN values /// If two consecutive days are NaN the entire column gets dropped. Don't know if this is a good
        # approach in this case, depends on how common NaNs are.
        self.df.dropna(axis=1, inplace=True)

        # Save the cleaned DataFrame
        PickleHelper(obj=self.df).pickle_dump(filename='cleaned_dataframe')

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

