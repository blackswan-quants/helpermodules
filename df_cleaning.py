
import datetime 
from datetime import timedelta
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
        load():
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

    def load(self):
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
    
    def fetch_twelve_data(self, start_date, end_date):
        
        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        td = TDClient(apikey=API_KEY)

        dataframes = []
        
        #divide tickers into batches
        def divide_tickers_inbatches(tickers):
            return [tickers[i:i+55] for i in range(0, len(tickers), 55)]
        
        #calculate how many datapoints based on the frequency and the time window
        def calculate_data_points(start_date, end_date, frequency):
            #define market trading hours (from 9:45 to 15:15, considering Simone's hypothesis)
            market_open_hour = 9
            market_open_minute = 45
            market_close_hour = 15
            market_close_minute = 15

            total_data_points = 0
            
            #determine the frequency type ('min' or 'h') and calculate data points accordingly
            if 'min' in frequency:
                current_date = start_date
                while current_date <= end_date:
                    #check if current_date is a trading day (Monday to Friday)
                    if current_date.weekday() < 5:  #monday (0) to Friday (4)
                        #calculate market open and close times for the current trading day
                        market_open_time = datetime(
                            current_date.year, current_date.month, current_date.day,
                            market_open_hour, market_open_minute
                        )
                        market_close_time = datetime(
                            current_date.year, current_date.month, current_date.day,
                            market_close_hour, market_close_minute
                        )
                        trading_duration = (market_close_time - market_open_time).total_seconds() / 60
                        total_data_points += trading_duration
                    current_date += timedelta(days=1)  

            elif 'h' in frequency:
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:
                        market_open_time = datetime(
                            current_date.year, current_date.month, current_date.day,
                            market_open_hour, market_open_minute
                        )
                        market_close_time = datetime(
                            current_date.year, current_date.month, current_date.day,
                            market_close_hour, market_close_minute
                        )
                        trading_duration = (market_close_time - market_open_time).total_seconds() / 3600
                        total_data_points += trading_duration
                    current_date += timedelta(days=1)  
            else:
                raise ValueError("Unsupported frequency")
            return total_data_points

        #get date ranges
        def split_date_range(start_date, end_date, tot_datapoints):
            start = start_date
            batch_size_days=5000
            end = end_date

            total_days = (end - start).days
            num_batches = (total_days // batch_size_days) + 1

            ranges = []
            for i in range(num_batches):
                if i == 0:
                    part_start = start
                else:
                    part_start = part_end + timedelta(days=1)
                    part_end = part_start + timedelta(days=batch_size_days - 1)
                if part_end > end:
                    part_end = end  #ensure end date of last batch does not exceed end_date
                ranges.append((part_start.strftime("%Y-%m-%d"), part_end.strftime("%Y-%m-%d")))

            return ranges
        
        ticker_batches = divide_tickers_inbatches(tickers=self.tickers) 
        data_points_per_call = 5000  # Maximum data points per API call
        total_data_points = calculate_data_points(start_date, end_date, self.frequency)
        date_ranges = split_date_range(start_date, end_date, total_data_points)
        
        for i, ticker_list in enumerate(ticker_batches):
                print(f'Processing batch {i+1}/{len(ticker_batches)}')
                for ticker in ticker_list:
                    
                    ticker_dataframes = []

                    for j, (call_start, call_end) in enumerate(date_ranges):
                        print(f'Fetching data for {ticker} - Call {j+1}/{len(date_ranges)}')
                        try:
                            dataframe = td.time_series(
                                symbol=ticker,
                                interval=self.frequency,
                                start_date=call_start,
                                end_date=call_end,
                                outputsize=data_points_per_call,
                                timezone="America/New_York",
                            ).as_pandas()
                            ticker_dataframes.append(dataframe)
                        except Exception as e:
                            print(f"Error fetching data for {ticker} - Call {j+1}/{len(date_ranges)}: {e}")
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

        end_date = datetime.date.today()
        start_date = end_date - pd.DateOffset(months=time_window_months)

        stocks_df = self.fetch_twelve_data(self.tickers, start_date=start_date, end_date=end_date, frequency=self.frequency)
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

