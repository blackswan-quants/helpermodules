import datetime as dt
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from twelvedata import TDClient

from memory_handling import PickleHelper  # Switched to relative import


def divide_tickers(tickers, batch_size=55):
    """
    Splits a list of ticker symbols into smaller batches for API requests.

    Args:
        tickers (List[str]): List of ticker symbols.
        batch_size (int): Maximum number of tickers per batch.

    Returns:
        List[List[str]]: A list of batches, each containing up to batch_size tickers.
    """
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]


def API_limit(ticker_batches, frequency, start_date, end_date):
    """
    Retrieve historical stock price data from the Twelve Data API in batches, with rate limiting.

    Args:
        ticker_batches (List[List[str]]): List of ticker symbol batches.
        frequency (str): Time frequency for data (e.g., '1day').
        start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame or None: DataFrame with stock price data for all tickers in ticker_batches, or None if no data was retrieved.
    """
    # Load API key from environment variables
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        raise ValueError("API_KEY not found in environment variables.")

    td = TDClient(apikey=API_KEY)
    all_dataframes = []

    for i, ticker_list in enumerate(ticker_batches):
        logging.info(f'Processing batch {i + 1}/{len(ticker_batches)}')
        try:
            # Request data from Twelve Data API
            dataframe = td.time_series(
                symbol=ticker_list,
                interval=frequency,
                start_date=start_date,
                end_date=end_date,
                outputsize=5000,
                timezone="America/New_York",
            ).as_pandas()
            all_dataframes.append(dataframe)
            logging.info('Waiting 60 seconds before next batch...')
            time.sleep(60)  # Rate limiting

        except Exception as e:
            logging.error(f"Error fetching data for batch {i + 1}: {e}")
            continue

    if all_dataframes:
        return pd.concat(all_dataframes, ignore_index=True)
    logging.warning('No data retrieved from API.')
    return None


class DataFrameHelper:
    """
    A class for downloading and processing historical stock price data using the Twelve Data API.

    Parameters:
        filename (str): Name of the pickle file to save or load DataFrame.
        link (str): URL link to a Wikipedia page containing stock exchange information.
        interval (str): Time frequency of historical data to load (e.g., '1min', '1day', '1W').
        frequency (str): Frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
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
        self.dataframe = None  # Initialize dataframe attribute

    def load(self):
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
                self.dataframe = PickleHelper.pickle_load(self.filename).obj
                self.tickers = self.dataframe.columns.tolist()
                print(f"Loaded data from {self.filename}")
                return self.dataframe
            except Exception as e:
                print(f"Error loading pickle file '{self.filename}': {e}")

        # If file not found, attempt to fetch new data
        try:
            self.get_stockex_tickers()
            if not self.tickers:
                print("No tickers found. Unable to retrieve data.")
                return None  # Exit if no tickers are found

            self.dataframe = self.loaded_df()
            if self.dataframe is not None:
                print("New data fetched and loaded successfully.")
            else:
                print("Failed to fetch new data.")

            return self.dataframe
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
        Sets the `self.tickers` attribute with the extracted ticker symbols.
        """
        try:
            tables = pd.read_html(self.link)
            df = tables[4]
            df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'], axis=1, inplace=True)
            self.tickers = df['Ticker'].values.tolist()
            print(f"Retrieved {len(self.tickers)} tickers from {self.link}")
        except Exception as e:
            print(f"Error retrieving tickers from {self.link}: {e}")
            self.tickers = []  # Reset tickers if there's an error

    def loaded_df(self):
        """
        Download historical stock price data for the specified time window and tickers.

        Returns:
            pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.

        Raises:
            ValueError: If both years and months are provided, or neither is specified.
        """
        if (self.years is None) == (self.months is None):  # both or neither
            raise ValueError("Specify exactly one of 'years' or 'months'.")

        time_window_months = self.years * 12 if self.years else self.months
        end_date = dt.date.today()
        start_date = end_date - pd.DateOffset(months=time_window_months)
        start_date_str, end_date_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        # Prepare and call API_limit
        ticker_batches = divide_tickers(self.tickers)
        return API_limit(ticker_batches, self.frequency, start_date_str, end_date_str)

    def clean_df(self, percentage):
        """
        Cleans the DataFrame by removing columns (tickers) with NaN values exceeding a specified threshold.

        Parameters:
        ----------
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).

        Returns:
        -------
        None
        """
        # Convert percentage if given as an integer greater than 1
        threshold = percentage / 100 if percentage > 1 else percentage

        # Drop tickers with NaN values exceeding the threshold
        for ticker in self.tickers:
            if self.dataframe[ticker].isna().sum() > (len(self.dataframe) * threshold):
                self.dataframe.drop(columns=[ticker], inplace=True)

        # Fill remaining NaN values using forward fill
        self.dataframe.fillna(method='ffill', inplace=True)

        # Drop any remaining columns with NaN values /// If two consecutive days are NaN the entire column gets dropped. Don't know if this is a good
        # approach in this case, depends on how common NaNs are.
        self.dataframe.dropna(axis=1, inplace=True)

        # Save the cleaned DataFrame
        PickleHelper(obj=self.dataframe).pickle_dump(filename='cleaned_dataframe')
