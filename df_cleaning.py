
import datetime as dt
import os
import pandas as pd
from twelvedata import TDClient
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time

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
            self.tickers = self.get_stockex_tickers()
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

        end_date = dt.date.today()
        start_date = end_date - pd.DateOffset(months=time_window_months)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Divide tickers into batches
        def divide_tickers(tickers):
            return [tickers[i:i+55] for i in range(0, len(tickers), 55)]

        # Make API calls for each batch with rate limiting
        def API_limit(ticker_batches):
            """
            Retrieve historical stock price data from Twelve Data API in batches, with rate limiting to avoid exceeding API request limits.

            Args:
                ticker_batches (List[List[str]]): List of ticker symbol batches, each containing up to 55 tickers.

            Returns:
                pandas.DataFrame or None: Concatenated DataFrame containing stock price data for all tickers in ticker_batches,
                or None if no data was successfully retrieved.

            Raises:
                ValueError: If the API key is missing or the API call encounters an error.

            Notes:
                - The function loads the API key from environment variables.
                - The Twelve Data API has a rate limit, so the function pauses for 60 seconds between batches.
                - Requires `dotenv` to load the API key and `twelvedata` to make API calls.
            """

            # Load API key from environment variables
            load_dotenv()
            API_KEY = os.getenv('API_KEY')
            if not API_KEY:
                raise ValueError("API_KEY not found in environment variables.")

            td = TDClient(apikey=API_KEY)
            all_dataframes = []

            # Iterate through ticker batches with rate limiting
            for i, ticker_list in enumerate(ticker_batches):
                print(f'Processing batch {i + 1}/{len(ticker_batches)}')

                try:
                    # Request data from Twelve Data API
                    dataframe = td.time_series(
                        symbol=ticker_list,
                        interval=self.frequency,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        outputsize=5000,
                        timezone="America/New_York",
                    ).as_pandas()

                    # Append the dataframe for the current batch
                    all_dataframes.append(dataframe)
                    print('Please wait a minute while processing the next batch...')
                    time.sleep(60)  # Rate limiting: wait 60 seconds between batches

                except Exception as e:
                    print(f"Error fetching data for batch {i + 1}: {e}")
                    continue  # Skip to the next batch if an error occurs

            # Concatenate all dataframes if available, else return None
            if all_dataframes:
                stocks_dataframe = pd.concat(all_dataframes, ignore_index=True)
                return stocks_dataframe
            else:
                print('No data retrieved.')
                return None

        # Divide tickers into batches
        ticker_batches = divide_tickers(self.tickers)
        # Make API calls for each batch with rate limiting
        stocks_df = API_limit(ticker_batches)
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

