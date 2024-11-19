#NOTE: in order to use this module, you also need to import memory_handling

# Libraries used
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import matplotlib.colors
import scipy.stats as ss
from scipy import signal
from statsmodels.tsa.stattools import coint # import from statsmodels.tsa.vector_ar.vecm if it doesn't work
from datetime import timedelta, datetime

from sklearn.preprocessing import MinMaxScaler

from helpermodules.memory_handling import PickleHelper

class CorrelationAnalysis:
    """
    A class for performing correlation analysis on stock data.
    
    Attributes:
        dataframe (pandas.DataFrame): The DataFrame containing the stock data.
        tickers (list): List of ticker symbols representing the stocks.
        start_datetime (str): Start date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        end_datetime (str): End date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        corrvalues (np.ndarray): Array containing correlation coefficients.
        pvalues (np.ndarray): Array containing p-values.
        winner (list): A list containing ticker symbols corresponding to the pair with the maximum correlation coefficient.
    """

    def __init__(self, dataframe, tickers):
        """
        Initialize the CorrelationAnalysis object.
        
        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the stock data.
            tickers (list): List of ticker symbols representing the stocks.
            start_datetime (str): Start date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str): End date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        """
        self.dataframe = dataframe
        self.tickers = tickers 
#        self.start_datetime = start_datetime
 #       self.end_datetime = end_datetime
        self.corrvalues = None
        self.pvalues = None
        self.winner = None

    def get_correlated_stocks(self, use_pct_change=False):
        """
        Calculate correlation coefficients and p-values for the given stocks within a given time period.
        
        Parameters:
            use_pct_change (bool): If True, use percentage change instead of raw values.
        
        Returns:
            None
        """
        corr_values = np.zeros([len(self.tickers), len(self.tickers)])
        pvalue_array = np.zeros([len(self.tickers), len(self.tickers)])
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                if use_pct_change:
                    # Calculate percentage change
                    vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy()
                else:
                    # Use original values
                    vals_i = self.dataframe[self.tickers[i]].to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].to_numpy()
                # Ensure values are numeric
                try:
                    vals_i = pd.to_numeric(vals_i, errors='coerce')
                    vals_j = pd.to_numeric(vals_j, errors='coerce')
                    # Filter out NaN values caused by non-numeric data
                    valid_indices = ~np.isnan(vals_i) & ~np.isnan(vals_j)
                    vals_i = vals_i[valid_indices]
                    vals_j = vals_j[valid_indices]
                    # Check if there's enough valid data to calculate correlation
                    if len(vals_i) == 0 or len(vals_j) == 0:
                        corr_values[i, j] = np.nan
                        pvalue_array[i, j] = np.nan
                        continue

                    # Calculate correlation
                    r_ij, p_ij = ss.stats.pearsonr(vals_i, vals_j)
                    corr_values[i, j] = r_ij
                    pvalue_array[i, j] = p_ij
                except Exception as e:
                    # Handle any unexpected errors
                    print(f"Error calculating correlation for {self.tickers[i]} and {self.tickers[j]}: {e}")
                    corr_values[i, j] = np.nan
                    pvalue_array[i, j] = np.nan     
        self.corrvalues = corr_values
        self.pvalues = pvalue_array
        PickleHelper(self.corrvalues).pickle_dump('correlationvalues_array')
        PickleHelper(self.pvalues).pickle_dump('pvalues_array')

    def get_correlation_lags(self, use_pct_change=False):
        """
        Calculate and store cross-correlation lags as vectors in a 3D array for each stock pair (i, j).
        Store the best lag for each correlation in the best_lag 2D array.
        
        Parameters:
            use_pct_change (bool): If True, use percentage change instead of raw values.

        Returns:
            None
        """
        corr_lags = np.zeros([len(self.tickers), len(self.tickers), self.dataframe.shape[0]*2-1])
        best_lag = np.zeros([len(self.tickers), len(self.tickers)])
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                if use_pct_change:
                    vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy()
                else:
                    vals_i = self.dataframe[self.tickers[i]].to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].to_numpy()
                lags_ij = signal.correlation_lags(len(vals_i), len(vals_j), mode="full")
                corr_lags[i, j] = lags_ij
                correlation = signal.correlate(vals_i, vals_j, mode="full")
                best_lag[i, j] = lags_ij[np.argmax(correlation)]
        self.best_lag = best_lag
        self.corr_lags = corr_lags
        PickleHelper(self.corr_lags).pickle_dump('all_lags_array')
        PickleHelper(self.best_lag).pickle_dump('best_lags_array')

    def cointegration_study(self, use_pct_change=False):
        """
        Perform the cointegration study for the matrix given.
    
        Parameters:
            use_pct_change (bool): If True, use percentage change instead of raw values.
        
        Returns:
            None
        """
    
        coint_value = np.zeros((len(self.tickers), len(self.tickers)))

        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                if i != j:  
                    if use_pct_change:
                        vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy()
                        vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy()
                    else:
                        vals_i = self.dataframe[self.tickers[i]].to_numpy()
                        vals_j = self.dataframe[self.tickers[j]].to_numpy()

                    # Perform the cointegration test
                    values, _, _ = coint(vals_i, vals_j) # we only get the t-statistic of unit-root test on residual and not the rest
                
                    # Store the results
                    coint_value[i, j] = values

        self.coint_scores = coint_value
        PickleHelper(self.coint_scores).pickle_dump('cointegration_values_array')

    def plot_corr_matrix(self):
        """
        Plot the correlation matrix heatmap for the given DataFrame.
        
        Returns:
            None
        """
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1), "red"],
                  [norm(-0.93), "lightgrey"],
                  [norm(0.93), "lightgrey"],
                  [norm(1), "green"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        plt.figure(figsize=(40, 20))
        seaborn.heatmap(pd.DataFrame(self.corrvalues, columns=self.tickers, index=self.tickers), annot=True, cmap=cmap)
        plt.show()

    def corr_stocks_pair(self):
        """
        Identify the pair of stocks with the maximum correlation coefficient and save it to a pickle file.
        
        Returns:
            None
        """
        corr_values_filtered = np.where(self.pvalues > 0.05, self.corrvalues, np.nan)
        #min_corr = np.nanmin(corr_values_filtered)
        tmp_arr = corr_values_filtered.copy()
        for i in range(len(tmp_arr)):
            tmp_arr[i, i] = 0
        #max_corr = np.nanmax(tmp_arr)
        #max_indexes = np.where(self.corrvalues == max_corr)
        #max_pair = [self.tickers[max_indexes[0][0]], self.tickers[max_indexes[0][1]]]
        # I commented all the things above because they aren't saved
        corr_order = np.argsort(tmp_arr.flatten())
        corr_num = corr_order[-1]
        max_pair = [self.tickers[corr_num // len(self.tickers)], self.tickers[corr_num % len(self.tickers)]]
        self.winner = max_pair
        print(max_pair)
        PickleHelper(self.winner).pickle_dump('df_maxcorr_pair')
        plt.figure(figsize=(40,20))
        plt.plot(self.dataframe[max_pair[1]])
        plt.plot(self.dataframe[max_pair[0]])
        plt.show

    def print_cross_corr(self, threshold: float, max_lag: int, volumes=None):
        for i in range(len(self.dataframe.columns)):
            for j in range(len(self.dataframe.columns)):
                if i != j:
                    corr_list = signal.correlate(self.dataframe[self.tickers[i]], self.dataframe[self.tickers[j]], mode='full')
                    lags = signal.correlation_lags(len(self.dataframe[self.tickers[i]]), len(self.dataframe[self.tickers[j]]))
                    corr_list = corr_list / (len(self.dataframe[self.tickers[i]]) * self.dataframe[self.tickers[i]].std() * self.dataframe[self.tickers[j]].std())
                    
                    # Normalize correlations to the range [0, 1]
                    sc = MinMaxScaler(feature_range=(0, 1))
                    corr_list_scaled = sc.fit_transform(corr_list.reshape(-1, 1)).flatten()
                    
                    for k, corr in enumerate(corr_list_scaled):
                        if abs(lags[k]) <= max_lag and corr >= threshold:
                            print(f"{self.tickers[i]} and {self.tickers[j]} are correlated ({corr}) with lag = {lags[k]}")

    def print_corr(self, threshold: float, max_lag: int, volume_filter=None):
        for shift in range(max_lag + 1):
            shifted_df = self.dataframe.shift(shift)
            concat_dataframe = pd.concat([self.dataframe, shifted_df.add_suffix(f'_shifted_{shift}')], axis=1)
            corr_matrix = concat_dataframe.corr('pearson')

            for i in range(len(self.dataframe.columns)):
                for j in range(len(self.dataframe.columns), len(concat_dataframe.columns)):
                    if i != j - len(self.dataframe.columns):
                        if corr_matrix.iloc[i, j] >= threshold:
                            print(f"{concat_dataframe.columns[i]} and {concat_dataframe.columns[j]} are correlated ({corr_matrix.iloc[i, j]}, shift = {shift})")

            print('\n')

    def plot_stocks(self):
        self.dataframe.plot(subplots=True); # prints the price off all the stocks in the dataframe 

    def plot_lag_matrix(self):
        """
        Plot the correlation matrix heatmap for the given DataFrame.
        
        Returns:
            None
        """
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1), "red"],
                  [norm(-0.93), "lightgrey"],
                  [norm(0.93), "lightgrey"],
                  [norm(1), "green"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        plt.figure(figsize=(40, 20))
        seaborn.heatmap(pd.DataFrame(self.best_lag, columns=self.tickers, index=self.tickers), annot=True, cmap=cmap)
        plt.show()

    def plot_coint_matrix(self):
        """
        Plot the cointegration matrix heatmap for the given DataFrame.
        
        Returns:
            None
        """
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1), "red"],
                  [norm(-0.93), "lightgrey"],
                  [norm(0.93), "lightgrey"],
                  [norm(1), "green"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        plt.figure(figsize=(40, 20))
        seaborn.heatmap(pd.DataFrame(self.coint_scores, columns=self.tickers, index=self.tickers), annot=True, cmap=cmap)
        plt.show()
    
    def most_corr_stocks_pair(self):
        """
        Identify the most correlated pair of stocks.
        
        Steps:
        1. Copy the correlation matrix (`self.corrvalues`) to a temporary array.
        2. Remove diagonal values (self-correlation) by setting them to 0.
        3. Find the maximum correlation value in the adjusted matrix.
        4. Identify the stock pair corresponding to the maximum correlation.
        5. Save the most correlated pair in `self.winner`.
        
        Returns:
            list: The names of the two most correlated stocks.
        """
        tmp_arr = self.corrvalues.copy()
        np.fill_diagonal(tmp_arr, 0)  # Exclude self-correlation
        max_corr = np.nanmax(tmp_arr)
        max_indexes = np.where(self.corrvalues == max_corr)
        self.winner = [self.tickers[max_indexes[0][0]], self.tickers[max_indexes[1][0]]]
        print(f"Most correlated pair: {self.winner} with correlation: {max_corr}")
        return self.winner

    def rolling_correlation(self, stock1, stock2, window='1H'):
        """
        Compute rolling correlation for two stocks over a specified time window.
        
        Args:
            stock1 (str): The ticker symbol of the first stock.
            stock2 (str): The ticker symbol of the second stock.
            window (str): The size of the rolling time window (default is 1 hour).
        
        Returns:
            pandas.Series: A time series of rolling correlation values.
        """
        df = self.dataframe[[stock1, stock2]].dropna()
        df = df.sort_index()
        rolling_corr = df[stock1].rolling(window=window).corr(df[stock2])
        return rolling_corr

    def generate_feature_dfs(self, stock1, stock2, window='1H', fillna_method=None):
        """
        Create individual DataFrames for the given stocks, with correlation as a feature.

        Args:
            stock1 (str): The ticker symbol of the first stock.
            stock2 (str): The ticker symbol of the second stock.
            window (str): The size of the rolling time window (default is 1 hour).
            fillna_method (str, optional): The method to fill NaN values in the correlation column. 
                                        Options: 'ffill', 'bfill', or None (default).

        Returns:
            Two DataFrames, one for each stock.
        """
        # Calculate rolling correlation
        rolling_corr = self.rolling_correlation(stock1, stock2, window)
        
        # Handle NaN values in the rolling correlation
        if fillna_method:
            if fillna_method == 'ffill':
                rolling_corr = rolling_corr.fillna(method='ffill')
            elif fillna_method == 'bfill':
                rolling_corr = rolling_corr.fillna(method='bfill')
            else:
                raise ValueError("Invalid fillna_method. Use 'ffill', 'bfill', or None.")

        # Create DataFrames for each stock
        df_stock1 = self.dataframe[[stock1]].copy()
        df_stock1['correlation'] = rolling_corr
        
        df_stock2 = self.dataframe[[stock2]].copy()
        df_stock2['correlation'] = rolling_corr

        return df_stock1, df_stock2

    def generate_combined_df(self, stock1, stock2, window='1H'):
        """
        Create a single DataFrame containing the time series for both stocks
        and the rolling correlation as a shared feature.
        
        Args:
            stock1 (str): The ticker symbol of the first stock.
            stock2 (str): The ticker symbol of the second stock.
            window (str): The size of the rolling time window (default is 1 hour).
        
        Returns:
            pandas.DataFrame: A single DataFrame containing both stocks' data and the correlation.
        """
        rolling_corr = self.rolling_correlation(stock1, stock2, window)
        combined_df = self.dataframe[[stock1, stock2]].copy()
        combined_df['correlation'] = rolling_corr
        return combined_df

    def BEST_corr_stocks_pair(self):
        """
        Identify the two most correlated stocks and save their data with rolling correlation as a feature.
        
        Steps:
        1. Find the most correlated stock pair using `most_corr_stocks_pair`.
        2. Compute the rolling correlation for this pair.
        3. Create individual DataFrames for both stocks with the rolling correlation as a feature.
        4. Print the first few rows of these DataFrames for verification.
        5. Save the DataFrames for both stocks using `PickleHelper`.
        
        Returns:
            None
        """
        # Step 1: Find the most correlated stock pair
        most_correlated = self.most_corr_stocks_pair()

        # Step 2: Generate feature DataFrames with rolling correlation
        df_stock1, df_stock2 = self.generate_feature_dfs(most_correlated[0], most_correlated[1], window='60min', fillna_method='ffill')

        # Step 3: Print the first few rows for checking
       # print(df_stock1.head())
       # print(df_stock2.head())

        # Step 4: Save the DataFrames to pickle files
        PickleHelper(df_stock1).pickle_dump('dataframe_stock_1')
        PickleHelper(df_stock2).pickle_dump('dataframe_stock_2')