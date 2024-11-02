#NOTE: in order to use this module, you also need to import memory_handling

# Libraries used
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import matplotlib.colors
import scipy.stats as ss
from scipy import signal
from statsmodels.tsa.vector_ar.vecm import coint
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
  #      self.corrvalues = None
   #     self.pvalues = None
    #    self.winner = None

    def get_correlated_stocks(self, use_pct_change=True):
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
                r_ij, p_ij = ss.stats.pearsonr(vals_i, vals_j)
                corr_values[i, j] = r_ij
                pvalue_array[i, j] = p_ij
                
        self.corrvalues = corr_values
        self.pvalues = pvalue_array
        PickleHelper(self.corrvalues).pickle_dump('correlationvalues_array')
        PickleHelper(self.pvalues).pickle_dump('pvalues_array')

    def get_correlation_lags(self, use_pct_change=True):
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

        for i in range(n_tickers):
            for j in range(n_tickers):
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
        min_corr = np.nanmin(corr_values_filtered)
        tmp_arr = corr_values_filtered.copy()
        for i in range(len(tmp_arr)):
            tmp_arr[i, i] = 0
        max_corr = np.nanmax(tmp_arr)
        max_indexes = np.where(tmp_arr == max_corr) # put tmp_arr instead of self.corrvalues
        # max_pair = [self.tickers[max_indexes[0][0]], self.tickers[max_indexes[0][1]]]

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

