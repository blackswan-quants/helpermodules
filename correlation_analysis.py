import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy import signal
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import MinMaxScaler
from helpermodules.memory_handling import PickleHelper

class CorrelationAnalysis:
    """
    This class is designed to facilitate advanced correlation analysis on stock data. It provides a comprehensive framework for computing correlation coefficients, p-values, and other relevant metrics to identify relationships between different stock prices.

    Attributes:
        dataframe (pd.DataFrame): A pandas DataFrame containing historical stock price data. This DataFrame is expected to have a datetime index and columns representing the closing prices of various stocks.
        tickers (list): A list of strings, each representing a unique ticker symbol for a stock. These ticker symbols should match the column names in the dataframe attribute.
    """

    def __init__(self, dataframe, tickers):
        """
        Initializes a CorrelationAnalysis object with a given DataFrame and list of ticker symbols.

        This method sets up the CorrelationAnalysis object by storing the provided DataFrame and list of ticker symbols as instance attributes. The DataFrame is expected to contain historical stock price data, and the list of ticker symbols should match the column names in the DataFrame.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing historical stock price data. This DataFrame is expected to have a datetime index and columns representing the closing prices of various stocks.
            tickers (list): A list of strings, each representing a unique ticker symbol for a stock. These ticker symbols should match the column names in the dataframe attribute.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
            raise ValueError("tickers must be a list of strings.")

        self.dataframe = dataframe
        self.tickers = tickers
        self.corrvalues = None
        self.pvalues = None
        self.best_lag = None
        self.corr_lags = None
        self.coint_scores = None
        self.winner = None

    def get_correlated_stocks(self, use_pct_change=False):
        """
        Computes correlation coefficients and p-values for the stocks.

        Args:
            use_pct_change (bool): Use percentage changes instead of raw values.

        Raises:
            ValueError: If tickers are not columns in the DataFrame.
        """
        self._validate_tickers()

        num_stocks = len(self.tickers)
        corr_values = np.zeros((num_stocks, num_stocks))
        pvalue_array = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            for j in range(num_stocks):
                vals_i, vals_j = self._get_values(i, j, use_pct_change)
                r_ij, p_ij = pearsonr(vals_i, vals_j)
                corr_values[i, j] = r_ij
                pvalue_array[i, j] = p_ij

        self.corrvalues = corr_values
        self.pvalues = pvalue_array
        PickleHelper(self.corrvalues).pickle_dump('correlationvalues_array')
        PickleHelper(self.pvalues).pickle_dump('pvalues_array')

    def get_correlation_lags(self, use_pct_change=False):
        """
        Computes cross-correlation lags for each stock pair.

        Args:
            use_pct_change (bool): Use percentage changes instead of raw values.
        """
        self._validate_tickers()

        num_stocks = len(self.tickers)
        lag_dim = self.dataframe.shape[0] * 2 - 1
        corr_lags = np.zeros((num_stocks, num_stocks, lag_dim))
        best_lag = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            for j in range(num_stocks):
                vals_i, vals_j = self._get_values(i, j, use_pct_change)
                lags = signal.correlation_lags(len(vals_i), len(vals_j), mode="full")
                correlation = signal.correlate(vals_i, vals_j, mode="full")
                corr_lags[i, j] = correlation
                best_lag[i, j] = lags[np.argmax(correlation)]

        self.corr_lags = corr_lags
        self.best_lag = best_lag
        PickleHelper(self.corr_lags).pickle_dump('all_lags_array')
        PickleHelper(self.best_lag).pickle_dump('best_lags_array')

    def cointegration_study(self, use_pct_change=False):
        """
        Performs cointegration analysis on stock pairs.

        Args:
            use_pct_change (bool): Use percentage changes instead of raw values.
        """
        self._validate_tickers()

        num_stocks = len(self.tickers)
        coint_values = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            for j in range(num_stocks):
                if i != j:
                    vals_i, vals_j = self._get_values(i, j, use_pct_change)
                    t_stat, _, _ = coint(vals_i, vals_j)
                    coint_values[i, j] = t_stat

        self.coint_scores = coint_values
        PickleHelper(self.coint_scores).pickle_dump('cointegration_values_array')

    def plot_corr_matrix(self):
        """Plots a heatmap of the correlation matrix."""
        self._validate_matrix(self.corrvalues, "correlation coefficients")
        self._plot_heatmap(self.corrvalues, "Correlation Matrix", center_color=True)

    def corr_stocks_pairs(self):
        """Identifies and plots the top three most correlated stock pairs."""
        self._validate_matrix(self.corrvalues, "correlation coefficients")
        self._validate_matrix(self.pvalues, "p-values")

        filtered_corr = np.where(self.pvalues > 0.05, np.nan, self.corrvalues)
        np.fill_diagonal(filtered_corr, np.nan)
        top_corr_idx = np.nanargmax(filtered_corr)
        second_corr_idx = np.nanargmax(filtered_corr, axis=None, out=None, mode='wrap', nan_policy='omit')
        third_corr_idx = np.nanargmax(filtered_corr, axis=None, out=None, mode='wrap', nan_policy='omit')

        i1, j1 = divmod(top_corr_idx, len(self.tickers))
        i2, j2 = divmod(second_corr_idx, len(self.tickers))
        i3, j3 = divmod(third_corr_idx, len(self.tickers))

        self.top_winner = [self.tickers[i1], self.tickers[j1]]
        self.second_winner = [self.tickers[i2], self.tickers[j2]]
        self.third_winner = [self.tickers[i3], self.tickers[j3]]

        print(f"Top three most correlated pairs: {self.top_winner}, {self.second_winner}, {self.third_winner}")

        PickleHelper(self.top_winner).pickle_dump('df_topcorr_pair')
        PickleHelper(self.second_winner).pickle_dump('df_secondcorr_pair')
        PickleHelper(self.third_winner).pickle_dump('df_thirdcorr_pair')

        self.dataframe[self.top_winner].plot(figsize=(12, 6), title=f"Price Comparison: {self.top_winner}")
        plt.show()

    def plot_lag_matrix(self):
        """Plots a heatmap of the best lag matrix."""
        self._validate_matrix(self.best_lag, "best lags")
        self._plot_heatmap(self.best_lag, "Best Lag Matrix")

    def plot_coint_matrix(self):
        """Plots a heatmap of the cointegration matrix."""
        self._validate_matrix(self.coint_scores, "cointegration scores")
        self._plot_heatmap(self.coint_scores, "Cointegration Matrix", center_color=True)

    def plot_stocks(self):
        """Plots the price of all stocks in the DataFrame."""
        self.dataframe.plot(subplots=True, figsize=(12, 6), title="Stock Prices")
        plt.show()

    # Helper Methods
    def _validate_tickers(self):
        """Validates that tickers exist in the DataFrame."""
        if not all(ticker in self.dataframe.columns for ticker in self.tickers):
            raise ValueError("Some tickers are missing in the DataFrame columns.")

    def _validate_matrix(self, matrix, name):
        """Validates that a matrix is not None."""
        if matrix is None:
            raise ValueError(f"{name.capitalize()} matrix has not been calculated yet.")

    def _get_values(self, i, j, use_pct_change):
        """Extracts values for analysis."""
        vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy() if use_pct_change else self.dataframe[self.tickers[i]].to_numpy()
        vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy() if use_pct_change else self.dataframe[self.tickers[j]].to_numpy()
        return vals_i, vals_j

    def _plot_heatmap(self, data, title, center_color=False):
        """Plots a heatmap for given data."""
        cmap = "coolwarm" if center_color else "viridis"
        plt.figure(figsize=(12, 6))
        sns.heatmap(data, annot=True, xticklabels=self.tickers, yticklabels=self.tickers, cmap=cmap, center=0 if center_color else None)
        plt.title(title)
        plt.show()
