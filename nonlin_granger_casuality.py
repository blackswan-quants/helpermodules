
"""" https://pypi.org/project/nonlincausality/#description 
     https://github.com/mrosol/Nonlincausality/blob/master/README.md
"""
import pandas as pd
import numpy as np
from memory_handling import PickleHelper
from nonlincausality import nonlincausalityNN as nlc_nn

class NonlinearNNGrangerCausalityAnalysis:
    """
    This class performs nonlinear Granger causality analysis between pairs of stock tickers 
    using the nonlincausalityNN method from the nonlincausality library. It applies neural 
    networks to forecast and test for causality in time series data, allowing for the capture 
    of nonlinear dependencies.
    """
    
    def __init__(self, dataframe, max_lag=5, nn_config=['d','dr','d','dr'], nn_neurons=[100, 0.05, 100, 0.05]): #MLP Version
        """
        Initializes the NonlinearNNGrangerCausalityAnalysis object with stock data and configuration.

        Args:
            dataframe (pd.DataFrame): A DataFrame containing time series data of stock prices.
            max_lag (int): The maximum number of lags to consider when testing for causality.
            nn_config (list): List defining the configuration of the neural network layers.
                - 'd' stands for dense layer, 'dr' stands for dropout regularization.
            nn_neurons (list): List specifying the number of neurons in each layer and dropout rate.
                - The format: [neurons, dropout_rate, neurons, dropout_rate, ...]
        """
        # Validate that the DataFrame has columns
        if dataframe.empty or dataframe.columns.empty:
            raise ValueError("DataFrame must have at least one column representing tickers.")
        # Store the dataframe with stock data
        self.dataframe = dataframe
        # Extract the column names (tickers) from the DataFrame
        self.tickers = dataframe.columns
        # Store the maximum lag for causality tests
        self.max_lag = max_lag
        # Store the neural network configuration
        self.nn_config = nn_config
        # Store the number of neurons and dropout rates for the neural network layers
        self.nn_neurons = nn_neurons
        # Initialize an empty dictionary to store causality results
        self.results = {}

    def calculate_nonlinear_nn_causality(self, epochs=[50, 50], learning_rate=[0.0001, 0.00001], batch_size=32):
        """
        Calculate nonlinear Granger causality for each pair of tickers using a neural network-based 
        forecasting method.

        Args:
            epochs (list): List specifying the number of epochs for training the neural network 
                           for each phase of the model.
            learning_rate (list): List of learning rates for each training phase.
            batch_size (int): The size of the batch used during training.

        Returns:
            dict: A dictionary where keys are ticker pairs (ticker_x, ticker_y), and the values are 
                  another dictionary containing causality scores and p-values.
        """
        # Iterate over all pairs of tickers
        for i, ticker_x in enumerate(self.tickers):
            for j, ticker_y in enumerate(self.tickers):

                if i == j:
                    # If testing the same ticker, set NaNs
                    self.results[(ticker_x, ticker_y)] = {
                    "p_values": np.nan,
                    "f_statistics": np.nan
                    }
                else:  # Skip if the tickers are the same
                    # Prepare data for the analysis: Extract the time series of stock prices for both tickers
                    data_x = self.dataframe[ticker_x].dropna().values.reshape(-1, 1)  # Reshape for model input
                    data_y = self.dataframe[ticker_y].dropna().values.reshape(-1, 1)
                    
                    # Split data into training and validation sets (70% training, 30% validation)
                    train_size = int(0.7 * len(data_x))  # 70% for training
                    data_train, data_val = data_x[:train_size], data_x[train_size:]
                    # Perform the nonlinear Granger causality test using the nonlincausalityNN method
                    result = nlc_nn(
                        x=data_train,  # Training data for ticker_x
                        maxlag=self.max_lag,  # Maximum lag for Granger causality
                        NN_config=self.nn_config,  # Neural network configuration
                        NN_neurons=self.nn_neurons,  # Number of neurons and dropout rates
                        x_test=data_y,  # Testing data for ticker_y
                        run=1,  # Number of runs for the analysis
                        epochs_num=epochs,  # Number of epochs per training phase
                        learning_rate=learning_rate,  # Learning rate for each phase
                        batch_size_num=batch_size,  # Batch size during training
                        x_val=data_val,  # Validation data
                        verbose=True,  # Set to True to print verbose output during training
                        plot=False  # Set to False to disable plotting
                    )
                    
                    # Store the results (causality score and p-value) in the dictionary
                    self.results[(ticker_x, ticker_y)] = {
                        "causality_score": result['causality_score'],  # Causality score from the model
                        "p_value": result['p_value']  # p-value from the test indicating statistical significance
                    }

        return self.results

    def significant_causality_pairs(self, alpha=0.05):
        """
        Identifies ticker pairs with significant nonlinear Granger causality based on p-values.

        Args:
            alpha (float): The significance threshold for p-values, typically 0.05.

        Returns:
            list: A list of pairs (ticker_x, ticker_y) that exhibit significant nonlinear causality.
        """
        significant_pairs = []
        
        # Iterate through the results to check which pairs are significant
        for (ticker_x, ticker_y), result in self.results.items():
            if result['p_value'] < alpha:  # Check if the p-value is less than the significance threshold
                significant_pairs.append((ticker_x, ticker_y))
                print(f"{ticker_x} nonlinearly causes {ticker_y} with p-value {result['p_value']}")
        PickleHelper(significant_pairs).pickle_dump('Non_Linear_Granger_Casuality_Significants')
        return significant_pairs
