import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from typing import Union


class Estimator:

    def __init__(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, pd.DataFrame]):
        """
        Initialize the Estimator with data.

        Parameters:
        X (Union[pd.DataFrame, np.ndarray]): Independent variables
        y (Union[pd.Series, np.ndarray, pd.DataFrame]): Dependent variable
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
            self.X = X

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y = y.to_numpy()
        else:
            self.y = y

    def estimate_1NN(self):
        """
        Estimate using the 1NN method described by Devroye et al. (2018).

        Returns:
        float: The 1NN estimator value.
        """
        nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
        nn.fit(self.X)
        distances, indices = nn.kneighbors(self.X)
        NN = indices[:, 1]
        m_1 = self.y[NN]
        n = len(self.y)
        S = np.dot(self.y, m_1) / n
        EY = np.dot(self.y, self.y) / n
        L = EY - S
        return L

    def estimate_LS(self):
        """
        Estimate variance using OLS.

        Returns:
        float: The estimated variance.
        """
        X_const = sm.add_constant(self.X)  # Adds a constant term to the predictors
        model = sm.OLS(self.y, X_const).fit()
        rss = sum(model.resid ** 2)
        degrees_of_freedom = len(self.y) - model.df_model - 1  # Minus 1 for the intercept
        variance = rss / degrees_of_freedom
        return variance

    def estimate(self, method='1NN'):
        """
        General method to estimate based on the specified method.

        Parameters:
        method (str): The method to use for estimation ('1NN' or 'variance').

        Returns:
        float: The estimated value based on the specified method.
        """
        if method == '1NN':
            return self.estimate_1NN()
        elif method == 'LS':
            return self.estimate_LS()
        else:
            raise ValueError("Unsupported method. Use '1NN' or 'LS'.")