import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer
from sktime.transformers.compose import Tabulariser
from sktime.utils.validation.supervised import validate_X, check_X_is_univariate
from sktime.utils.matrix_profile import stomp_self


class MatrixProfile(BaseTransformer):
    """
        Takes as input a time series dataset and returns the matrix profile and
        index profile for each time series of the dataset.

        Example of use:
        # Xt = MatrixProfile(m).transform(X)
        X, a pandas DataFrame, is the the dataset.
        m, an integer, is the desired subsequence length to be used.
        Xt is the transformed X, i.e., a pandas DataFrame with the same number
        of rows as X, but each row has the matrix profile for the corresponding time series.
    """

    def __init__(self, subs_len):
        self.m = subs_len  # subsequence length

    def transform(self, X):
        """
            Takes as input a time series dataset and returns the matrix profile
            for each single time series of the dataset.

            Parameters
            ----------
                X: pandas.DataFrame
                   Time series dataset.

            Output
            ------
                Xt: pandas.DataFrame
                    Dataframe with the same number of rows as the input.
                    The number of columns equals the number of subsequences
                    of the desired length in each time series.
        """

        # Input checks
        validate_X(X)
        check_X_is_univariate(X)

        n_instances = X.shape[0]

        # Convert into tabular format
        tabulariser = Tabulariser()
        X = tabulariser.transform(X)

        n_subs = X.shape[1]-self.m+1

        Xt = pd.DataFrame(stomp_self(np.array([X.iloc[i]]), self.m) for i in range(0, n_instances))

        return Xt
