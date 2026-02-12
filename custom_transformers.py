from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


class MultiColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.mlbs = {col: MultiLabelBinarizer() for col in columns}

    def fit(self, X, y=None):
        for col in self.columns:
            self.mlbs[col].fit(X[col])
        return self

    def transform(self, X):
        transformed = [self.mlbs[col].transform(X[col]) for col in self.columns]
        return np.hstack(transformed)
