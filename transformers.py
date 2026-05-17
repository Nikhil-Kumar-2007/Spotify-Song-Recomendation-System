from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity



class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols):
        self.drop_cols = drop_cols
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        self.X = X.copy()
        self.common_cols = list(filter(lambda com_col : com_col in self.drop_cols, self.X.columns))
        return self.X.drop(columns = self.common_cols)

class DurationConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        self.X = X.copy()
        if 'duration_ms' not in self.X:
            return self.X
        def convert_ms_to_cat(ms):
            if ms <= 120000: return 1
            elif ms <= 360000: return 2
            elif ms <= 1200000: return 3
            elif ms <= 2400000: return 4
            else: return 5
        self.X['duration_ms'] = self.X['duration_ms'].apply(convert_ms_to_cat)
        return self.X    

class FrequencyEncoder(BaseEstimator, TransformerMixin): 
    def __init__(self, freq_cols):
        self.freq_cols = freq_cols
        self.freq_map = {}
    def fit(self, X, y = None):
        for col in self.freq_cols:
            self.freq_map[col] = X[col].value_counts()
        return self
    def transform(self, X, y = None):
        self.X = X.copy()
        self.active_cols = list(filter(lambda com_col : com_col in self.freq_cols, self.X.columns))
        for col in self.active_cols:
            self.X[f"{col}_pop"] = self.X[col].map(self.freq_map[col])
            self.X[f"{col}_pop"] = self.X[f"{col}_pop"].fillna(0.00001)
        self.X = self.X.drop(columns = self.active_cols)
        return self.X