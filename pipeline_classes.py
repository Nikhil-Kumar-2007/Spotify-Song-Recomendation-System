import numpy as np
import pandas as pd

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from transformers import ColumnDropper, DurationConverter, FrequencyEncoder


df = pd.read_csv("X_train.csv")


# # Columns to be encoded by frequency
freq_cols = [
    'album_name',
    'artists',
    'track_genre'
]

# columns to be deleted
drop_cols = [
    'track_id',
    'track_name',   
    'loudness',     
    'acousticness',
    'mode',
    'speechiness',
    'time_signature',
    'key',
    'explicit'
]


pipeline = Pipeline([
    ('drop_col', ColumnDropper(drop_cols = drop_cols)), 
    ('duration_converter', DurationConverter()),
    ('freq_encoder', FrequencyEncoder(freq_cols=freq_cols)),
    ('std_scaler', StandardScaler()),
    ('pca', PCA(
        n_components=10,
        svd_solver='randomized',
        whiten=True
    )),
    ("mbkmeans", MiniBatchKMeans(
        n_clusters=10,
        init='k-means++',
        batch_size=50,
        random_state=42
    ))
])


ndf = pipeline[:-1].fit_transform(df)
y_pred = pipeline.fit_predict(df)


ndf = pd.DataFrame(ndf)
ndf.insert(loc = 0, column = "track_id", value = df['track_id'])
ndf.insert(loc = 1, column = "track_name", value = df['track_name'])
ndf.insert(loc = 2, column = "track_genre", value = df['track_genre'])
ndf.insert(loc = 3, column = "artists", value = df['artists'])
ndf.insert(loc = 4, column = "album_name", value = df['album_name'])
ndf['cluster'] = y_pred


joblib.dump(ndf, "vectors.joblib")
joblib.dump(pipeline, "pipeline.joblib")