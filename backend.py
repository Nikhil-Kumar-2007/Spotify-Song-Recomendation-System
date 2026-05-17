import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

from transformers import ColumnDropper, DurationConverter, FrequencyEncoder


df = pd.read_csv("dataset.csv")
# load all pipelines
pipeline = joblib.load('pipeline.joblib')
vectors = joblib.load('vectors.joblib')



def recommended_songs(input_song, n):
    input_df = pd.DataFrame([input_song])
    assigned_vector = pipeline[:-1].transform(input_df)
    assigned_cluster = pipeline.predict(input_df)[0]
    
    cluster_vectors = vectors[vectors['cluster'] == assigned_cluster].iloc[:, :-1]
    cluster_masks = cluster_vectors.select_dtypes(include='number')
    similarities = cosine_similarity(
        cluster_masks, assigned_vector
    )
    sim_scores = sorted(zip(cluster_masks.index, similarities),
        key = lambda x : x[1],
        reverse = True
    )[:n]
    indices = [ss[0] for ss in sim_scores]
    scores  = [round(s[1].item(), 5) for s in sim_scores]
    pred_songs = cluster_vectors.loc[indices, ['track_id', 'track_name', 'artists', 'track_genre', 'album_name']].copy()
    pred_songs['similarity'] = scores
    return pred_songs



def findRecomendation(song_name, n = 6):
    song_name = song_name.strip()
    matched_song = df[df['track_name'] == song_name]
    recomendation = pd.DataFrame()
    if matched_song.empty:
       return recomendation
    matched_song_dict = matched_song.to_dict(orient = 'records')
    matched_song = matched_song[['track_id', 'track_name', 'artists', 'track_genre', 'album_name']]
    for input_song in matched_song_dict:
       recomendation = pd.concat([recomendation, recommended_songs(input_song, n)])
    recomendation = recomendation[~recomendation['track_id'].isin(matched_song['track_id'])]
    recomendation = recomendation.sort_values(by = 'similarity', ascending = False)[:n]
    recomendation = recomendation.reset_index(drop = True)
    recomendation.index += 1
    return recomendation, matched_song.iloc[0]
