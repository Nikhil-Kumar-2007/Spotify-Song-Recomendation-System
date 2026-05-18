import streamlit as st
import pandas as pd
from backend import findRecomendation

# PAGE CONFIG 
st.set_page_config(page_title="Song Recommender", layout="wide")

# SIDEBAR NAVIGATION 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Song Recommendation", "About Page"])


# PAGE 1 : SONG RECOMMENDATION 

if page == "Song Recommendation":
    st.title("🎵 Song Recommendation System")
    st.subheader("Enter a Song Name")

    song_name = st.text_input("Song Name", placeholder="e.g. Shape of You")

    if st.button("Get Recommendations"):
        if not song_name.strip():
            st.warning("Please enter a song name.")
        else:
            result = findRecomendation(song_name)

            # returns empty DataFrame if song not found
            if isinstance(result, pd.DataFrame) and result.empty:
                st.error(f"❌ '{song_name}' is not in our database. Please try another song.")
            else:
                recommendations, matched_song = result

                # Given Song Details
                st.subheader("🎤 You Searched For")
                st.dataframe(
                    pd.DataFrame([matched_song])[['track_id', 'track_name', 'artists', 'track_genre', 'album_name']],
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # Recommended Songs
                st.subheader("✨ Recommended Songs")
                st.dataframe(
                    recommendations[['track_name', 'artists', 'track_genre', 'album_name', 'similarity']],
                    use_container_width=True,
                )

# PAGE 2 : ABOUT

elif page == "About Page":
    st.title("👨‍💻 About Me")
    st.markdown("""
    **Name:** Priyansi Yadav\n
    **GitHub ID:** https://github.com/priyanshiyadav03/
    """)
    st.markdown("""
    **Name:** Pranjali Singh\n
    **GitHub ID:** https://github.com/pranjali-t 
    """)
    st.markdown("""
    **Name:** Naitik Singh\n
    **GitHub ID:** https://github.com/Naitik152singh 
    """)
    st.markdown("""
    **Name:** Nikhil Kumar\n
    **GitHub ID:** https://github.com/Nikhil-Kumar-2007
    """)
    st.markdown("""
    **Project Repository:** https://github.com/Nikhil-Kumar-2007/Spotify-Song-Recomendation-System
    """)
