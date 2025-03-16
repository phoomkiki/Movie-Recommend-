import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load filtered IMDb data
movies = pd.read_csv(r"C:\Users\NP30\Desktop\Movie_recommend\Dataset\filtered_movies.csv", low_memory=False)

# Convert genres to lists
movies["genres_list"] = movies["genres"].apply(lambda x: x.split(","))
movies["numVotes"] = movies["numVotes"].astype(int)

# Get unique genres
unique_genres = sorted(set(genre for sublist in movies["genres_list"] for genre in sublist))

# Streamlit UI
st.title("Movie Recommender System")

# Genre selection
genre_selection = st.multiselect("Select movie genres:", unique_genres)

if st.button("Recommend Movies"):
    # Filter movies by selected genres, rating, and votes
    filtered_movies = movies[
        (movies["genres_list"].apply(lambda x: all(g in x for g in genre_selection))) &
        (movies["averageRating"] >= 7.0) &
        (movies["numVotes"] > 10000)
    ]

    if filtered_movies.empty:
        st.warning(f"No movies found for genres {', '.join(genre_selection)} with a rating above 7.0 and more than 10,000 votes.")
    else:
        # Encode genres using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(filtered_movies["genres_list"])
        
        # Feature matrix including ratings and votes
        feature_matrix = np.hstack((genre_matrix, filtered_movies[["averageRating", "numVotes"]].values))
        
        # Train Nearest Neighbors model
        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        model.fit(feature_matrix)
        
        # Select reference movie (first filtered movie)
        selected_movie = filtered_movies.iloc[0]
        selected_index = 0
        
        # Find nearest neighbors
        distances, indices = model.kneighbors([feature_matrix[selected_index]])
        
        # Get recommended movies (excluding itself)
        recommended_movies = filtered_movies.iloc[indices[0][1:]]
        
        # Display selected movie
        st.subheader("Recommended Movies:")
        st.write(f"**Title:** {selected_movie['primaryTitle']} ({int(selected_movie['startYear'])})")
        st.write(f"**Genres:** {', '.join(selected_movie['genres_list'])}")
        st.write(f"**Rating:** {selected_movie['averageRating']}")
        st.write(f"**Votes:** {int(selected_movie['numVotes'])}")
        
        # Display recommended movies
        st.subheader("More movie recommendations :")
        for _, movie in recommended_movies.iterrows():
            st.write(f"- {movie['primaryTitle']} ({int(movie['startYear'])}) - {int(movie['numVotes'])} votes")
