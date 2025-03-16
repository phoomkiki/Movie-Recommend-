import pandas as pd

# Load IMDb data
movies = pd.read_csv(r"C:\Users\NP30\Desktop\Movie_recommend\Dataset\title.basics.tsv", sep="\t", low_memory=False, dtype=str)
ratings = pd.read_csv(r"C:\Users\NP30\Desktop\Movie_recommend\Dataset\title.ratings.tsv", sep="\t", low_memory=False)

# Keep only movie-related data
movies = movies[movies["titleType"] == "movie"]
movies = movies[["tconst", "primaryTitle", "startYear", "genres"]]

# Merge with ratings
movies = movies.merge(ratings, on="tconst", how="left")

# Drop missing values
movies.dropna(subset=["genres", "averageRating", "numVotes"], inplace=True)

# Save filtered data to new files
movies.to_csv(r"C:\Users\NP30\Desktop\Movie_recommend\Dataset\filtered_movies.csv", index=False)
print("Filtered data saved successfully!")
