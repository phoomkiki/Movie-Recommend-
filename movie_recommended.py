import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import datetime
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import random

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load filtered IMDb data
movies = pd.read_csv("Dataset/filtered_movies.csv", low_memory=False)

# Convert genres to lists
movies["genres_list"] = movies["genres"].apply(lambda x: x.split(","))
movies["numVotes"] = movies["numVotes"].astype(int)

# Get unique genres
unique_genres = sorted(set(genre for sublist in movies["genres_list"] for genre in sublist))

def movie_recommendation_doc():
    st.markdown("""
    # 🎬 Movie Recommendation System
    
    **Data Source:** [IMDb Datasets](https://datasets.imdbws.com/)
    
    This dataset provides information on movies, including:
    - **Genres** – Categories of movies
    - **Average Rating** – Movie rating score
    - **Number of Votes** – How many people voted for the movie
    
    ## 🎯 Problem Type
    - **Recommender System**: Suggests movies based on genre, rating, and popularity.
    
    ## 📌 Model Used
    - **KNN (Content-Based Filtering)**: Recommends movies based on their content (genre, votes, and rating).
    
    ---
    **Steps:**
    1. Load IMDb dataset (`title.basics.tsv`, `title.ratings.tsv`).
    2. Filter movies based on genre and rating.
    3. Encode genres using MultiLabelBinarizer.
    4. Use KNN to find similar movies.
    
    ---
    **Example Python Code:**
    ```python
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    ```
    
    """)

def pm25_prediction_doc():
    st.markdown("""
    # 🌿 PM2.5 Prediction using LSTM
    
    **Data Source:** [Kaggle - PM2.5 Chiang Mai](https://www.kaggle.com/datasets/natchaphonkamhaeng/pm-25-chiangmai-thailand)
    
    This dataset includes meteorological attributes (2016-2023) affecting air quality in Chiang Mai, such as:
    - **Temperature, Pressure, Humidity, Wind Speed**
    - **PM2.5 Concentration** (Target variable)
    
    ## 🎯 Problem Type
    - **Regression Problem**: Predict PM2.5 level on a given date based on historical data.
    
    ## 📌 Model Used
    - **LSTM (Long Short-Term Memory)**: Suitable for time series prediction.
    
    ---
    **Steps:**
    1. Load and preprocess data (`Weather_chiangmai.csv`).
    2. Normalize data using `MinMaxScaler`.
    3. Train LSTM model to predict PM2.5 levels.
    
    ---
    **Example Python Code:**
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(24, len(features))),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    ```
    
    """)
# st.write(docm):
# Load PM2.5 data
# ฟังก์ชันโหลดข้อมูล
def load_pm25_data():
    df = pd.read_csv("VPJ-main/Weather_chiangmai.csv", parse_dates=["Date"], dayfirst=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna()
    return df

# เตรียมข้อมูล
def prepare_pm25_data(df, feature_cols, target_col='PM25', lookback=24):
    data_X = df[feature_cols].values  
    data_y = df[[target_col]].values  

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    data_X_scaled = scaler_X.fit_transform(data_X)
    data_y_scaled = scaler_y.fit_transform(data_y)

    X, y = [], []
    for i in range(len(data_X_scaled) - lookback):
        X.append(data_X_scaled[i:i+lookback])  
        y.append(data_y_scaled[i+lookback])  

    return np.array(X), np.array(y), scaler_X, scaler_y

# สร้างโมเดล LSTM
def build_lstm_model(input_shape):
    model = Sequential([ 
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# เทรนและแคชโมเดล
@st.cache_resource
def train_pm25_model():
    df = load_pm25_data()
    feature_cols = [
        'Pressure_max', 'Pressure_min', 'Pressure_avg', 'Temp_max', 'Temp_min', 'Temp_avg', 
        'Humidity_max', 'Humidity_min', 'Humidity_avg', 'Precipitation', 'Sunshine', 
        'Evaporation', 'Wind_direct', 'Wind_speed'
    ]
    X, y, scaler_X, scaler_y = prepare_pm25_data(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    mse = mean_squared_error(y_test, model.predict(X_test))

    return model, scaler_X, scaler_y, feature_cols, mse

# Movie Recommendation Page
def movie_recommender():
    st.title("Movie Recommender System")
    
    genre_selection = st.multiselect("Select movie genres:", unique_genres)
    
    if st.button("Recommend Movies"):
        filtered_movies = movies[
            (movies["genres_list"].apply(lambda x: all(g in x for g in genre_selection))) &
            (movies["averageRating"] >= 7.0) &
            (movies["numVotes"] > 10000)
        ]
        
        if filtered_movies.empty:
            st.warning(f"No movies found for genres {', '.join(genre_selection)} with a rating above 7.0 and more than 10,000 votes.")
        else:
            mlb = MultiLabelBinarizer()
            genre_matrix = mlb.fit_transform(filtered_movies["genres_list"])
            feature_matrix = np.hstack((genre_matrix, filtered_movies[["averageRating", "numVotes"]].values))
            
            model = NearestNeighbors(n_neighbors=6, metric='cosine')
            model.fit(feature_matrix)
            
            selected_movie = filtered_movies.iloc[0]
            selected_index = 0
            distances, indices = model.kneighbors([feature_matrix[selected_index]])
            recommended_movies = filtered_movies.iloc[indices[0][1:]]
            
            st.subheader("Selected Movie:")
            st.write(f"**Title:** {selected_movie['primaryTitle']} ({int(selected_movie['startYear'])})")
            st.write(f"**Genres:** {', '.join(selected_movie['genres_list'])}")
            st.write(f"**Rating:** {selected_movie['averageRating']}")
            st.write(f"**Votes:** {int(selected_movie['numVotes'])}")
            
            st.subheader("Recommended Movies:")
            for _, movie in recommended_movies.iterrows():
                st.write(f"- {movie['primaryTitle']} ({int(movie['startYear'])}) - {int(movie['numVotes'])} votes")

# PM2.5 Prediction Page
def pm25_forecasting():
    st.title("PM2.5 Forecasting using LSTM")

    model, scaler_X, scaler_y, feature_cols, mse = train_pm25_model()

    st.write("### Enter meteorological variable values ​​(or press the button to randomize values))")

    # กำหนดช่วงค่าต่ำสุด-สูงสุดของแต่ละตัวแปร
    param_ranges = {
        'Pressure_max': (980, 1020), 'Pressure_min': (980, 1020), 'Pressure_avg': (980, 1020),
        'Temp_max': (10, 42), 'Temp_min': (5, 30), 'Temp_avg': (10, 35),
        'Humidity_max': (30, 100), 'Humidity_min': (10, 80), 'Humidity_avg': (20, 90),
        'Precipitation': (0, 100), 'Sunshine': (0, 12),
        'Evaporation': (0, 10), 'Wind_direct': (0, 360), 'Wind_speed': (0, 20)
    }

    # ตัวแปรเก็บค่าที่ผู้ใช้ใส่
    user_input = {}

    # สร้างปุ่มสุ่มค่า
    if st.button("🎲 Random parameter values"):
        for col, (min_val, max_val) in param_ranges.items():
            user_input[col] = round(random.uniform(min_val, max_val), 2)
    else:
        for col, (min_val, max_val) in param_ranges.items():
            user_input[col] = st.number_input(
                f"{col} ({min_val}-{max_val})", 
                min_value=float(min_val),  
                max_value=float(max_val),  
                value=float((min_val + max_val) / 2),  
                step=0.1  
            )
    if st.button("🔍 Predict PM2.5"):
        X_selected = np.array([list(user_input.values())])  
        X_selected_scaled = scaler_X.transform(X_selected)  
        X_selected_scaled = np.expand_dims(X_selected_scaled, axis=0)  

        y_pred_scaled = model.predict(X_selected_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        st.write(f"### 🎯 Predicted PM2.5: {y_pred[0][0]:.2f} µg/m³")
        st.write(f"📉 Mean Squared Error: {mse:.2f}")
        st.success("✅ Prediction Completed!")



# Main function
def main(): 
    page = st.sidebar.selectbox("Select a page", ["Movie Recommender", "PM2.5 Forecasting","Document of Movie recommended","Document of PM2.5 Forecasting"])  
    if page == "Movie Recommender":
        movie_recommender()
    elif page == "PM2.5 Forecasting":
        pm25_forecasting()
    elif page == "Document of Movie recommended":
        movie_recommendation_doc()
    elif page == "Document of PM2.5 Forecasting":
        pm25_prediction_doc()

if __name__ == "__main__":
    main()
