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

def doc1():
    st.markdown("""
    # 🎬 Movie Recommendation System
    ---
    
    ## 📊 Data Used for Movie Recommendation
    
    The dataset from [IMDb](https://datasets.imdbws.com/) provides fascinating information related to movies, including:
    - 🎭 **Genres** – The category or type of a movie.
    - ⭐ **Average Rating** – The score or rating of a movie.
    - 🗳️ **Number of Votes** – The number of people who voted, indicating the reliability of the rating.
    
    ### 🏆 Problem Type
    A **Recommender System** suggests movies that users might like based on data such as ratings, number of votes, and genres.
    
    ### 🧠 Using Models
    - **KNN Content-Based Filtering** recommends movies based on content similarity (e.g., genre, number of votes).
    
    ---
    ## 📌 Steps:
    
    ### 1️⃣ Import the necessary libraries
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.neighbors import NearestNeighbors
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    ```
    
    ### 2️⃣ Load and clean the data
    ```python
    movies = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False, dtype=str)
    ratings = pd.read_csv("title.ratings.tsv", sep="\t", low_memory=False)
    ```
    
    - Filters movies only (`titleType == "movie"`)
    - Merges movies with ratings
    - Filters movies with **rating ≥ 7.0** and **votes > 10,000**
    
    ---
    ### 3️⃣ Let users select genres
    ```python
    movies["genres_list"] = movies["genres"].apply(lambda x: x.split(","))
    ```
    - Users select genres from a **list of unique movie genres**.
    - Movies are filtered based on the selected genres.
    
    ---
    ### 4️⃣ Use KNN for recommendations
    ```python
    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    ```
    - **MultiLabelBinarizer** encodes genres as numerical data.
    - **KNN (Cosine Similarity)** finds similar movies.
    
    """)

def doc2():
    st.markdown("""
    # 🌫️ PM2.5 Prediction System
    ---
    
    ## 📊 Data Used for PM2.5 Prediction
    
    The dataset from [Kaggle](https://www.kaggle.com/datasets/natchaphonkamhaeng/pm-25-chiangmai-thailand) includes:
    - 📆 **Date** – The date of recorded weather data.
    - ⬆️ **Pressure_max** – Maximum atmospheric pressure (hPa).
    - 🌡️ **Temp_avg** – Average temperature (°C).
    - 💨 **Wind Speed** – Average wind speed (m/s).
    - 🌫️ **PM2.5** – Fine particulate matter concentration (µg/m³).
    
    ### 📌 Problem Type
    - **Regression Problem**: Predicts PM2.5 levels based on historical data.
    
    ### 🧠 Using Models
    - **LSTM (Long Short-Term Memory)** is used for time-series forecasting.
    
    ---
    ## 📌 Steps:
    
    ### 1️⃣ Import the necessary libraries
    ```python
    from sklearn.metrics import mean_squared_error
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    ```
    
    ### 2️⃣ Load and preprocess data
    ```python
    df = pd.read_csv("Weather_chiangmai.csv", parse_dates=["Date"], dayfirst=True)
    df = df.dropna()
    ```
    
    - **Missing values removed**
    - **Date formatted for time-series analysis**
    
    ---
    ### 3️⃣ Prepare data for LSTM model
    ```python
    X, y, scaler_X, scaler_y = prepare_data(df, feature_cols)
    ```
    - **Normalize data using MinMaxScaler**
    - **Create sequences for LSTM training**
    
    ---
    ### 4️⃣ Build and Train LSTM Model
    ```python
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    ```
    
    - **Two LSTM layers** for deep learning processing.
    - **Adam optimizer** with Mean Squared Error loss function.
    
    ---
    ### 5️⃣ Create a Streamlit Web App
    ```python
    st.title("PM2.5 Forecasting using LSTM")
    ```
    - Users select a **date** to predict PM2.5 levels.
    - Displays real-time **data visualization** and predictions.
    
    ---
    ### ✅ Prediction Example:
    ```python
    if st.button("Predict PM2.5"):
        y_pred_scaled = model.predict(X_selected_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        st.success(f"Predicted PM2.5: {y_pred[0][0]:.2f} µg/m³")
    ```
    - **Predicts PM2.5 levels** based on selected weather conditions.
    - **Displays the forecasted value** along with Mean Squared Error.
    
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
    page = st.sidebar.selectbox("Select a page", ["Document of Movie recommended","Document of PM2.5 Forecasting","Movie Recommender", "PM2.5 Forecasting"])  
    if page == "Movie Recommender":
        movie_recommender()
    elif page == "PM2.5 Forecasting":
        pm25_forecasting()
    elif page == "Document of Movie recommended":
        doc1()
    elif page == "Document of PM2.5 Forecasting":
        doc2()

if __name__ == "__main__":
    main()
