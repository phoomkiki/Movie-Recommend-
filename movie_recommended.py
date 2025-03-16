import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load filtered IMDb data
movies = pd.read_csv(r"Dataset\filtered_movies.csv", low_memory=False)

# Convert genres to lists
movies["genres_list"] = movies["genres"].apply(lambda x: x.split(","))
movies["numVotes"] = movies["numVotes"].astype(int)

# Get unique genres
unique_genres = sorted(set(genre for sublist in movies["genres_list"] for genre in sublist))

# Load PM2.5 data
def load_pm25_data():
    df = pd.read_csv(r"VPJ-main\Weather_chiangmai.csv", parse_dates=["Date"], dayfirst=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna()
    return df

# Prepare data for LSTM model
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

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([ 
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Cache trained LSTM model
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

    return model, scaler_X, scaler_y, feature_cols, df, mse

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
    st.title("📊 PM2.5 Forecasting using LSTM")
    model, scaler_X, scaler_y, feature_cols, df, mse = train_pm25_model()
    selected_date = st.date_input("📅 Select a date", datetime.date(2016, 7, 11))
    df_filtered = df[df["Date"] == pd.to_datetime(selected_date)]
    st.write("### Data Preview:")
    st.dataframe(df_filtered.head())
    
    if st.button("🔮 Predict PM2.5"):
        if not df_filtered.empty:
            X_selected = df_filtered[feature_cols].values  
            X_selected_scaled = scaler_X.transform(X_selected)  
            X_selected_scaled = np.expand_dims(X_selected_scaled, axis=0)
            y_pred_scaled = model.predict(X_selected_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            st.write(f"⛅ Predicted PM2.5 for {selected_date}: ")
            st.write(f"### {y_pred[0][0]:.2f}")
            st.write(f"🎯 Mean Squared Error: {mse:.2f}" )
            st.success("✅ Prediction Completed!")
        else:
            st.error("❌ ไม่มีข้อมูลในวันที่ที่เลือก กรุณาเลือกวันใหม่")

# Main function
def main(): 
    page = st.sidebar.selectbox("Select a page", ["Movie Recommender", "PM2.5 Forecasting"])  
    if page == "Movie Recommender":
        movie_recommender()
    elif page == "PM2.5 Forecasting":
        pm25_forecasting()

if __name__ == "__main__":
    main()
