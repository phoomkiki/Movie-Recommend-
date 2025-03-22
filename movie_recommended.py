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
movies = pd.read_csv("Dataset\filtered_movies.csv", low_memory=False)

# Convert genres to lists
movies["genres_list"] = movies["genres"].apply(lambda x: x.split(","))
movies["numVotes"] = movies["numVotes"].astype(int)

# Get unique genres
unique_genres = sorted(set(genre for sublist in movies["genres_list"] for genre in sublist))

def doc():
    docm = '''
Data Used for Movie recomanded

The dataset from dataset.imdbws provides fascinating information related to movies. Different countries have varying preferences for popular movie genres. This database includes a wide range of details, such as movie titles, ratings, and the popularity of films across different nations. It can be used as a reference to randomly suggest an interesting movie for users to explore.

https://datasets.imdbws.com/

Genres – Used to indicate the category or type of a movie.
Average Rating – The score or rating of a movie.
Number of Votes – The number of people who voted, representing the reliability of the rating.

Problem Type

A Recommender System is a system used to suggest items that users might be interested in by utilizing available data, such as ratings, number of votes, viewing behavior, and movie information.
Using Models

KNN Content-Based Filtering recommends movies based on movie content information, such as genre, Number of votes


1. Import the necessary libraries
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors


# 2. Load and clean the data
movies = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False, dtype=str)
ratings = pd.read_csv("title.ratings.tsv", sep="\t", low_memory=False)

# Load title.basics.tsv and title.ratings.tsv
# Use dtype=str to treat all data as text before processing

movies = movies[movies["titleType"] == "movie"]
movies = movies[["tconst", "primaryTitle", "startYear", "genres"]]
movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")

# Filter only movies (titleType == "movie")
# Select relevant columns: tconst (movie ID), primaryTitle (title), startYear (release year), genres (movie genres)
# Convert startYear to a numeric value (invalid values will become NaN)

movies = movies.merge(ratings, on="tconst", how="left")
movies.dropna(subset=["genres", "averageRating", "numVotes"], inplace=True)
movies["numVotes"] = movies["numVotes"].astype(int)

# Merge movies with ratings using tconst as the key
# Remove rows with missing values (NaN) in genres, averageRating, and numVotes
# Convert numVotes to an integer for filtering purposes

# 3. Let the user select movie genres
movies["genres_list"] = movies["genres"].apply(lambda x: x.split(","))
unique_genres = sorted(set(genre for sublist in movies["genres_list"] for genre in sublist))

print("Select movie genres by number (comma-separated):")
for idx, genre in enumerate(unique_genres, start=1):
    print(f"{idx}. {genre}")

genre_input_idx = input("\nEnter the genre numbers (e.g., 1,3,5): ").strip().split(",")
selected_genres = [unique_genres[int(idx) - 1] for idx in genre_input_idx if idx.isdigit() and 1 <= int(idx) <= len(unique_genres)]

# Create a list of all available movie genres
# Display options for the user to select genres using numbers
# Convert user input numbers into actual genre names

# 4. Filter movies based on selected genres
filtered_movies = movies[
    (movies["genres_list"].apply(lambda x: all(g in x for g in selected_genres))) &
    (movies["averageRating"] >= 7.0) &
    (movies["numVotes"] > 10000)
]

# Select movies that contain all selected genres (all(g in x for g in selected_genres))
# Filter movies with a rating of 7.0 or higher
# Filter movies with more than 10,000 votes

# If no movies match the criteria:
if filtered_movies.empty:
    print("No movies found matching your selected genres with a rating above 7.0 and more than 10,000 votes.")
    exit()

# Notify the user if no movies match the criteria and terminate the program

# 5. Convert movie genres into numerical data
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(filtered_movies["genres_list"])

# Use MultiLabelBinarizer to convert genres into numerical data, e.g.:
# Action, Comedy -> [1, 0, 1, 0, 0, ...]
# Adventure -> [0, 1, 0, 0, 0, ...]
# This transformed data allows the model to analyze movie similarity

# 6. Use KNN to find similar movies
features = np.hstack((filtered_movies[["averageRating", "numVotes"]].values, genre_encoded))

knn = NearestNeighbors(n_neighbors=6, metric="cosine")










Data Used for PM 2.5
This dataset, sourced from Kaggle, provides information related to weather conditions in Chiang Mai, Thailand. It specifically includes PM2.5 data from 2016 to 2023, obtained from Air4Thai, along with other meteorological attributes from the Thai Meteorological Department. The dataset contains various weather indicators, such as the minimum and maximum atmospheric pressure recorded on a given day, the average temperature throughout the day, and other factors that reflect air quality and PM2.5 levels in Chiang Mai.

https://www.kaggle.com/datasets/natchaphonkamhaeng/pm-25-chiangmai-thailand

Date – The date of the recorded weather data in the format YYYY-MM-DD. 
Pressure_max – The maximum atmospheric pressure recorded on that day (measured in hPa). 
Pressure_min – The minimum atmospheric pressure recorded on that day (measured in hPa).
 Pressure_avg – The average atmospheric pressure throughout the day (measured in hPa). 
Temp_max – The highest temperature recorded on that day (measured in degrees Celsius).
 Temp_min – The lowest temperature recorded on that day (measured in degrees Celsius).
 Temp_avg – The average temperature throughout the day (measured in degrees Celsius). 
Humidity_max – The highest relative humidity recorded on that day (measured in percentage %).
 Humidity_min – The lowest relative humidity recorded on that day (measured in percentage %). 
Humidity_avg – The average relative humidity throughout the day (measured in percentage %). 
Precipitation – The total amount of rainfall recorded on that day (measured in millimeters, mm). 
Sunshine – The total duration of sunshine throughout the day (measured in hours). 
Evaporation – The amount of water evaporated from the surface (measured in millimeters, mm).
 Wind_direct – The dominant wind direction on that day (measured in degrees, where 0° = North, 90° = East, etc.).
 Wind_speed – The average wind speed recorded on that day (measured in meters per second, m/s). 
PM25 – The concentration of fine particulate matter (PM2.5) in the air (measured in micrograms per cubic meter, µg/m³). This is the target variable for prediction.

Problem Type

Regression Problem : predicts the PM2.5 level on a selected date based on historical data.


Using Models

Recurrent Neural Networks (RNN) or LSTM are suitable for time series data, as PM2.5 levels tend to change over time.




from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
import datetime

# 1. Load and preprocess data
def load_data():
    df = pd.read_csv("Weather_chiangmai.csv", parse_dates=["Date"], dayfirst=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna()  # Remove missing values
    return df

# 2. Prepare data for LSTM model
def prepare_data(df, feature_cols, target_col='PM25', lookback=24):
    data_X = df[feature_cols].values  # Extract feature columns
    data_y = df[[target_col]].values  # Extract target column (PM2.5)

    # Normalize data using MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    data_X_scaled = scaler_X.fit_transform(data_X)
    data_y_scaled = scaler_y.fit_transform(data_y)

    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(data_X_scaled) - lookback):
        X.append(data_X_scaled[i:i+lookback])  # Use past `lookback` hours as input
        y.append(data_y_scaled[i+lookback])  # Predict PM2.5 for the next time step
    
    return np.array(X), np.array(y), scaler_X, scaler_y

# 3. Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([ 
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)  # Output layer for predicting PM2.5
    ])
    model.compile(optimizer='adam', loss='mse')  # Compile model with Mean Squared Error loss
    return model

#  Use @st.cache_resource to ensure the model is trained only once
@st.cache_resource
def train_model():
    df = load_data()
    feature_cols = [
        'Pressure_max', 'Pressure_min', 'Pressure_avg', 'Temp_max', 'Temp_min', 'Temp_avg', 
        'Humidity_max', 'Humidity_min', 'Humidity_avg', 'Precipitation', 'Sunshine', 
        'Evaporation', 'Wind_direct', 'Wind_speed'
    ]

    # 4. Split data into training and testing sets
    X, y, scaler_X, scaler_y = prepare_data(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LSTM model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    # Evaluate model using Mean Squared Error
    mse = mean_squared_error(y_test, model.predict(X_test))

    return model, scaler_X, scaler_y, feature_cols, df, mse

# 5. Create Streamlit app for user interaction
def PM25():
    st.title(" PM2.5 Forecasting using LSTM")

    #  Load pre-trained model
    model, scaler_X, scaler_y, feature_cols, df, mse = train_model()

    #  User selects a date
    selected_date = st.date_input(" Select a date", datetime.date(2016, 7, 11))

    #  Filter data for the selected date
    df_filtered = df[df["Date"] == pd.to_datetime(selected_date)]
    st.write("### Data Preview:")
    st.dataframe(df_filtered.head())

    #  Predict button
    if st.button(" Predict PM2.5"):
        if not df_filtered.empty:
            X_selected = df_filtered[feature_cols].values  # Extract selected features
            X_selected_scaled = scaler_X.transform(X_selected)  # Normalize input
            X_selected_scaled = np.expand_dims(X_selected_scaled, axis=0)  # Reshape for LSTM

            y_pred_scaled = model.predict(X_selected_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Convert back to original scale

            #  Display prediction result
            st.write(f" Predicted PM2.5 for {selected_date}: ")
            st.write(f"### {y_pred[0][0]:.2f}")
            st.write(f" Mean Squared Error: {mse:.2f}")

            st.success(" Prediction Completed!")
        else:
            st.error(" No data available for the selected date. Please choose another date.")

# 6. Run Streamlit app
def main(): 
    page = st.sidebar.selectbox("Select a page", ["PM25"])  
    if page == "PM25":
        PM25() 

if __name__ == "__main__":
    main()



'''
    st.markdown(docm)
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
    if st.button("🎲 Random parameter valuesร์"):
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
    page = st.sidebar.selectbox("Select a page", ["Movie Recommender", "PM2.5 Forecasting","Document"])  
    if page == "Movie Recommender":
        movie_recommender()
    elif page == "PM2.5 Forecasting":
        pm25_forecasting()
    elif page == "Document":
        doc()

if __name__ == "__main__":
    main()
