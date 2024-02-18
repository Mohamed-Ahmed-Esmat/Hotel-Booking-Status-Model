from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import datetime
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

# Z-score scaling function
def z_score_scaling(column):
    mean_val = column.mean()
    std_dev = column.std()
    scaled_column = (column - mean_val) / std_dev
    return scaled_column

def normalize_inputs(int_features):
    # Convert int_features to a numpy array
    features_array = np.array(int_features)

    # Apply Z-score scaling to each column
    scaled_features = np.apply_along_axis(z_score_scaling, axis=0, arr=features_array)

    return scaled_features


# Function to extract month, day, and year from the date and perform additional transformations
def process_date(date_str):
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    
    # Extracting month, day, and year
    month = date_obj.month
    day = date_obj.day
    year = date_obj.year
    
    # Encoding months using sine and cosine functions
    month_sine = np.sin(2 * np.pi * month / 12)
    month_cosine = np.cos(2 * np.pi * month / 12)
    day_sine = np.sin(2 * np.pi * day / 31)
    # Determine season
    if 3 <= month <= 5:
        season = 1  # Spring
    elif 6 <= month <= 8:
        season = 2  # Summer
    elif 9 <= month <= 11:
        season = 3  # Fall
    else:
        season = 4  # Winter
    
    return month_cosine, season, year

# Function to encode meal type
def encode_meal_type(meal_type):
    if meal_type == 'Meal Plan 1':
        return 1
    elif meal_type == 'Meal Plan 2':
        return 2
    elif meal_type == 'Meal Plan 3':
        return 3
    else:
        return 0

# Function to encode meal type
def encode_year(year):
    if year == 2015:
        return 0
    elif year == 2016:
        return 1
    elif year == 2017:
        return 2
    elif year == 2018:
        return 3
    else:
        return year - 2015

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Extracting input values from the form
    lead_time = float(request.form['lead time'])
    average_price = float(request.form['average price '])
    special_requests = float(request.form['special requests'])
    num_week_nights = float(request.form['number of week nights'])
    num_weekend_nights = float(request.form['number of weekend nights'])
    number_of_adults = float(request.form['number of adults'])

    # Extracting date input and processing it
    date_of_registration = request.form['date of registration']
    month_cosine, season, year = process_date(date_of_registration)

    # Handling radio button inputs for online market type and car parking space
    complementary_market_type = 1 if request.form.get('market type_Complementary') == 'Yes' else 0
    corporaye_market_type = 1 if request.form.get('market type_Corporate') == 'Yes' else 0
    online_market_type = 1 if request.form.get('market type_Online') == 'Yes' else 0
    repeated = 1 if request.form.get('repeated') == 'Yes' else 0
    car_parking_space = 1 if request.form.get('car_parking_space') == 'Yes' else 0
    # Encoding meal type
    encoded_year = encode_year(year)

    # Normalizing inputs using z-score
    normalized_features = normalize_inputs([
        lead_time, special_requests, num_weekend_nights,average_price,
        num_week_nights, number_of_adults
    ])

    # Flatten the normalized features array
    flattened_normalized_features = normalized_features.flatten()

    # Combine flattened normalized features with other features
    features = [
        encoded_year, month_cosine, season, online_market_type, corporaye_market_type, repeated, car_parking_space, complementary_market_type
    ]
    merged_features = np.concatenate((flattened_normalized_features, features), axis=None)

    # Manually print and label each feature
    feature_names =['lead time', 'special requests', 'number of weekend nights', 'average price ', 'number of week nights', 
                    'number of adults', 'year', 'month_cos', 'season', 'market type_Online', 'market type_Corporate', 'repeated', 
                    'car parking space', 'market type_Complementary']

    # Create a dictionary to hold the name of each feature with its corresponding value
    feature_dict = {feature_names[i]: merged_features[i] for i in range(len(feature_names))}

     # Create a DataFrame from the feature dictionary
    feature_df = pd.DataFrame([feature_dict])

    # Manually print and label each feature

    prediction = model.predict(feature_df)
    new_prediction = "Canceled" if prediction == 1 else "Not Canceled"
    print(f"Prediction: {prediction}")

    # Modify this line to include the necessary variables
    return render_template('index.html', prediction_text=f'The customer\'s booking status could be {new_prediction}', feature_dict=feature_dict)

if __name__ == "__main__":
    app.run(debug=True)
