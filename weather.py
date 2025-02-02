from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle  # For saving and loading models
import os


def predict_previous_weather(bydate):
    # Load the data
    df = pd.read_csv('rainfall_final.csv')
    return df.loc[df['datetime'] == bydate.strftime('%d-%m-%Y')] 

def predict_weather(date_input):
    """
    Predicts multiple weather attributes for a specific date using combined LightGBM-Prophet models.
    
    Args:
        date_input (str): The date for prediction in the format 'DD-MM-YYYY'.
    
    Returns:
        dict: A dictionary containing predictions for each attribute and additional evaluation metrics.
    """
    attributes = ['rainfall', 'tempmax', 'humidity', 'windspeed']  # Attributes to model
    predictions = {}
    messages = {}

    try:
        # Convert input date to datetime format
        date_input = pd.to_datetime(date_input, format='%d-%m-%Y')

        # Load the dataset for reference
        df = pd.read_csv('rainfalll_final.csv')  # Update file name as needed
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')

        # Check for each attribute
        for attr in attributes:
            try:
                # Load the Prophet model
                with open(f'{attr}_prophet_model.pkl', 'rb') as f:
                    prophet_model = pickle.load(f)

                # Load the LightGBM model
                with open(f'{attr}_lightgbm_model.pkl', 'rb') as f:
                    best_lgb_model = pickle.load(f)

                # Check if the date is in the past or future
                if date_input in df['datetime'].values:
                    # If the date exists in the dataset, return the actual value
                    actual_value = df.loc[df['datetime'] == date_input, attr].values[0]
                    predictions[attr] = actual_value
                    messages[attr] = 'Date is within the dataset.'
                    continue

                # Determine the number of days to forecast into the future
                future_days = (date_input - df['datetime'].max()).days
                if future_days <= 0:
                    predictions[attr] = None
                    messages[attr] = 'Date is before the dataset range.'
                    continue

                # Forecast using Prophet
                future_df = prophet_model.make_future_dataframe(periods=future_days, freq='D')
                forecast = prophet_model.predict(future_df)

                # Extract Prophet's prediction for the input date
                if date_input in forecast['ds'].values:
                    predicted_prophet = forecast.loc[forecast['ds'] == date_input, 'yhat'].values[0]
                    trend_value = forecast.loc[forecast['ds'] == date_input, 'trend'].values[0]
                else:
                    predictions[attr] = None
                    messages[attr] = f'Prophet model failed to forecast the given date for {attr}.'
                    continue

                # Prepare features for LightGBM
                if len(df) >= 10:
                    recent_attr_sum = df[attr].rolling(window=10).sum().iloc[-1]
                else:
                    recent_attr_sum = df[attr].sum()

                X_future = pd.DataFrame({
                    'trend': [trend_value],
                    f'cumulative_{attr}_10d': [recent_attr_sum]
                })

                # Predict residual using LightGBM
                predicted_residual = best_lgb_model.predict(X_future)[0]

                # Combine Prophet and LightGBM predictions
                combined_prediction = predicted_prophet + predicted_residual
                predictions[attr] = max(combined_prediction, 0)  # Ensure non-negative prediction
                messages[attr] = 'Prediction successful.'

            except Exception as e:
                predictions[attr] = None
                messages[attr] = f'Error occurred for {attr}: {e}'

        return {'predictions': predictions, 'messages': messages}

    except Exception as e:
        return {'predictions': None, 'messages': f'General error occurred: {e}'}



def predict_previous_10_days_rainfall(
    date_input,
    file_name='rainfalll_final.csv',
    prophet_model_file='rainfall_prophet_model.pkl',
    lgb_model_file='rainfall_lightgbm_model.pkl'
):
    """
    Predicts the cumulative rainfall for the previous 10 days from the given date using Prophet and LightGBM models.
    
    Args:
        date_input (str): The date for prediction in the format 'DD-MM-YYYY'.
        file_name (str): The CSV file name containing rainfall data. Defaults to 'rainfall_final.csv'.
        prophet_model_file (str): The file name of the pre-trained Prophet model.
        lgb_model_file (str): The file name of the pre-trained LightGBM model.
    
    Returns:
        dict: A dictionary containing the predicted cumulative rainfall and a status message.
    """
    try:
        # Convert input date to datetime format
        date_input = pd.to_datetime(date_input, format='%d-%m-%Y')

        # Load the dataset
        df = pd.read_csv(file_name)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')

        # Load the pre-trained Prophet model
        with open(prophet_model_file, 'rb') as f:
            prophet_model = pickle.load(f)

        # Load the pre-trained LightGBM model
        with open(lgb_model_file, 'rb') as f:
            lgb_model = pickle.load(f)

        # Calculate the range for the previous 10 days
        start_date = date_input - timedelta(days=10)
        if start_date < df['datetime'].min():
            return {
                'cumulative_rainfall': None,
                'message': 'Insufficient data to predict the previous 10 days rainfall.'
            }

        # Generate future DataFrame for the Prophet model
        future_df = pd.date_range(start=start_date, end=date_input - timedelta(days=1), freq='D').to_frame(index=False, name='ds')

        # Predict rainfall using Prophet model
        forecast = prophet_model.predict(future_df)
        forecast['cumulative_rainfall_10d'] = forecast['yhat'].rolling(window=10).sum().shift(1).fillna(0)

        # Prepare features for LightGBM model
        X_future = forecast[['trend', 'cumulative_rainfall_10d']]

        # Predict residuals using LightGBM model
        residuals = lgb_model.predict(X_future)

        # Combine Prophet predictions and LightGBM residuals
        forecast['final_rainfall'] = forecast['yhat'] + residuals

        # Calculate cumulative rainfall for the 10 days
        cumulative_rainfall = forecast['final_rainfall'].sum()

        return {
            'cumulative_rainfall': max(cumulative_rainfall, 0),  # Ensure non-negative prediction
            'message': 'Predicted cumulative rainfall for the previous 10 days successfully.'
        }

    except Exception as e:
        return {
            'cumulative_rainfall': None,
            'message': f'An error occurred: {e}'
        }

