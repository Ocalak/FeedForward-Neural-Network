from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pytz
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model, Input, regularizers
import statistics
import shap
import joblib

# Assuming this only needs to be done once, not within the loop
os.listdir('/content/')

file_path = '/content/pythondf.csv'

#upload fitte model
loaded_model = load_model('model1.h5')# Save the entire model to a file

#upload weather forecasts
wea = pd.read_csv('/content/forecast8Aug.csv')
wea = wea[['DateTime','forecast','temp_forecast']]



# Define a function to add '00:00:00' if time component is missing
def add_missing_time_component(date_str):
    if len(date_str) == 10:  # Assuming 'YYYY-MM-DD' format
        return date_str + ' 00:00:00'
    return date_str

# Apply the function to the DateTime column
wea['DateTime'] = wea['DateTime'].apply(add_missing_time_component)
wea['forecast'] = wea['forecast'].apply(add_missing_time_component)
wea['DateTime'] = pd.to_datetime(wea['DateTime'], format='%Y-%m-%dT%H:%M:%SZ')

wea['forecast'] = pd.to_datetime(wea['forecast'], format='%Y-%m-%dT%H:%M:%SZ')
local_tz = pytz.timezone('Europe/Paris')

# Convert UTC times to local times
wea['DateTime_local'] = wea['DateTime'].dt.tz_localize('UTC').dt.tz_convert(local_tz)
wea['forecast_local'] = wea['forecast'].dt.tz_localize('UTC').dt.tz_convert(local_tz)

wea = wea[8*24*25*192:]
wea= wea.drop(columns=['DateTime','forecast'])

predictions_df = pd.DataFrame()

#update this everday for the new predictions
loop = 0

dfload = pd.read_csv(file_path)
dfload = dfload.drop(dfload.columns[0], axis=1)

    # Apply the function to the DateTime column
dfload['DateTime'] = dfload['DateTime'].apply(add_missing_time_component)

    # Now convert the DateTime column to datetime format
dfload['DateTime'] = pd.to_datetime(dfload['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Convert the datetime column to datetime objects
dfload['DateTime'] = pd.to_datetime(dfload['DateTime'])

    # Extract hour of the day, day of the week, and month of the year
dfload['hour'] = dfload['DateTime'].dt.hour
dfload['day_of_week'] = dfload['DateTime'].dt.dayofweek  # Monday=0, Sunday=6
dfload['month'] = dfload['DateTime'].dt.month

    # Generate lagged features for temperature
for lag in [192, 336, 504]:
 dfload[f'load_lag_{lag}'] = dfload['Load_DA'].shift(lag)

dfload["Rolling_Avg_15"] = dfload["Load_DA"].rolling(window=15 * 24).mean().shift(8 * 24)
dfload["Rolling_Std_15"] = dfload["Load_DA"].rolling(window=15 * 24).std().shift(8 * 24)
dfload["Rolling_Sum_15"] = dfload["Load_DA"].rolling(window=15 * 24).sum().shift(8 * 24)
dfload["Rolling_Avg_30"] = dfload["Load_DA"].rolling(window=30 * 24).mean().shift(8 * 24)
dfload["Rolling_Sum_30"] = dfload["Load_DA"].rolling(window=30 * 24).sum().shift(8 * 24)
dfload["Rolling_Std_30"] = dfload["Load_DA"].rolling(window=30 * 24).std().shift(8 * 24)

# Update temp_actual column with the temperature forecast values
dfload.iloc[(43825 + (24 * loop)):(43825 + 192 + (24 * loop)), dfload.columns.get_loc('temp_actual')] = wea['temp_forecast'].iloc[(56064 + (24 * loop) * 192):(56256 + (24 * loop) * 192)].values

for lag in range(1, 4):
 dfload[f'temp_lag_{lag}'] = dfload['temp_actual'].shift(lag)

    # Generate rolling mean features for temperature
for window in [6,12,24]:
 dfload[f'temp_mean_{window}'] = dfload['temp_actual'].rolling(window=window).mean()
for lag in [24, 48]:
 dfload[f'temp_lag_{lag}'] = dfload['temp_actual'].shift(lag)

dfload['temp_min'] = dfload['temp_actual'].rolling(window=24).min()
dfload['temp_max'] = dfload['temp_actual'].rolling(window=24).max()

    # Calculate HDD and CDD

dfload['HDD'] = dfload['temp_actual'].apply(lambda x: max(0, 18 - x))
dfload['CDD'] = dfload['temp_actual'].apply(lambda x: max(0, x - 21))

    # Combine original DataFrame with dummy variables
dfx = pd.concat([dfload.drop(columns=['dwpt_actual', 'wspd_actual', 'pres_actual'])], axis=1)

test_set = dfx[(43825 + 24 * loop):]
X_selected = test_set[['temp_actual',
    'day_of_week',
    'load_lag_336',
    'load_lag_504',
    'Rolling_Sum_15',
    'Rolling_Avg_30',
    'Rolling_Std_15',
    'temp_mean_6',
    'temp_lag_1',
    'temp_lag_48','temp_mean_24',
    'HDD',
    'CDD']]
target = test_set['Load_DA'].values

    # Normalize the selected features
scaler = RobustScaler()
X_selected_scaled = scaler.fit_transform(X_selected)
target_scaled = scaler.fit_transform(target.reshape(-1, 1))

    # Reshape the target variable to have the shape (number of samples, 192)
y_t = np.array([target_scaled[i:i + 192] for i in range(0, len(target_scaled) - 192 + 1, 24)])

    # Adjust the input features accordingly
X_t = np.array([X_selected_scaled[i:i + 192, :] for i in range(0, len(X_selected_scaled) - 192 + 1, 24)])

    # Make a prediction for the first sample in X_t
y_pred_single = model.predict(X_t[0:1])  # This will output shape (1, 192)

    # Rescale the prediction back to the original scale
y_pred_single_rescaled = scaler.inverse_transform(y_pred_single.reshape(-1, 192)[:, 0].reshape(-1, 1)).reshape(1, 192, 1)

    # Flatten the prediction to save in the DataFrame
y_pred_single_flat = y_pred_single_rescaled.flatten()  # Shape (192,)

#save_path = '/content/ens1.csv'
#predictions_df.to_csv(save_path, index=False)
# Calculate the MAE for the single prediction
mae_single = mean_absolute_error(y_true_single.flatten(), y_pred_single_rescaled.flatten())




#Shapp
# Sample K data points from the training set for the background
K = 100  
X_train_sampled = shap.sample(X_selected_scaled, K)

# Initialize the SHAP explainer with the sampled data
explainer = shap.KernelExplainer(model.predict, X_train_sampled)

# SHAP values for the test set predictions
shap_values = explainer.shap_values(X_test[:192])


# Save the SHAP values to a file
joblib.dump(shap_values, 'shap_values11.joblib')


# Aggregate SHAP values across the 192 outputs by taking the mean
shap_values_aggregated = np.mean(shap_values, axis=2)

#shap plot
shap.summary_plot(shap_values_aggregated, X_test[:192],
                  feature_names=['temp_actual','day_of_week','load_lag_336','load_lag_504','Rolling_Sum_15','Rolling_Avg_30','Rolling_Std_15','temp_mean_6','temp_lag_1','temp_lag_48','temp_mean_24','HDD','CDD'])

