import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error


from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model, Input,regularizers
import statistics


file_path = '/Users/ocalkaptan/Desktop/MasterTh/pythondf.csv'
dfload = pd.read_csv(file_path)






dfload = dfload.drop(dfload.columns[0], axis=1)
# Define a function to add '00:00:00' if time component is missing
def add_missing_time_component(date_str):
    if len(date_str) == 10:  # Assuming 'YYYY-MM-DD' format
        return date_str + ' 00:00:00'
    return date_str

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

# Convert these variables to dummy variables
#df_hour_dummies = pd.get_dummies(dfload['hour'], prefix='hour').astype(int)
#df_day_dummies = pd.get_dummies(dfload['day_of_week'], prefix='day_of_week').astype(int)
#df_month_dummies = pd.get_dummies(dfload['month'], prefix='month').astype(int)



# Generate lagged features for temperature
for lag in [192,336,504]:
    dfload[f'load_lag_{lag}'] = dfload['Load_DA'].shift(lag)

dfload["Rolling_Avg_15"] = dfload["Load_DA"].rolling(window=15*24).mean().shift(8*24)
dfload["Rolling_Std_15"] = dfload["Load_DA"].rolling(window=15*24).std().shift(8*24)
dfload["Rolling_Sum_15"] = dfload["Load_DA"].rolling(window=15*24).sum().shift(8*24)
dfload["Rolling_Avg_30"] = dfload["Load_DA"].rolling(window=30*24).mean().shift(8*24)
dfload["Rolling_Sum_30"] = dfload["Load_DA"].rolling(window=30*24).sum().shift(8*24)
# Calculate the moving standard deviation over the last 30 days, excluding the first 7 days
dfload["Rolling_Std_30"] = dfload["Load_DA"].rolling(window=30*24).std().shift(8*24)

for lag in range(1, 4):
    dfload[f'temp_lag_{lag}'] = dfload['temp_actual'].shift(lag)

# Generate rolling mean features for temperature
for window in [6,24]:
    dfload[f'temp_mean_{window}'] = dfload['temp_actual'].rolling(window=window).mean()

# Generate min and max features for temperature over the last 24 hours
 ##temp_min_24 =  dfload['temp_actual'].rolling(window=24).min()

 #dfload['temp_max_24'] =  dfload['temp_actual'].rolling(window=24).max()

# Generate lag features for temperature at 24, 48, and 96 hours
for lag in [24, 48]:
     dfload[f'temp_lag_{lag}'] =  dfload['temp_actual'].shift(lag)

# Generate lag features for wind speed
#for lag in range(1, 8):
 #    dfload[f'wspd_lag_{lag}'] =  dfload['wspd_actual'].shift(lag)

#for lag in range(1, 8):
 #   dfload[f'dwpt_lag_{lag}'] = dfload['dwpt_actual'].shift(lag)

#for lag in range(1, 8):
  #dfload[f'pres_lag_{lag}'] = dfload['pres_actual'].shift(lag)

#for window in [2,4,8,12,24]:
 #   dfload[f'dew_mean_{window}'] = dfload['dwpt_actual'].rolling(window=window).mean()
# Generate min and max features for temperature over the last 24 hours
dfload['temp_min'] =  dfload['temp_actual'].rolling(window=24).min()
dfload['temp_max'] =  dfload['temp_actual'].rolling(window=24).max()


# Assuming df is your DataFrame with relevant columns

# Wind Chill calculation
#def calculate_wind_chill(temp, wind_speed):
 #   # Only calculate wind chill if temperature < 10°C and wind speed > 4.8 km/h
  #  wind_chill = np.where(
   #     (temp < 10) & (wind_speed > 4.8),
    #    13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16),
     #   temp
    #)
    #return wind_chill

# Humidity calculation
#def calculate_humidity(temp, dew_point):
 #   humidity = 100 * (np.exp((17.625 * dew_point) / (dew_point + 243.04)) /
  #                    np.exp((17.625 * temp) / (temp + 243.04)))
   # return humidity

# Apply the calculations to the DataFrame
#dfload['Wind_Chill'] = calculate_wind_chill(dfload['temp_actual'], dfload['wspd_actual'])
#dfload['Humidity'] = calculate_humidity(dfload['temp_actual'], dfload['dwpt_actual'])


#for lag in range(1, 6):
   # dfload[f'Wind_Chill_L{lag}'] = dfload['Wind_Chill'].shift(lag)
 #   dfload[f'Humidity_L{lag}'] = dfload['Humidity'].shift(lag)

#for window in [4,8,12,24]:
    #dfload[f'Wind_Chill_M{window}'] = dfload['Wind_Chill'].rolling(window=window).mean()
   # dfload[f'Humidity_M{window}'] = dfload['Humidity'].rolling(window=window).mean()

#Example: "Climate Change and Energy Demand in France" by Hallegatte et al., which discusses the impact of temperature on energy demand and uses 18°C as a reference for HDD.
T_base = 18

# Calculate HDD and CDD
dfload['HDD'] = dfload['temp_actual'].apply(lambda x: max(0, T_base - x))

dfload['CDD'] = dfload['temp_actual'].apply(lambda x: max(0, x - T_base))
#for lag in range(1, 8):
    #dfload[f'HDD_L{lag}'] = dfload['HDD'].shift(lag)
   # dfload[f'CDD_L{lag}'] = dfload['CDD'].shift(lag)
#for window in [4,6,8,12,24]:
 #   dfload[f'HDD_M{window}'] = dfload['HDD'].rolling(window=window).mean()
  #  dfload[f'CDD_M{window}'] = dfload['CDD'].rolling(window=window).mean()

#dfload['dwpt_min'] =  dfload['dwpt_actual'].rolling(window=24).min()
#dfload['dwpt_max'] =  dfload['dwpt_actual'].rolling(window=24).max()
#dfload['pres_min'] =  dfload['pres_actual'].rolling(window=24).min()
#dfload['pres_max'] =  dfload['pres_actual'].rolling(window=24).max()
#fload['wspd_min'] =  dfload['wspd_actual'].rolling(window=24).min()
#dfload['wspd_max'] =  dfload['wspd_actual'].rolling(window=24).max()
#dfload['HDD_min'] =  dfload['HDD'].rolling(window=24).min()
#dfload['HDD_max'] =  dfload['HDD'].rolling(window=24).max()
#dfload['CDD_min'] =  dfload['CDD'].rolling(window=24).min()
#dfload['CDD_max'] =  dfload['CDD'].rolling(window=24).max()
#dfload['Wind_Chill_min'] =  dfload['Wind_Chill'].rolling(window=24).min()
#dfload['Wind_Chill_max'] =  dfload['Wind_Chill'].rolling(window=24).max()
#dfload['Humidity_min'] =  dfload['Humidity'].rolling(window=24).min()
#dfload['Humidity_max'] =  dfload['Humidity'].rolling(window=24).max()

# Combine original DataFrame with dummy variables
#f_combined = pd.concat([dfload, df_hour_dummies, df_day_dummies, df_month_dummies], axis=1)
#f_combined = pd.concat([dfload, df_day_dummies, df_month_dummies], axis=1)
#dfload['hourdm'] =pd.get_dummies(dfload['hour'], prefix='hour').astype(int)
#f_combined = pd.concat([dfload, df_day_dummies, df_month_dummies], axis=1)
dfx = pd.concat([dfload.drop(columns=['dwpt_actual','wspd_actual','pres_actual'])], axis=1)
#dfx = pd.concat([f_combined.drop(columns=['month','day_of_week','wspd_actual'])], axis=1)
window=slice(8785+720*4, 43825+720*4)#m
train_val_set = dfx[window]

test_set = dfx[43825+720*4:]
#features = train_val_set[features]#.drop(columns=['DateTime',	'Load_DA',	'Load_Act'])
#target = train_val_set['Load_DA'].values




    # Subset the features based on the selection
X_selected = train_val_set[['day_of_week','hour','month',
  'load_lag_504','Rolling_Avg_15','Rolling_Sum_15','Rolling_Avg_30','temp_mean_24',
'temp_lag_1','temp_lag_3','temp_lag_24',
  'temp_min','temp_max']]

target = train_val_set['Load_DA'].values

    # Normalize the selected features
scaler = RobustScaler()
#X_selected_scaled = scaler.fit_transform(X_selected)
#target_scaled = scaler.fit_transform(target.reshape(-1, 1))

    # Reshape the target variable to have the shape (number of samples, 192)
#y_reshaped = np.array([target_scaled[i:i+192] for i in range(0, len(target_scaled) - 192 + 1, 24)])

    # Adjust the input features accordingly
#X_selected_scaled = np.array([X_selected_scaled[i:i+192, :] for i in range(0, len(X_selected_scaled) - 192 + 1, 24)])
X_selected_scaled = scaler.fit_transform(X_selected)
target = train_val_set['Load_DA'].values.reshape(-1, 1)
target= scaler.fit_transform(target)
y_reshaped = np.array([target[i:i+192] for i in range(len(target) - 192)])

# Adjust the input features accordingly
X_selected_scaled = X_selected_scaled[192:]
#

    # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y_reshaped, test_size=0.25, random_state=42)


inputs = Input(shape=(X_train.shape[1],))#Input(shape=(X_train.shape[1], X_train.shape[2]))
ll = inputs#layers.Dropout(rate=best_params['dropout_rate'])(inputs)


hidden_layer1 = layers.Dense(best_params['neurons_1'],
                                    activation=best_params['activation_1'])(ll)


hidden_layer1 = LayerNormalization()(hidden_layer1)
hidden_layer2 = layers.Dense(best_params['neurons_2'],
                                    activation=best_params['activation_2'])(hidden_layer1)

    # Output layer
hidden_layer2 = LayerNormalization()(hidden_layer2)
outputs = layers.Dense(192,activation='linear')(hidden_layer2)

    # Define the model

model = Model(inputs=inputs, outputs=outputs)
    # Compile the model
optimizer =  tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='mean_absolute_error')
    # Train the mode
callbacks = [tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
model.fit(X_train, y_train, epochs=1000, callbacks=callbacks, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    # Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
loss


from tensorflow.keras.models import load_model

# Load the model from file
model.save('/Users/ocalkaptan/Desktop/MasterTh/shap5.h5')



import shap
import joblib
features_t = test_set.drop(columns=['DateTime',	'Load_DA',	'Load_Act'])

X_t = test_set[['day_of_week','hour','month',
  'load_lag_504','Rolling_Avg_15','Rolling_Sum_15','Rolling_Avg_30','temp_mean_24',
'temp_lag_1','temp_lag_3','temp_lag_24',
  'temp_min','temp_max']].values
scaler = RobustScaler()
X_s = scaler.fit_transform(X_t)
X_se = X_s[192:]
X_test= X_se

# Sample K data points from the training set for the background
K = 100  # Choose an appropriate number of samples (e.g., 100)
X_train_sampled = shap.sample(X_selected_scaled, K)

# Initialize the SHAP explainer with the sampled background data
explainer = shap.KernelExplainer(model.predict, X_train_sampled)

# Calculate SHAP values for the test set predictions
shap_values = explainer.shap_values(X_test[:192])


#  shap_values are computed from your SHAP explainer

# Save the SHAP values to a file
joblib.dump(shap_values, 'shap_values.joblib')


# Aggregate SHAP values across the 192 outputs by taking the mean
shap_values_aggregated = np.mean(shap_values, axis=2)

# Now plot the SHAP summary plot with the aggregated values
shap.summary_plot(shap_values_aggregated, X_test[:192],
                  feature_names=['day_of_week','hour','month',
  'load_lag_504','Rolling_Avg_15','Rolling_Sum_15','Rolling_Avg_30','temp_mean_24',
'temp_lag_1','temp_lag_3','temp_lag_24',
  'temp_min','temp_max'])
