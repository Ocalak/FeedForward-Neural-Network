import tensorflow as tf
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input, regularizers
import optuna

# Load the CSV file into a DataFrame
file_path = '/content/pythondf.csv'  # Update this path to your actual file
dfload = pd.read_csv(file_path)

# Preprocessing steps
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

# Extract hour of the day, day of the week, and month of the year
dfload['hour'] = dfload['DateTime'].dt.hour
dfload['day_of_week'] = dfload['DateTime'].dt.dayofweek  # Monday=0, Sunday=6
dfload['month'] = dfload['DateTime'].dt.month

# Generate lagged features for load
for lag in [192, 336, 504]:
    dfload[f'load_lag_{lag}'] = dfload['Load_DA'].shift(lag)

# Rolling statistics for Load_DA
dfload["Rolling_Avg_15"] = dfload["Load_DA"].rolling(window=15*24).mean().shift(8*24)
dfload["Rolling_Std_15"] = dfload["Load_DA"].rolling(window=15*24).std().shift(8*24)
dfload["Rolling_Sum_15"] = dfload["Load_DA"].rolling(window=15*24).sum().shift(8*24)
dfload["Rolling_Avg_30"] = dfload["Load_DA"].rolling(window=30*24).mean().shift(8*24)
dfload["Rolling_Sum_30"] = dfload["Load_DA"].rolling(window=30*24).sum().shift(8*24)
dfload["Rolling_Std_30"] = dfload["Load_DA"].rolling(window=30*24).std().shift(8*24)

# Generate lagged features for temperature
for lag in range(1, 4):
    dfload[f'temp_lag_{lag}'] = dfload['temp_actual'].shift(lag)

# Generate rolling mean features for temperature
for window in [6, 12, 24]:
    dfload[f'temp_mean_{window}'] = dfload['temp_actual'].rolling(window=window).mean()

# Generate lag features for temperature at 24, 48 hours
for lag in [24, 48]:
    dfload[f'temp_lag_{lag}'] = dfload['temp_actual'].shift(lag)

# Generate min and max features for temperature over the last 24 hours
dfload['temp_min'] = dfload['temp_actual'].rolling(window=24).min()
dfload['temp_max'] = dfload['temp_actual'].rolling(window=24).max()

# Calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD)
dfload['HDD'] = dfload['temp_actual'].apply(lambda x: max(0, 18 - x))
dfload['CDD'] = dfload['temp_actual'].apply(lambda x: max(0, x - 21))

# Drop unnecessary columns
dfx = dfload.drop(columns=['dwpt_actual', 'wspd_actual', 'pres_actual'])

# Define the window for training and validation data
#Window for month 01.08
window = slice(8785, 43825)#01.08.2019 to 01.08.2924 08:00
train_val_set = dfx[window]

# Remove rows with NaN values after feature engineering
train_val_set = train_val_set.dropna()

# Prepare features and target
features = train_val_set.drop(columns=['DateTime', 'Load_DA', 'Load_Act'])
target = train_val_set['Load_DA'].values

# Example Hp result# Here repelcaed with actual Hp from tuning.
best_params = {
    'dropout': True,
    'dropout_rate': 0.2,
    'regularize_h1_activation': True,
    'regularize_h1_kernel': False,
    'h1_activation_rate_l1': 0.001,
    'neurons_1': 64,
    'activation_1': 'relu',
    'regularize_h2_activation': False,
    'regularize_h2_kernel': False,
    'neurons_2': 32,
    'activation_2': 'relu',
    'learning_rate': 0.001,
    # Feature selection parameters (Update these based on best_params)
    'use_hour': True,
    'use_day_of_week': True,
    'use_month': True,
    'use_load_lag_192': True,
    'use_load_lag_336': True,
    'use_load_lag_504': False,
    'use_Rolling_Avg_15': True,
    'use_Rolling_Std_15': False,
    'use_Rolling_Sum_15': False,
    'use_Rolling_Avg_30': True,
    'use_Rolling_Sum_30': False,
    'use_Rolling_Std_30': False,
    'use_temp_lag_1': True,
    'use_temp_lag_2': False,
    'use_temp_lag_3': False,
    'use_temp_mean_6': True,
    'use_temp_mean_12': False,
    'use_temp_mean_24': False,
    'use_temp_lag_24': True,
    'use_temp_lag_48': False,
    'use_temp_min': True,
    'use_temp_max': True,
    'use_HDD': True,
    'use_CDD': True
}

# Select features based on best hyperparameters
selected_features = [feature for feature in features.columns if best_params.get(f'use_{feature}', False)]

# Subset the features
X_selected = features[selected_features].values

# Normalize features and target
scaler_X = RobustScaler()
X_selected_scaled = scaler_X.fit_transform(X_selected)

scaler_y = RobustScaler()
target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))

# Create sequences
sequence_length = 192
step = 24  # Assuming you want to create sequences every day (24 hours)

# Prepare input sequences
X_sequences = np.array([
    X_selected_scaled[i:i + sequence_length]
    for i in range(0, len(X_selected_scaled) - sequence_length + 1, step)
])

# Prepare output sequences
y_sequences = np.array([
    target_scaled[i:i + sequence_length].flatten()
    for i in range(0, len(target_scaled) - sequence_length + 1, step)
])
 X_selected_scaled = scaler.fit_transform(X_selected)
    target_scaled = scaler.fit_transform(target.reshape(-1, 1))

 #Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.25, random_state=42
)

# Define the model creation function
def create_model_from_params(params, input_shape):
    inputs = Input(shape=input_shape)
    x = inputs

    # Apply dropout if specified
    if params.get('dropout', False):
        rate = params['dropout_rate']
        x = layers.Dropout(rate)(x)

    # First hidden layer
    h1_units = params['neurons_1']
    h1_activation = params['activation_1']
    h1_kernel_regularizer = regularizers.L1(
        params['h1_kernel_rate_l1']) if params.get('regularize_h1_kernel', False) else None
    h1_activity_regularizer = regularizers.L1(
        params['h1_activation_rate_l1']) if params.get('regularize_h1_activation', False) else None

    x = layers.Dense(
        units=h1_units,
        activation=h1_activation,
        kernel_regularizer=h1_kernel_regularizer,
        activity_regularizer=h1_activity_regularizer
    )(x)
    x = layers.LayerNormalization()(x)

    # Second hidden layer
    h2_units = params['neurons_2']
    h2_activation = params['activation_2']
    h2_kernel_regularizer = regularizers.L1(
        params['h2_kernel_rate_l1']) if params.get('regularize_h2_kernel', False) else None
    h2_activity_regularizer = regularizers.L1(
        params['h2_activation_rate_l1']) if params.get('regularize_h2_activation', False) else None

    x = layers.Dense(
        units=h2_units,
        activation=h2_activation,
        kernel_regularizer=h2_kernel_regularizer,
        activity_regularizer=h2_activity_regularizer
    )(x)
    x = layers.LayerNormalization()(x)

    # Output layer
    outputs = layers.Dense(sequence_length, activation='linear')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    learning_rate = params['learning_rate']
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    return model

# Get input shape
input_dim=(X_train.shape[1], X_train.shape[2]
input_shape = Input(shape=(input_dim))

# Create the model
model = create_model_from_params(best_params, input_shape)

# Train the model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss (Mean Absolute Error): {loss}')

# Save the model
model.save('trained_model.h5')
