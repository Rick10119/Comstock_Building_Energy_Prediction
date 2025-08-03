import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import optimizers
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import concurrent.futures
import s3fs
import logging
from tensorflow.keras import callbacks


upgrade_total_data_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/HVAC_dissagregation_selected.xlsx')
upgrade_total_data_df = upgrade_total_data_df.drop(columns=['Unnamed: 0','out.electricity.total.energy_consumption.kwh'])

X = upgrade_total_data_df.drop(columns=['timestamp','out.electricity.cooling.energy_consumption.kwh'])#'season'
y = upgrade_total_data_df['out.electricity.cooling.energy_consumption.kwh']
print(X.columns)



#Replace NAN values with data mean and make all values double
y = y.fillna(y.mean())
X = X.select_dtypes(include=[np.number])
X = X.values.astype('double')
y = y.values.astype('double')

#Replace NAN values with data mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print(X.shape)
print(y.shape)
#Remove bottom 5% outliers
X = X[y > np.percentile(y, 5)]
y = y[y > np.percentile(y, 5)]


# 5. Create sequences (e.g., 24-hour windows, 1 datapoint is 15 minutes)
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i:i+seq_length])
    return np.array(Xs), np.array(ys)

# def create_sequences(X, y, seq_length=96):
#     Xs, ys = [], []
#     for i in range(len(X) - seq_length + 1):
#         Xs.append(X[i:i+seq_length])
#         # Predict only CENTER point (sequence-to-point)
#         center_index = i + seq_length // 2
#         ys.append(y[center_index])  # Single value per sequence
#     return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, seq_length=96)
print(X_seq.shape[2])

# 6. Reshape for Conv2D: (samples, height, width, channels)
# Here, height=96 (timesteps), width=2 (features), channels=1
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], X_seq.shape[2], 1))

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(X_test.shape)

print(X_train.shape)
# 8. Build CNN model
model = Sequential([
    Conv2D(128, (2, 2),padding='valid', input_shape=(96, 2, 1)), #64,32
    BatchNormalization(),
    Activation('relu'),

    Conv2D(256, (2, 1),padding='valid', input_shape=(96, 2, 1)), #128
    BatchNormalization(),
    Activation('relu'),

    MaxPooling2D((2, 1)),
    Flatten(),
    Dropout(0.2),
    Dense(256, activation='relu'), #128 ori
    Dense(96)  # Output:96 HVAC values per window
])

model.compile(optimizer= optimizers.Adam(learning_rate=0.005), loss='mse', metrics=['mae'])


# Learning rate scheduling and early stopping
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Update model training
history = model.fit(X_train, y_train, epochs=30, batch_size= 64, validation_split=0.2,callbacks=[lr_scheduler, early_stop])

# 10. Predict and evaluate
# 评估模型
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train.flatten(), y_train_pred.flatten()))
test_rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_test_pred.flatten()))

train_nrmse = train_rmse/np.mean(y_train.flatten())*100
test_nrmse = train_rmse/np.mean(y_test.flatten())*100

print(f'Train RMSE: {train_nrmse:.2f} %')
print(f'Test RMSE: {test_nrmse:.2f} %')


# 11. Plot example
import matplotlib.pyplot as plt
plt.plot(X_test[0,:24,0], label='Base Load', linestyle='--')
plt.plot(y_test[0,:24], label='Actual HVAC',linewidth=2)
plt.plot(y_test_pred[0,:24], label='Predicted HVAC',linewidth=2)
plt.xlabel('Time (h)')
plt.ylabel('Electricity Load (kwh)')
plt.title('HVAC Load Disaggregation Graph (24 hours)')
plt.legend()
plt.grid(True)
plt.show()