import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
# import streamlit as st
# st.title('Stock Prediction Using CNN')

# Step 1: Data Retrieval
# user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download('AAPL', period="1d", interval="15m")

# Step 2: Data Preprocessing
data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Step 3: Data Normalization
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Step 4: Create Sequences for CNN
sequence_length = 10  # Adjust as needed
X, y = [], []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(1 if data[i+sequence_length][3] > data[i+sequence_length-1][3] else 0)

X = np.array(X)
y = np.array(y)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build CNN Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Step 8: Evaluate the Model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Accuracy: {accuracy*100:.2f}%")

y_test_pred = model.predict(X_test)

# Step 9: Visualize the difference between test and training
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Test Prices', color='green')
plt.plot(y_test_pred, label='Predicted Test Prices', color='blue')
plt.xlabel('Time')
plt.ylabel('Price Movement')
plt.title('Intraday Stock Prediction - Test vs. Predicted')
plt.legend()
plt.show()

# st.map(df)

