import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import streamlit as st
from datetime import timedelta

# Streamlit Title and Description
st.title("Stock Price Prediction Using RNN")
st.write("This app uses RNN to predict stock prices based on historical data and allows future price predictions.")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your stock prices CSV file", type=["csv"])

if uploaded_file is not None:
    # Reading the uploaded CSV file
    ford_df = pd.read_csv(uploaded_file)

    # Converting Date column into datetime and setting index
    ford_df['Date'] = pd.to_datetime(ford_df['Date'], format='%d-%m-%Y')
    ford_df.set_index('Date', inplace=True)

    # Display the data
    st.write("Preview of the dataset:")
    st.dataframe(ford_df.head())

    # Plotting the original data
    st.write("Historical Stock Prices:")
    st.line_chart(ford_df['Close'])

    # Prepare data for RNN
    data = ford_df['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create dataset function
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 10
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # RNN Model
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(50, return_sequences=False, input_shape=(time_step, 1)))
    model_rnn.add(Dense(1))
    model_rnn.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.write("Training the RNN model...")
    model_rnn.fit(X_train, y_train, epochs=10, batch_size=16)

    # Predict
    predicted_rnn = model_rnn.predict(X_test)
    predicted_rnn = scaler.inverse_transform(predicted_rnn)

    # Plot the predictions
    st.write("Predicted vs Actual Stock Prices:")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ford_df.index[-len(y_test):], ford_df['Close'].values[-len(y_test):], label='Actual Prices', color='blue')
    ax.plot(ford_df.index[-len(y_test):], predicted_rnn, label='RNN Predictions', color='orange')
    ax.set_title('RNN Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

    # Future Predictions
    st.write("Future Stock Price Prediction")
    future_date = st.date_input("Select a future date for prediction:")

    if future_date:
        # Generate future predictions
        last_time_steps = scaled_data[-time_step:]
        #future_steps = (future_date - ford_df.index[-1]).days
        future_steps = (pd.Timestamp(future_date) - ford_df.index[-1]).days


        # Predict step-by-step for future dates
        future_predictions = []
        for _ in range(future_steps):
            next_step = model_rnn.predict(last_time_steps.reshape(1, time_step, 1))
            future_predictions.append(next_step[0, 0])
            # Update last_time_steps for next prediction
            last_time_steps = np.append(last_time_steps[1:], next_step).reshape(-1, 1)

        # Inverse transform the predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create future dates
        future_dates = [ford_df.index[-1] + timedelta(days=i + 1) for i in range(future_steps)]

        # Display the future predictions
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})
        future_df.set_index('Date', inplace=True)
        st.write("Future Predictions:")
        st.dataframe(future_df)

        # Plot future predictions
        st.write("Future Predictions Plot:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ford_df.index, ford_df['Close'], label='Historical Prices', color='blue')
        ax.plot(future_df.index, future_df['Predicted Price'], label='Future Predictions', color='green')
        ax.set_title('Future Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)
