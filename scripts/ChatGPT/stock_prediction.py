#!/usr/bin/python

import csv
import os
import sys
import time
import sim_logging
import stock_constants
import math
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dot, Input, Dense, Bidirectional, Dropout, LSTM, Conv1D
from keras.layers import MaxPooling1D, Flatten, GaussianNoise, Activation
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

class STOCK:
    def __init__(self, i_AI_model_name, i_values_list):
        self.model_name = i_AI_model_name
        self.rmse = i_values_list[0]
        self.sharpe_ratio = i_values_list[1]
        self.next_day_price = i_values_list[2]
        self.percentage_change = None

        #if self.next_day_price:
        #    self.percentage_change = (((self.next_day_price-i_currentPrice)/i_currentPrice) * 100)
        #else:
        #    self.percentage_change = None


class STOCK_PREDICTION:
    def __init__(self, i_simlog, i_stock, i_df):
        self.simlog = i_simlog
        self.stock = i_stock
        self.df = i_df
        self.master_list = []
        self.action = self.stock_prediction()

    def get_AI_data(self):
        i_master_list = []
        log_master_filename = "MasterList_AI_model.csv"
        date = str(datetime.date.today())
        log_master_filename = os.path.abspath(os.path.dirname(sys.argv[0])).split('robinhood')[0] + \
                       "robinhood/logs//" + date + "/" + log_master_filename

        # Create a new file if the file doesn't already exists
        if not os.path.isfile(log_master_filename):
            # Create directories in the path if they don't exist
            os.makedirs(os.path.dirname(log_master_filename), exist_ok=True)
            # If the file doesn't exist, create it and append to it
            with open(log_master_filename, 'a') as file:
                file.write("tickerName,modelName,RMSE,sharpeRatio,expectedValue\n")

        #Now check if the tickerName exists in that file or not
        if len(getTicker(log_master_filename, self.stock)) == 0:
            self.master_list.insert(-1, STOCK('Attention-based LSTM', self.AttLSTM()))
            self.master_list.insert(-1, STOCK('BiDirectional LSTM', self.BiDirectionalLSTM()))
            self.master_list.insert(-1, STOCK('LSTM', self.LSTM()))
            self.master_list.insert(-1, STOCK('RNN', self.RNN()))
            self.master_list.insert(-1, STOCK('ANN', self.ANN()))
            self.master_list.insert(-1, STOCK('CNN', self.CNN()))
            self.master_list.insert(-1, STOCK('Random Forrest', self.RandomForest()))

            with open(log_master_filename, 'a') as file:
                for i in range(len(self.master_list)):
                    file.write(str(self.stock) + ",")
                    file.write(self.master_list[i].model_name + ",")
                    file.write(str(self.master_list[i].rmse) + ",")
                    file.write(str(self.master_list[i].sharpe_ratio) + ",")
                    file.write(str(self.master_list[i].next_day_price) + "\n")

        # Now read the csv file to get stock model data
        i_master_list = getTicker(log_master_filename, self.stock)

        return i_master_list


    def stock_prediction(self):
        # This will be used to determine if the AI models are worth depending on
        i_score_sell = 0
        i_score_buy  = 0

        # Get data on the ticker
        try:
            tickerData = yf.Ticker(self.stock); i_currentPrice = tickerData.history(period='1d')['Close'][0]
        except IndexError as e:
            self.simlog.error("Error found while trying to get current stock price for " + str(self.stock))
            self.simlog.error(str(e))
            return stock_constants.STOCK_LEAVE

        # We either need to pull the data from a csv file or recalculate the AI model data
        self.master_list = self.get_AI_data()

        self.simlog.info("AI model result for stock:  " + str(self.stock))
        self.simlog.info("Current Price = $" + str(i_currentPrice))
        for i in range(len(self.master_list)):

            # Calculate the percentage change in price
            i_percentage_change = None
            x = self.master_list[i]['expectedValue']
            if not self.master_list[i]['expectedValue'] == 'None':
                i_percentage_change = (((float(self.master_list[i]['expectedValue']) - i_currentPrice) / i_currentPrice) * 100)

            self.simlog.info("\nmodel_name = " + str(self.master_list[i]['modelName']))
            self.simlog.info("RMSE= " + str(self.master_list[i]['RMSE']))
            self.simlog.info("sharpeRatio= " + str(self.master_list[i]['sharpeRatio']))
            self.simlog.info("Next Day expectedValue = $" + str(self.master_list[i]['expectedValue']))
            self.simlog.info("Percentage Change = " + str(i_percentage_change))
            if float(self.master_list[i]['RMSE']) < 60:
                if i_percentage_change:
                    if i_percentage_change > 10:
                        i_score_buy += 1
                    elif i_percentage_change < 7:
                        i_score_sell += 1
                    else:
                        continue

        if i_score_sell >= 3:
            self.simlog.info("The current action is to SELL")
            return stock_constants.STOCK_SELL
        elif i_score_buy >= 5:
            self.simlog.info("The current action is to BUY")
            return stock_constants.STOCK_BUY
        else:
            self.simlog.info("The current action is to do nothing")
            return stock_constants.STOCK_LEAVE

    # Recurrent Neural Networks (RNNs)
    def RNN(self):

        # THis is the number of days the prediction is based on
        seq_len = 30

        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)

        # Preprocess the data
        data = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Split the data into training and testing sets
        training_data_len = math.ceil(len(scaled_data) * .9)
        train_data = scaled_data[0:training_data_len, :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - seq_len:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the RNN model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        model.add(tf.keras.layers.Dense(25))
        model.add(tf.keras.layers.Dense(1))

        # Train the RNN model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)

        # Test the RNN model
        test_data = scaled_data[training_data_len - seq_len:, :]
        x_test = []
        y_test = data[training_data_len:, :]

        for i in range(seq_len, len(test_data)):
            x_test.append(test_data[i - seq_len:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions_scaled = model.predict(x_test, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)

        # Calculate the RMSE. Both formulae generates the same result
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

        # Calculate the Sharpe ratio
        mean_return = self.df['Close'].pct_change().mean()
        volatility = self.df['Close'].pct_change().std()
        sharpe_ratio_actual = (mean_return/volatility)

        df_predicted = pd.DataFrame(predictions)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the stock price for next day
        last_days = scaler.fit_transform(df.tail(seq_len)['Close'].values.reshape(-1, 1))
        next_day = model.predict(np.array([last_days]), verbose=0)
        next_day = scaler.inverse_transform(next_day)[0][0]

        return [rmse, sharpe_ratio_predicted, next_day]

    # Artifical Neural Network
    def ANN(self):
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)

        # THis is the number of days the prediction is based on
        seq_len = 30

        # Prepare the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df['Close'].values.reshape(-1, 1))

        # Split the data into training and testing sets
        training_data_len = int(len(scaled_data) * 0.9)
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len:, :]

        # Define the training data
        X_train = []
        y_train = []
        for i in range(seq_len, len(train_data)):
            X_train.append(train_data[i - seq_len:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # Test the model
        X_test = []
        y_test = []
        for i in range(seq_len, len(test_data)):
            X_test.append(test_data[i - seq_len:i, 0])
            y_test.append(test_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price_scaled = model.predict(X_test, verbose=0)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price_scaled)

        # Calculate the root mean squared error
        #rmse = math.sqrt(mean_squared_error(y_test, predicted_stock_price_scaled)) #Shouldn't use scaled prices
        rmse = math.sqrt(mean_squared_error(df['Close'][training_data_len + seq_len:], predicted_stock_price))

        # Calculate the Sharpe ratio based on the Actual prediction
        mean_return = df['Close'][training_data_len + seq_len:].pct_change().mean()
        volatility = df['Close'][training_data_len + seq_len:].pct_change().std()
        sharpe_ratio_actual = (mean_return / volatility)

        df_predicted = pd.DataFrame(predicted_stock_price)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the stock price for next day
        last_days = scaler.fit_transform(df.tail(seq_len)['Close'].values.reshape(-1, 1))
        next_day = model.predict(np.array([last_days]))
        next_day = scaler.inverse_transform(next_day)[0][0]

        # Visualize the results
        #plt.plot(df['Close'][training_data_len:])
        #plt.plot(df['Close'][training_data_len + seq_len:].index, predicted_stock_price)
        #plt.legend(['Actual', 'Predicted'])
        #plt.show()
        return [rmse, sharpe_ratio_predicted, next_day]

    def BiDirectionalLSTM(self):

        # `look_back` is the number of previous time steps to use as input to the LSTM network
        # (e.g. 1 for using only the previous day's price)
        look_back = 30

        # Create a copy of df to prevent overwrite
        dataset = self.df.copy(deep=True)

        # Normalize the data
        dataset = dataset.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # Split into training and testing sets
        train_size = int(len(dataset) * 0.9)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        trainX, trainY = create_dataset_LSTM(train, look_back)
        testX, testY = create_dataset_LSTM(test, look_back)

        # Reshape data for LSTM input (samples, time steps, features)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Create the Deep Belief Network model
        model = Sequential()
        model.add(Bidirectional(LSTM(128), input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=0)

        # Make predictions on testing set
        testPredict = model.predict(testX, verbose=0)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Calculate root mean squared error
        rmse = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

        df_predicted = pd.DataFrame(testPredict)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the next days price
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)
        last_days = df.tail(look_back)
        last_days = scaler.fit_transform(last_days.filter(['Close']).values)
        last_days = np.reshape(last_days, (1, 1, last_days.shape[0]))
        next_day = model.predict(last_days, verbose=0)
        next_day = scaler.inverse_transform(next_day)[0][0]

        return [rmse, sharpe_ratio_predicted, next_day]


    def RandomForest(self):
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)

        # Split into training and testing sets
        X = df.drop('Close', axis=1); X = X.drop('Date', axis=1)
        y = df['Close']
        test_size = math.ceil(len(df) * .9)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0)
        rf.fit(X_train, y_train)

        # Make predictions on testing set
        y_pred = rf.predict(X_test)

        # Calculate root mean squared error
        rmse = mean_squared_error(y_test, y_pred)

        # Calculate the Sharpe ratio based on the Actual prediction
        mean_return = df['Close'][test_size:].pct_change().mean()
        volatility = df['Close'][test_size:].pct_change().std()
        sharpe_ratio_actual = (mean_return / volatility)

        df_predicted = pd.DataFrame(y_pred)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the stock price for next day
        next_day = None

        return [rmse, sharpe_ratio_predicted, next_day]

    def LSTM(self):

        # `look_back` is the number of previous time steps to use as input to the LSTM network
        # (e.g. 1 for using only the previous day's price)
        look_back = 30

        # Create a copy of df to prevent overwrite
        dataset = self.df.copy(deep=True)

        # Normalize the data
        dataset = dataset.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # Split into training and testing sets
        train_size = int(len(dataset) * 0.9)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        trainX, trainY = create_dataset_LSTM(train, look_back)
        testX, testY = create_dataset_LSTM(test, look_back)

        # Reshape data for LSTM input (samples, time steps, features)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Create LSTM model
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=0)

        # Make predictions on testing set
        testPredict = model.predict(testX, verbose=0)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Calculate root mean squared error
        rmse = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

        df_predicted = pd.DataFrame(testPredict)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the next days price
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)
        last_days = df.tail(look_back)
        last_days = scaler.fit_transform(last_days.filter(['Close']).values)
        last_days = np.reshape(last_days, (1, 1, last_days.shape[0]))
        next_day = model.predict(last_days, verbose=0)
        next_day = scaler.inverse_transform(next_day)[0][0]

        return [rmse, sharpe_ratio_predicted, next_day]


    # Convolutional Neural Networks (CNNs): CNNs are a type of neural network that are
    # commonly used for image recognition tasks. They have also been applied to stock 
    # price prediction by treating historical stock prices as a type of image. 
    # The CNN can then learn patterns and trends in the stock prices over time to make predictions.
    def CNN(self):

        seq_len = 30

        # Create a copy of df to prevent overwrite
        data = self.df.copy(deep=True)
        data = data.drop(['Date'], axis=1); data = data.drop(['Volume'], axis=1)

        # Use last seq_len days of data to predict next day's closing price
        X = []
        y = []
        for i in range(seq_len, len(data)):
            X.append(data.iloc[i - seq_len:i, 1:].values)
            y.append(data.iloc[i, -1])
        X = np.array(X)
        y = np.array(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        # Define CNN model
        model = Sequential()
        model.add(
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        # Compile model
        model.compile(loss='mse', optimizer='adam')

        # Train model
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        predictions = model.predict(X_test, verbose=0)

        # Calculate the RMSE. Both formulae generates the same result
        rmse = math.sqrt(mean_squared_error(y_test, predictions))

        # Calculate the Sharpe ratio
        df_predicted = pd.DataFrame(predictions)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)


        # Use last 30 days of data to make prediction
        last_days = data.iloc[-seq_len:, 1:].values
        last_days = np.reshape(last_days, (1, seq_len, last_days.shape[1]))
        next_day = model.predict(last_days, verbose=0)[0][0]

        return rmse, sharpe_ratio_predicted, next_day

    # Reinforcement Learning (RL): RL is a type of machine learning that focuses on
    # decision-making in dynamic environments. RL has been applied to stock price prediction
    # by training an agent to make decisions about buying and selling stocks based on historical
    # stock prices and other market indicators.
    def AttLSTM(self):
        data = self.df.copy(deep=True)
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Split data into training and testing sets
        train_size = int(len(scaled_prices) * 0.8)
        train_data = scaled_prices[:train_size]
        test_data = scaled_prices[train_size:]

        # Define constants
        window_size = 30  # Number of previous days' prices to consider for prediction
        hidden_units = 32
        output_size = 1
        learning_rate = 0.001
        epochs = 50
        batch_size = 32

        x_train, y_train = create_sequences(train_data, window_size)
        x_test, y_test = create_sequences(test_data, window_size)

        # Build the Attention-based LSTM model
        input_seq = Input(shape=(window_size, 1))
        lstm_out = LSTM(hidden_units, return_sequences=True)(input_seq)
        attention_weights = Dense(1, activation='tanh')(lstm_out)
        attention_weights = Activation('softmax')(attention_weights)
        context = Dot(axes=1)([attention_weights, lstm_out])
        context = Flatten()(context)
        output = Dense(output_size)(context)
        model = Model(inputs=input_seq, outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Make predictions on the test set
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Calculate Sharpe ratio (assuming daily returns)
        daily_returns = (y_test[1:] - y_test[:-1]) / y_test[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)

        # Predict the next days price
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)
        last_days = df.tail(window_size)
        last_days = scaler.fit_transform(last_days.filter(['Close']).values.reshape(-1,1))

        last_days = np.reshape(last_days, (1, last_days.shape[0], 1))
        next_day = model.predict(last_days, verbose=0)
        next_day = scaler.inverse_transform(next_day)[0][0]

        return [rmse, sharpe_ratio, next_day]




# Create input sequences and target values
def create_sequences(data, window_size):
    x = []
    y = []
    for i in range(len(data) - window_size - 1):
        x.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(x), np.array(y)

# Reshape data for LSTM input
def create_dataset_LSTM(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def getTicker(i_file, i_stockName):
    i_list = []
    # Open the CSV file
    with open(i_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row['tickerName'] == i_stockName:
                i_list.append(row)
    return i_list
