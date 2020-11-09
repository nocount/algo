# Basic Model for predicting next day stock price

import config
from data import process_dataset
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt


# Importing data
ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = process_dataset('AAPL_daily.csv')

# Splitting into train and test
test_split = 0.8
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)

####################### DOES NOT WORK PYTHON 3.8 because tf dependency #############################

# Setting up model
lstm_input = Input(shape=(50, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


# Evaluating performance
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(real_mse)
print(scaled_mse)


# Plot results
plt.gcf().set_size_inches(22, 15, forward=True)

real = plt.plot(unscaled_y_test[0:-1], label='real')
pred = plt.plot(y_test_predicted[0:-1], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

# Save model
model.save(f'basic_model.v1')