"""1. Import Library"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--training',
                    default='training_data.csv',
                    help='input training data file name')
parser.add_argument('--output',
                    default='submission.csv',
                    help='output file name')
args = parser.parse_args()




import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional


from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from numpy.random import seed 
seed(1)


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import array

"""# 資料準備"""

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

n_timestamp = 10
#all 818
train_days = 718  # number of days to train from
testing_days = 100 # number of days to be predicted
# train_days = 669  # number of days to train from
# testing_days = 7 # number of days to be predicted
# train_days = 1500  # number of days to train from
# testing_days = 500 # number of days to be predicted
n_epochs = 25
filter_on = 1

model_type = 2


# dataset = pd.read_csv(url)
dataset = pd.read_csv(args.training)

if filter_on == 1:
    dataset['operating_reserve'] = medfilt(dataset['operating_reserve'], 3)
    dataset['operating_reserve'] = gaussian_filter1d(dataset['operating_reserve'], 1.2)

# print(dataset.shape)

train_set = dataset[0:train_days].reset_index(drop=True)
test_set = dataset[train_days: train_days+testing_days].reset_index(drop=True)
training_set = train_set.iloc[:, 1:2].values
testing_set = test_set.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(testing_set)

# print(test_set)

def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# print(X_test.shape)

if model_type == 1:
    # Single cell LSTM
    model = Sequential()
    model.add(LSTM(units = 50, activation='relu',input_shape = (X_train.shape[1], 1)))
    model.add(Dense(units = 1))
if model_type == 2:
    # Stacked LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
if model_type == 3:
    # Bidirectional LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)
loss = history.history['loss']
epochs = range(len(loss))

y_predicted = model.predict(X_test)

y_predicted_descaled = sc.inverse_transform(y_predicted)

df_result = y_predicted_descaled[83:90]

numpy.savetxt("submission.csv", df_result,fmt="%.0f", delimiter=",")