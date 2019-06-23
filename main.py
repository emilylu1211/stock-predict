#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import math
# Input data files are available in the "data/" directory.

# split the data into training and test
split_ratio = 0.90
# how many previous days as input
steps = 60
# how many inputs
num_input = 1
symbol = '^GSPC'


scl = MinMaxScaler()


def scale_data(myinput):
    result = myinput
    # passed in myinput is dataGrid.Series and needs to be convered to list first
    result = result.tolist()
    # MinMaxScaler needs 2D array so just change it from n to (n,1)
    # for reshape, number of elements must be exactly the same
    result = np.reshape(result, (len(result), 1))
    # nornalize the data
    result = scl.fit_transform(result)
    return result


def main():

    num_elements_one_input = steps * num_input

    # use pandas to read csv file
    # result is a 2D data structure with labels
    data = pd.read_csv('./data/' + symbol + '.csv')
    adjclose = scale_data(data.AdjClose)

    # process the data to input and output
    # X is input, 1D with n * steps * num_input elements
    # y is output, 1D with n elements
    X, y = processData(adjclose, steps)

    # split the data into 90% for train and testing
    # split_point has to be an integer so use //
    splity = int(len(y) * split_ratio)
    # remember that for each output element there are num_elements_one_input
    splitx = int(splity * num_elements_one_input)

    # :splitx = [0, splitx), splitx: = [splitx, len(X))
    X_train, X_test = X[:splitx], X[splitx:]
    y_train,y_test = y[:splity],y[splity:]

    #print first data slice
    print(X_train.shape[0])
    print(X_test.shape[0])
    print(y_train.shape[0])
    print(y_test.shape[0])

    print(X_train[0])
    print(X_test[0])

    #Build the model
    model = Sequential()
    # Add one layer of LSTM with 256 tensors
    model.add(LSTM(256,input_shape=(steps,num_input)))
    # Dense is for output layer
    model.add(Dense(1))
    # optimizer is the gradient descent algorithm
    # mse is the most popular loss function
    model.compile(optimizer='adam',loss='mse')

    # reshape input from 1D array to 3D so model can use
    X_train = X_train.reshape((len(X_train) // num_elements_one_input, steps, num_input))
    X_test = X_test.reshape((len(X_test) // num_elements_one_input, steps, num_input))

    # train the model
    model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),shuffle=False)

    # train completed, let's do predict
    y_predicted = model.predict(X_test)


    # how is predicted comapred with actual?
    testScore = mean_absolute_percentage_error(y_test, y_predicted)
    print('Test Score: %.2f MAPE' % (testScore))  # Root Mean Square Error,

    # draw it
    # y_test and y_predicted are still in 0 - 1 so we need to call inverse_transform
    # to change them into real prices
    # as before, scl needs to work on 2D,
    # reshape(-1, 1) will do the trick, -1 means it will calculate for us
    line1, = plt.plot(scl.inverse_transform(y_test.reshape(-1,1)),  marker='d', label='Actual')
    line2, = plt.plot(scl.inverse_transform(y_predicted.reshape(-1,1)), marker='o', label='Predicted')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.Text(symbol)
    plt.show()

# one input and one output
# if data has 5000 elements and look back days is 20
# then there will be 4980 * 20 elements for X and 4980 elements for Y
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        for j in range(lb):
            X.append(data[i+j])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = []
    for i in range(len(y_true)):
        #print("%.3f, %.3f, %.3f"%(y_true[i], y_pred[i], np.abs((y_true[i] - y_pred[i]) / y_true[i])))
        diff.append(np.abs((y_true[i] - y_pred[i]) / y_true[i]))
    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mymean = np.mean(diff)
    return  mymean * 100


if __name__ == '__main__':
    main()
