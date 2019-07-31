#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense, GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import os


import math
# Input data files are available in the "data/" directory.

# split the data into training and test
split_ratio = 0.95
# how many previous days as input
steps = 20
# how many inputs
num_input =5
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



def main_multi_files():
    global num_input

    #list files from data
    path = './data'

    common_start_date = ''
    files = []
    filelist = os.listdir(path)
    for file in filelist:
        if '.csv' in file:
            files.append(file)
            # print(file)
            df = pd.read_csv('./data/' + file, usecols = ["Date"])
            mydate = df.Date[0]
            if mydate > common_start_date:
                common_start_date = mydate

    print(common_start_date)
    print("number of files %d"%(len(files)))

    # use pandas to read csv file
    # result is a 2D data structure with labels
    data = pd.read_csv('./data/' + symbol + '.csv')
    data = data[data.Date >= common_start_date]
    open2 = scale_data(data.Open)
    high = scale_data(data.High)
    low = scale_data(data.Low)
    volume = scale_data(data.Volume)
    dates = data.Date
    dates = dates.tolist()

    lenth_GRPC = len(data.AdjClose)
    print("lenth_GRPC = %d"%(lenth_GRPC))
    mydata = []
    for file in files:
        if ("^GSPC" in file) == False:
            df = pd.read_csv('./data/' + file, usecols = ["Date","AdjClose"])
            df = df[df.Date >= common_start_date]
            myclose = scale_data(df.AdjClose)
            if len(myclose) != lenth_GRPC:
                print("%s, len = %d"%(file, len(myclose)))
            mydata.append(myclose)


    adjclose = scale_data(data.AdjClose)


    lenth_otherclose = len(mydata)
    num_input = num_input + lenth_otherclose

    print("num_input=%d"%(num_input))

    num_elements_one_input = steps * num_input

    # process the data to input and output
    # X is input, 1D with n * steps * num_input elements
    # y is output, 1D with n elements
    #X, y = processData(adjclose, steps)
    X, y, z = processData_multi(open2, high, low, volume, adjclose, steps, dates, mydata)


    # split the data into 90% for train and testing
    # split_point has to be an integer so use //
    splity = int(len(y) * split_ratio)
    # remember that for each output element there are num_elements_one_input
    splitx = int(splity * num_elements_one_input)

    # :splitx = [0, splitx), splitx: = [splitx, len(X))
    X_train, X_test = X[:splitx], X[splitx:]
    y_train,y_test = y[:splity],y[splity:]
    Z_test = z[splity:]

    #print first data slice
    print(X_train.shape[0])
    print(X_test.shape[0])
    print(y_train.shape[0])
    print(y_test.shape[0])

    print(X_train[0])
    print(X_test[0])

    #Build the model
    model = Sequential()
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
    # we also want to see first half and second half test score
    testScore, testScore2, testScore3 = mean_absolute_percentage_error(y_test, y_predicted)
    print('Test Score: %.2f MAPE' % (testScore))  # Root Mean Square Error,
    print('Test Score 2: %.2f MAPE' % (testScore2))
    print('Test Score 3: %.2f MAPE' % (testScore3))
    # draw it
    # y_test and y_predicted are still in 0 - 1 so we need to call inverse_transform
    # to change them into real prices
    # as before, scl needs to work on 2D,
    # reshape(-1, 1) will do the trick, -1 means it will calculate for us
    line1, = plt.plot(Z_test, scl.inverse_transform(y_test.reshape(-1,1)),  marker='d', label='Actual')
    line2, = plt.plot(Z_test, scl.inverse_transform(y_predicted.reshape(-1,1)), marker='o', label='Predicted')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()

def processData_multi(open1, high1, low1, volume1, close1, lb, dates1, othercloses):
    X,Y,Z = [],[],[]
    for i in range(len(close1)-lb):
        for j in range(lb):
            X.append(open1[i+j])
            X.append(high1[i+j])
            X.append(low1[i+j])
            X.append(volume1[i+j])
            X.append(close1[i+j])
            lenth_other_symbols = len(othercloses)
            for k in range(lenth_other_symbols):
                otherclose = othercloses[k]
                lenth_num_dates = len(otherclose)
                X.append(otherclose[i+j])
        Y.append(close1[(i+lb)])
        Z.append(dates1[(i+lb)])
    return np.array(X),np.array(Y),np.array(Z)







def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = []
    diff2 = []
    diff3 = []
    for i in range(len(y_true)):
        #print("%.3f, %.3f, %.3f"%(y_true[i], y_pred[i], np.abs((y_true[i] - y_pred[i]) / y_true[i])))
        diff.append(np.abs((y_true[i] - y_pred[i]) / y_true[i]))
        if i < len(y_true)/2:
            diff2.append(np.abs((y_true[i] - y_pred[i]) / y_true[i]))
        else:
            diff3.append(np.abs((y_true[i] - y_pred[i]) / y_true[i]))

    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mymean = np.mean(diff)
    mymean2 = np.mean(diff2)
    mymean3 = np.mean(diff3)
    return  mymean * 100, mymean2 * 100, mymean3 * 100


if __name__ == '__main__':
    main_multi_files()
