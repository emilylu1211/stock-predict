## Predict S&P500 Index Prices Using LSTM 

### Motivation
Search engines, autonomous vehicles, and image recognition— all made possible by the fascinating malleability of deep learning. Inspired in part by our very own brains, deep learning relies on artificial neural networks interconnected by layers upon layers of nodes. It’s this complex web of artificial networks that makes contemporary technology so adaptable. For evaluating stock prices, pricing models have to be flexible to an environment where market trends and discrete data sets are constantly fluctuating. 

This project relies on TensorFlow, an open source Python library, What this project hopes to develop then is a stock price prediction model of the S&P 500 that sufficiently follows these patterns. To determine the optimal stock pricing model, multiple input variables, models, etc. will be tested to evaluate their impact on overarching accuracy, which is represented by the MAPE (mean absolute percentage error).

### RNN, LSTM, GRU
What is essential to the function of this deep learning stock model in particular is RNN, or Recurrent Neural Networks. The advantage of RNN lies in modelling time-series, wherein ‘time elapsed’ is a major determinant in the data set (the independent variable, in essence). My stock model relies heavily on a specialized subset model of RNN, called LSTM (Long Short-Term Memory), which shines in accounting for long-term dependencies. In other words, LSTM is capable of recognizing and using long-term data in determining overarching relationships. Another model of importance is GRU (Gated Recurrent Unit), whose function and structure align closely with LSTM.

### Data
I downloaded 10,000 days worth of S&P 500 Index data (^GSPC) from Yahoo Finance API, hosted by rapidapi.com. The data was downloaded in JSON format, then parsed and converted to the csv file ‘data/^GSPC.csv.’ By running the function ‘download_history()’ in ‘download.py’, I was able to download each individual stock’s data and save it in ‘data/{symbol}.csv.’ Using the ‘sp500()’ function, I was also able to download data for all 500 individual stocks in the S&P 500.

### Python Libraries Used
Python is the most popular language for machine learning due to the followin reasons:

* Python itself is easy to understand and learn.
* Python has a lot of packages for machine learning and data processing with almost zero learning curve.

The Python libraries I used are as follows:

1. Tensorflow - the backend framework for Keras
2. Keras - sequential deep learning model with LSTM, Dense, and GRU layers.
3. NumPy - converts data into multidimensional arrays 
4. Pandas - read csv files into data frames 
5. Scikit-learn - uses MinMaxScaler to normalize input data
6. Matplotlib - creates plot visualization of final stock predictions


### Code Explained

The following code imports all relevant libraries to the stock predict model.

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense, GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
```

The following code ensures results can be replicated by fixing the random seed number. 

```
np.random.seed(5)
```

The following code loads the dataset as a Pandas dataframe which is later transformed into a NumPy array.

```python
data = pd.read_csv('./data/' + symbol + '.csv')
open2 = scale_data(data.Open)
high = scale_data(data.High)
low = scale_data(data.Low)
volume = scale_data(data.Volume)
adjclose = scale_data(data.AdjClose)
dates = data.Date
```

The following code builds a LSTM model, with one input layer, one hidden layer containing 256 LSTM neurons, and one output layer with a single prediction value.

```
model = Sequential()
model.add(LSTM(256,input_shape=(steps,num_input)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
```

The following code trains the model.

```python
model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),shuffle=False)
```

The following code predicts the stock close price based on test data input.

```python
y_predicted = model.predict(X_test)
```

### Parameters

The parameters that I used for the model are as follows:

1. Num_input - The number of inputs I used in each run (i.e. close price)
2. Steps - How many days in the past the model would use as input values to predict the following day’s close price
3. Train_test_split_ratio - How much of the downloaded data would be used to train the model, and how much would be used for testing prediction 
4. Num_layers - The number of layers the model used


### ## Predict S&P500 Index Prices Using LSTM 

### Motivation
Search engines, autonomous vehicles, and image recognition— all made possible by the fascinating malleability of deep learning. Inspired in part by our very own brains, deep learning relies on artificial neural networks interconnected by layers upon layers of nodes. It’s this complex web of artificial networks that makes contemporary technology so adaptable. For evaluating stock prices, pricing models have to be flexible to an environment where market trends and discrete data sets are constantly fluctuating. 

This project relies on TensorFlow, an open source Python library, What this project hopes to develop then is a stock price prediction model of the S&P 500 that sufficiently follows these patterns. To determine the optimal stock pricing model, multiple input variables, models, etc. will be tested to evaluate their impact on overarching accuracy, which is represented by the MAPE (mean absolute percentage error).

### RNN, LSTM, GRU
What is essential to the function of this deep learning stock model in particular is RNN, or Recurrent Neural Networks. The advantage of RNN lies in modelling time-series, wherein ‘time elapsed’ is a major determinant in the data set (the independent variable, in essence). My stock model relies heavily on a specialized subset model of RNN, called LSTM (Long Short-Term Memory), which shines in accounting for long-term dependencies. In other words, LSTM is capable of recognizing and using long-term data in determining overarching relationships. Another model of importance is GRU (Gated Recurrent Unit), whose function and structure align closely with LSTM.

### Data
I downloaded 10,000 days worth of S&P 500 Index data (^GSPC) from Yahoo Finance API, hosted by rapidapi.com. The data was downloaded in JSON format, then parsed and converted to the csv file ‘data/^GSPC.csv.’ By running the function ‘download_history()’ in ‘download.py’, I was able to download each individual stock’s data and save it in ‘data/{symbol}.csv.’ Using the ‘sp500()’ function, I was also able to download data for all 500 individual stocks in the S&P 500.

### Python Libraries Used
Python is the most popular language for machine learning due to the followin reasons:

* Python itself is easy to understand and learn.
* Python has a lot of packages for machine learning and data processing with almost zero learning curve.

The Python libraries I used are as follows:

1. Tensorflow - the backend framework for Keras
2. Keras - sequential deep learning model with LSTM, Dense, and GRU layers.
3. NumPy - converts data into multidimensional arrays 
4. Pandas - read csv files into data frames 
5. Scikit-learn - uses MinMaxScaler to normalize input data
6. Matplotlib - creates plot visualization of final stock predictions


### Code Explained

The following code imports all relevant libraries to the stock predict model.

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense, GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
```

The following code ensures results can be replicated by fixing the random seed number. 

```
np.random.seed(5)
```

The following code loads the dataset as a Pandas dataframe which is later transformed into a NumPy array.

```python
data = pd.read_csv('./data/' + symbol + '.csv')
open2 = scale_data(data.Open)
high = scale_data(data.High)
low = scale_data(data.Low)
volume = scale_data(data.Volume)
adjclose = scale_data(data.AdjClose)
dates = data.Date
```

The following code builds a LSTM model, with one input layer, one hidden layer containing 256 LSTM neurons, and one output layer with a single prediction value.

```
model = Sequential()
model.add(LSTM(256,input_shape=(steps,num_input)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
```

The following code trains the model.

```python
model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),shuffle=False)
```

The following code predicts the stock close price based on test data input.

```python
y_predicted = model.predict(X_test)
```

### Parameters

The parameters that I used for the model are as follows:

1. Num_input - The number of inputs I used in each run (i.e. close price)
2. Steps - How many days in the past the model would use as input values to predict the following day’s close price
3. Train_test_split_ratio - How much of the downloaded data would be used to train the model, and how much would be used for testing prediction 
4. Num_layers - The number of layers the model used


### Results:

Mean Absolute Percentage Error (MAPE) is used to measure the prediction accuracy. Originally, the stock model only had one input: previous close prices of GSPC up to 20 days in the past. The desired output was the close price following these previous input dates.

Multiple changes were made to decrease MAPE values and increase accuracy, including:

__Original Model: One input (Close price, using data from 20 days in the past); MAPE=1.18__

1. Changing input number to 5 inputs: close price, open price, high, low, and volume
··* More information: Since I only considered close prices for inputs the first round, I decided to expand the amount of input data by including other basic yet essential stock information to the program.
··* MAPE=1.05
2. Adding 156 additional symbols (S&P 500 stocks) as inputs
··* More information: Since the S&P 500 is composed of the 500 largest American stocks, I decided to add in stock data from 156 S&P 500 stocks with the largest range in data values, such as AAPL and AMZN. 
··* MAPE=2.18
3. Changing model type from LSTM to GRU
··* Since LSTM and GRU are similar in structure and function, I wanted to see if a change in RNN model would make a significant difference.
··* MAPE=1.02
4. Adding second layer to model
··* More information: Much of the deep learning technology we see makes use of numerous layers and neural networks to improve efficacy. I added a second LSTM layer to test this application within my model.
··* MAPE=1.10

![Image](../blob/master/Plot_Graph.png?raw=true)

### Conclusion

Taken together, these results tell us that simply increasing the breadth of data inputs doesn’t necessarily increase accuracy: trial 3 (adding 156 symbols as inputs) used the largest volume of data, but had the highest MAPE of all trials at 2.18. On the other hand, adding in five basic stock inputs through trial 1 generated the second-most accurate stock pricing model, with a MAPE of 1.05. 

To improve upon the logistics of trial 3, it would probably be more helpful to determine which stock data to input by judging the valued weight of each stock to the S&P 500 index instead of the stock’s range of data. In regards to trial 3, which had the lowest MAPE of 1.02, it’s interesting to see how similar models can still generate differences in outcome. Further study of the limits and benefits of each model would definitely be helpful in resolving this front. 

As for trial 4, the pattern of ‘more isn’t necessarily better’ is shown through the MAPE of 1.10 as a result of adding a second LSTM layer, as opposed to the MAPE of 1.05 without it. In further trials, I hope to tweak more variable combinations and potentially add more variables (such as P/E ratio and analyst recommendations) to further the precision of the model. 

At the end of the day, though, this model only serves as a short-term visualization of how a stock could trend, and is definitely not indicative of absolute stock prices. But it should be taken as a valuable learning opportunity in delving into the varied uses of AI deep learning, its accessibility, and its endless ability to adapt to changes large and small—just like the human brain, if you think about it.

### Future Enhancements

Alongside the proposed changes mentioned in the conclusion, edits to data size can be made to possibly enhance the accuracy of the model. Since the model currently operates on daily data sets with up to ten thousand data points, examining stock data on an hourly or even minute basis might illuminate more avenues of improvement, especially on the short-term prediction front. Though having access to so much more data points would be helpful, such information is not available on the API I am currently using.




### References
http://faroit.com/keras-docs/1.2.0/applications/
https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673
https://keras.io/backend/
https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
http://benalexkeen.com/feature-scaling-with-scikit-learn/

