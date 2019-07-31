#!/usr/bin/python

import requests
import time
import os
import os.path
import json
from datetime import datetime, timedelta
import pandas as pd


# Range format "&from=1547524844&to=1557283610&events=quote&interval=1d"


def get_range(days_to_subtract):
    now = datetime.now()
    time_from = now - timedelta(days=days_to_subtract)
    range = "&from=%d&to=%d&events=quote&interval=1d" % (time_from.timestamp(), now.timestamp())
    return range

# converting unix time to yyyy-mm-dd
def get_date_from_unix_time(unix_time):
    dt = datetime.utcfromtimestamp(unix_time)
    # yyyy-mm-dd
    return "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)

# total_process : getting data from yahoo and saving it in a csv file
# symbol = stock name, ex: msft
# days_to_subtract= days from today into the past
def download_history(symbol, days_to_subtract):
    base_url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/get-histories?region=US&lang=en&symbol="
    myrange = get_range(days_to_subtract)
    URL = base_url + symbol + myrange
    # API needs the following headers
    headers = {'X-RapidAPI-Host': 'apidojo-yahoo-finance-v1.p.rapidapi.com', 'X-RapidAPI-Key': 'api key'}

    # get the data from yahoo api pages
    #r = response
    r = requests.get(url=URL, headers=headers)

    data = r.content
    r.close()

    # convert data into json chart
    j = json.loads(data)
    # all the fields we're interested in, they are all lists
    timestamp = j['chart']['result'][0]['timestamp']
    # open is keyword
    myopen = j['chart']['result'][0]['indicators']['quote'][0]['open']
    high = j['chart']['result'][0]['indicators']['quote'][0]['high']
    low = j['chart']['result'][0]['indicators']['quote'][0]['low']
    myclose = j['chart']['result'][0]['indicators']['quote'][0]['close']
    volume = j['chart']['result'][0]['indicators']['quote'][0]['volume']
    adjclose = j['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']

    # name the file
    file_path = "./data/" + symbol + ".csv"
    print(file_path)

    #open file for writing (save new file over old file)
    file = open(file_path, "w")
    #save label in file
    file.write('Date,Open,High,Low,Close,AdjClose,Volume\n')
    length = len(timestamp)
    #loop through data
    for i in range(0, length):
        str_date = get_date_from_unix_time(timestamp[i])
        #save data
        file.write(str_date + ',' + str(myopen[i]) + ',' + str(high[i]) + ',' + str(low[i]) + ',' +
                   str(myclose[i]) + ',' + str(adjclose[i]) + ',' + str(volume[i]) + '\n')

    file.close()


download_history('^GSPC', 10000)
#download_history('MSFT', 10000)
#download_history('GM', 10000)
#download_history('AMZN', 10000)

def sp500(sp_list):
    data = pd.read_csv(sp_list)
    symbols = data.Symbol
    symbols = symbols.tolist()

    for idx, symbol in enumerate(symbols):
        out_name = os.path.join("./data/", symbol + ".csv")
        if os.path.exists(out_name):
            print("%d, %s already downloaded, skip" % (idx, symbol))
            continue

        download_history(symbol, 10000)
        print("%d, %s downloaded" % (idx, symbol))

        if idx >= 499:
            break

        time.sleep(2)

    print("\n\nALL DOWNLOADED!!!")



# sp500("./sp_list.csv")
