import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression



df = pd.read_csv("sphist.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(ascending = True,by=['Date'],inplace=True)
df['index'] = range(0,df.shape[0],1)
df.set_index(['index'])


data_mean_5day = pd.rolling_mean(df.Close, window=5).shift(1)

data_mean_30day = pd.rolling_mean(df.Close, window=30).shift(1)

data_mean_365day = pd.rolling_mean(df.Close, window=365).shift(1)

data_mean_ratio = data_mean_5day/data_mean_365day

data_std_5day = pd.rolling_std(df.Close, window=5).shift(1)

data_std_365day = pd.rolling_std(df.Close, window=365).shift(1)

data_std_ratio = data_std_5day/data_std_365day

df['data_mean_5day'] = data_mean_5day
df['data_mean_365day'] = data_mean_365day
df['data_mean_ratio'] = data_mean_ratio
df['data_std_5day'] = data_std_5day
df['data_std_365day'] = data_std_365day
df['data_std_ratio'] = data_std_ratio


df_new = df[df["Date"] > datetime(year=1951, month=1, day=2)]
df_no_NA = df_new.dropna(axis=0)



train = df_no_NA[df_no_NA["Date"] < datetime(year=2013,month =1,day = 1)]

test = df_no_NA[df_no_NA["Date"] >= datetime(year=2013,month =1,day = 1)]





model = LinearRegression()
features = ['data_mean_5day','data_mean_365day','data_mean_ratio', 'data_std_5day','data_std_365day', 'data_std_ratio']

model.fit(train[features],train.Close)

predict = model.predict(test[features])

meanAbsoluteError = sum(abs(predict-test.Close))/len(predict)
print(meanAbsoluteError)
print(model.score(train[features],train.Close))

#Here are some ideas that might be helpful:

#The average volume over the past five days.
#The average volume over the past year.
#The ratio between the average volume for the past five days, and the average volume for the past year.
#The standard deviation of the average volume over the past five days.
#The standard deviation of the average volume over the past year.
#The ratio between the standard deviation of the average volume for the past five days, and the standard deviation of the average volume for the past year.
#The year component of the date.
#The ratio between the lowest price in the past year and the current price.
#The ratio between the highest price in the past year and the current price.
#The year component of the date.
#The month component of the date.
#The day of week.
#The day component of the date.
#The number of holidays in the prior month.











