import pandas as pd
import quandl, math, datetime, pickle
quandl.ApiConfig.api_key = 'iJonhB1g2rrufwqNVxUs'
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

########### To define specific features/columns
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

# define relationships
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

########### features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))

########### preprocessing to scale data
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # 10%
X = X[:-forecast_out] # 90%


df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

###########  classifier
clf = LinearRegression()

# fit() is for training purpose
clf.fit(X_train, y_train)

# to avoid classifier execute every time store it into pickle
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# score() is performed on testing data
accuracy = clf.score(X_test, y_test)

###########  predication
forecast_set = clf.predict(X_lately)

#print(accuracy)
df['forecast'] = np.nan

########### calculate next dates
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # sec in day
next_unix = last_unix + one_day # next day



for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
