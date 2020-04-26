import re
from random import random
import logging
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
import numpy as np
from sklearn.linear_model import LinearRegression

# config numpy
np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
pd.set_option('display.max_columns', None)

# log init
logger = logging.getLogger('information')
hdlr = logging.FileHandler('log/information.log', 'w+')
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# COVID-19 confirmed data
confirmed = pd.read_csv('data/confirmed/china_weekday_new.csv')  # 1.22-4.8
data = np.squeeze(confirmed[0:1].to_numpy())

# stock data
filePath = "data/"
file_list = os.listdir(filePath)

stock = pd.read_csv("data/stocks/Index_k_data_000001.csv")
high = stock['high']
low = stock['low']
average = ((high + low) / 2).to_numpy()

with open("stock_code_index.txt") as f:
    index = f.readlines()
index = [x.strip() for x in index]

# normalization
data = (data - np.mean(data))/np.std(data)
average = (average - np.mean(average))/np.std(average)

df = pd.DataFrame(data=average, columns=[index[0]])

# compute covariance matrix
# cov_mat = np.stack((), axis=1)
string = str(np.corrcoef(data, average)[0, 1])
replaced = re.sub(r'(?<!\])\n', '', string)
logger.info(replaced)

for i in range(1, len(file_list)):
    stock = pd.read_csv("data/" + file_list[i])
    high = stock['high']
    low = stock['low']
    average = ((high + low) / 2).to_numpy()

    # normalization
    data = (data - np.mean(data))/np.std(data)
    average = (average - np.mean(average))/np.std(average)

    # cov_mat = np.stack((data, average), axis=1)

    string = str(np.corrcoef(data, average)[0, 1])
    replaced = re.sub(r'(?<!\])\n', '', string)
    logger.debug(index[i] + ": " + replaced)

    # add new column to an dataframe
    df[index[i]] = average


# Load the boston dataset.
X = df.to_numpy()
y = data

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()  # tol=0.005

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.10)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

print(n_features)
print(sfm.transform(X))
print(sfm.get_support())
print(index)

# # Reset the threshold till the number of features equals two.
# while n_features > 2:
#     sfm.threshold += 0.1
#     X_transform = sfm.transform(X)
#     n_features = X_transform.shape[1]

# lrModel = LinearRegression()
# lrModel.fit(sfm.transform(X), y)
# print(lrModel.predict([[-0.465, 1.393, 1.485, - 0.598]]))
