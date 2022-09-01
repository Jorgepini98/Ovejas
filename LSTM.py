from tracemalloc import stop
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator


from numpy import mean
from numpy import std
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from tensorflow import keras
from functools import partial
import tensorflow as tf
import os
from keras.regularizers import l2 
from keras_visualizer import visualizer  
from scipy import stats


def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


df_train = pd.read_csv('x_train_2.csv')
df_train = df_train.drop("Unnamed: 0",axis = 1)

df_train = df_train.iloc[:round(df_train.shape[0]),:]

y_train = df_train['label'].values
x_train = df_train.drop('sheep',axis = 1)
x_train = df_train.drop('Sensor',axis = 1)
x_train = df_train.drop('Date',axis = 1)
x_train = df_train.drop('Hour',axis = 1)
x_train = df_train.drop('Minutes',axis = 1)
x_train = df_train.drop('label',axis = 1)

myset = set(y_train)
print(myset)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

print(d)

pca = PCA(n_components=d)
pca = pca.fit(x_train)

x_train = pca.transform(x_train)


train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

# print(y_train)
# print(y_test)
# print(y_val)

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
x_val = pd.DataFrame(x_val)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_val = pd.DataFrame(y_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1])
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1])

TIME_STEPS = 200
STEP = 200

x_train, y_train = create_dataset(
    x_train,
    y_train,
    TIME_STEPS,
    STEP
)

x_test, y_test = create_dataset(
    x_test,
    y_test,
    TIME_STEPS,
    STEP
)

x_val, y_val = create_dataset(
    x_val,
    y_val ,
    TIME_STEPS,
    STEP
)

# print(x_train.shape, y_train.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[x_train.shape[1], x_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


print(model.evaluate(x_test, y_test))

y_pred = model.predict(x_test)
