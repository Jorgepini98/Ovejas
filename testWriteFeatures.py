from numpy import true_divide
import pandas as pd
import os
import glob
import math
import multiprocessing
from joblib import Parallel, delayed
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def getFeatures(data):

    low = signal.butter(3, 0.5, 'low', output='sos')
    hp = signal.butter(3, 0.5, 'hp', output='sos',fs = 20)

    Accel_x = data.iloc[:,2]
    Accel_y = data.iloc[:,3]
    Accel_z = data.iloc[:,4]

    print(Accel_x)

    # Gyro_x = data.iloc[:,5]
    # Gyro_y = data.iloc[:,6]
    # Gyro_z = data.iloc[:,7]

    # Magnet_x = data.iloc[:,8]
    # Magnet_y = data.iloc[:,9]
    # Magnet_z = data.iloc[:,10]

    AccelX_int = []
    AccelY_int = []
    AccelZ_int = []

    # in order to avoid spike at filtering stage
    modified_Accel_x=[]
    modified_Accel_y=[]
    modified_Accel_z=[]

   
    SMA = 0
    SVM = 0

    for y in range(data.shape[0]):

        if y != 0:

            AccelX_int.append(int(Accel_x.iloc[y]))
            AccelY_int.append(int(Accel_y.iloc[y]))
            AccelZ_int.append(int(Accel_z.iloc[y]))

            
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.plot(AccelX_int)
    ax1.set_title('before')

    # MEdian filter
    Accel_x = signal.medfilt(AccelX_int)
    Accel_y = signal.medfilt(AccelY_int)
    Accel_z = signal.medfilt(AccelZ_int)

    ax2.plot(Accel_x)
    ax2.set_title('median')

    y = 0

    for y in range(50):
        modified_Accel_x.append(Accel_x[0])
        modified_Accel_y.append(Accel_x[0])
        modified_Accel_z.append(Accel_x[0])


    modified_Accel_x = np.append(modified_Accel_x,Accel_x)


    #low pass filter
    Accel_x_low = signal.sosfilt(low,modified_Accel_x)[50:250] - np.mean(modified_Accel_x)
    Accel_y_low = signal.sosfilt(low,modified_Accel_y)[50:250] - np.mean(modified_Accel_y)
    Accel_z_low = signal.sosfilt(low,modified_Accel_z)[50:250] - np.mean(modified_Accel_z)

    #high pass filter
    Accel_x_hp = signal.sosfilt(hp,modified_Accel_x)[50:250]
    Accel_y_hp = signal.sosfilt(hp,modified_Accel_x)[50:250]
    Accel_z_hp = signal.sosfilt(hp,modified_Accel_x)[50:250]

    ax3.plot(Accel_x_hp)
    ax3.set_title('butter')

    #plt.show()
    y = 0

    for y in range(Accel_x_hp.shape[0]):

        SMA = abs(Accel_x_hp[y]) + abs(Accel_y_hp[y]) + abs(Accel_z_hp[y])
        SVM = math.sqrt(pow(Accel_x_hp[y],2) + pow(Accel_y_hp[y],2) + pow(Accel_z_hp[y],2))


    # SMA = round(SMA/y, 4)
    # SVM = round(SVM/y, 4)

    features = pd.DataFrame({"SMA": [SMA], "SVM": [SVM]})

    return features

raw_path  = 'PrimerasMedidas/'

path2 = raw_path + "2" + "/"

file = "220409020017_sen.csv"

df = pd.read_csv(path2 + file, sep=';', names = ['HORA','Timestamp','Accel_x','Accel_y','Accel_z','Gyro_x','Gyro_y','Gyro_z','Magnet_x','Magnet_y','Magnet_z'])

df = df.iloc[0:201,:]

features = getFeatures(df)

# df1 = pd.read_csv("features/features_2.csv").iloc[:, 1:]

# date = df1["Date"]

# print(date)