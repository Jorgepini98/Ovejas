# From the features extracted from each sensor (.csv file), sorts it out by sheep number defined in a excel sheet. Range dates are also defined in order to link each sheep with it's pertinent sensor. If file already created, checks last date in order not to re-compute.

from tracemalloc import stop
from turtle import position
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
import time


def isAfter(date1,date2):
    #format date: [yy_mm_dd,hh,mm,ss]
    #returns true if date1 is after date2, false otherwise

    true = 0

    year1 = int(date1[0][0])*10 + int(date1[0][1])
    month1 = int(date1[0][3])*10 + int(date1[0][4])
    day1 = int(date1[0][6])*10 + int(date1[0][7])
    hour1 = int(date1[1])
    min1 = int(date1[2])
    sec1 = int(date1[3])

    year2 = int(date2[0][0])*10 + int(date2[0][1])
    month2 = int(date2[0][3])*10 + int(date2[0][4])
    day2 = int(date2[0][6])*10 + int(date2[0][7])
    hour2 = int(date2[1])
    min2 = int(date2[2])
    sec2 = int(date2[3])

    if year1 > year2:
        true = 1
    elif year1 == year2:

        if month1 > month2:
            true = 1
        elif month1 == month2:

            if day1 > day2:
                true = 1
            elif day1 == day2:

                if hour1 > hour2:
                    true = 1
                elif hour1 == hour2:

                    if min1 > min2:
                        true = 1
                    elif min1 == min2:

                        if sec1 > sec2:
                            true = 1

    return true

def getTime1(string):
    hour = string[0:2]
    min = string[3:5]
    sec = string[6:8]
    date = string[9:19]
    return date,hour,min,sec

def getTime2(string):
    hour = string[20:22]
    min = string[23:25]
    sec = string[26:28]
    date = string[29:39]
    return date,hour,min,sec

def sensorToSheep(crotal,featuresPath, sheepPath,df1):

    result = pd.DataFrame()

    print("Crotal: " + str(crotal))

    previousFile_path = sheepPath + str(crotal) + ".csv"

    lastDate = ["00_00_00","00","00","00"]

    if os.path.exists(previousFile_path):

        check_data = pd.read_csv(sheepPath + str(crotal) + ".csv")
        lastP = len(check_data)

        lastDate = check_data[["Date","Hour","Minutes","Seconds"]].iloc[1,:]

    pos1 = 0
    ## cant use df1.loc[df1.index[crotales.astype('str').str.contains(crotal)].tolist()]
    while crotal != df1['crotal'][pos1]:
        pos1 += 1

    sensors = df1.loc[pos1]

    print(sensors)

    # sensors = (df1.loc[df1.index[df1['crotal'].str.contains('4990')].tolist()].iloc[:,1:]).T

    # print(sensors)

    pos2 = 0

    for sensor in sensors:
        if pos2 > 0:

            if sensor > 0:

                date1,hour1,min1,sec1 = getTime1(sensors.index[pos2])
                date2,hour2,min2,sec2 = getTime2(sensors.index[pos2])

                archivo = featuresPath + str(int(sensor)) + ".csv"
                df2 = pd.read_csv(archivo)

                pos3 = 0

                for pos3 in range(0,len(df2)):

                    date = [df2["Date"][pos3],df2["Hour"][pos3],df2["Minutes"][pos3],df2["Seconds"][pos3]]

                    if isAfter(date,[date1,hour1,min1,sec1]) and isAfter([date2,hour2,min2,sec2],date) and isAfter(date,lastDate):

                         result = result.append(df2.iloc[pos3], ignore_index=True)
                        #  print(result)

                result = result.iloc[:,1:] # quito la columna "unnamed" que se ha añadido al leer el csv
                addCrotal = pd.DataFrame()

                for l in range(result.shape[0]):
                    addCrotal.at[l,'sheep'] = crotal

                result = pd.concat([addCrotal,result], axis=1)

                if not result.empty:
                    result.to_csv(path2 + str(pos2) + "/" + str(crotal) + ".csv", mode='a', index=False, header=False)
                result = pd.DataFrame()

        pos2 += 1
        print("sheep: " + str(crotal) + " sensor: " + str(sensor))

    # result = result.iloc[:,1:] # quito la columna "unnamed" que se ha añadido al leer el csv

    # addCrotal = pd.DataFrame()
    # for l in range(result.shape[0]):
    #     addCrotal.at[l,'sheep'] = crotal

    # result = pd.concat([addCrotal,result], axis=1)

    # if not result.empty:
    #     result.to_csv(path2 + str(crotal) + ".csv")

    # return result

organizador  = 'Organizador.xlsx'

df1 = pd.DataFrame()

df1 = pd.read_excel(organizador)

crotales = df1['crotal']

directorio = 'D:/DatosOvejas/'

path1 = directorio + "RawWindow/"

path2 = directorio + "resultado/"

start = time.time()
n_jobs = multiprocessing.cpu_count() - 14

#sensorToSheep(4990,path1,path2,df1)
Parallel(n_jobs=n_jobs)(delayed(sensorToSheep)(crotal,path1,path2,df1) for crotal in crotales)



end = time.time()
print("Tiempo: " + str(end - start))
