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
    #format date: yy_mm_dd
    #returns 1 if date1 is after date2, 0 otherwise
    
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

def readDate(path):
    #converts date from path(220407040422_sen.csv -> example) to format yy_mm_dd

    pos = path.find("_")

    year = path[pos - 12] + path[pos - 11]
    month = path[pos - 10] + path[pos - 9]
    day = path[pos - 8] + path[pos - 7]

    return year + "_" + month + "_" + day

def posTime(time):
    #returns positions from time format hour:min:sec

    counter = 0

    pos = []

    for x in time:

        counter = counter + 1

        if x == ":":

            pos.append(counter)

    return pos

def readHour(time):
    #read hour from time format hour:min:sec

    pos = posTime(time)

    return time[0:(pos[0] - 1)]

def readMinutes(time):
    #read minutes from time format hour:min:sec

    pos = posTime(time)

    return time[(pos[0]):(pos[1] - 1)]

def readSeconds(time):
    #read seconds from time format hour:min:sec

    pos = posTime(time)

    return time[(pos[1]):]

def isNum(char):
    #returns 1 if input date corresponds to a number


    true = 0

    x = 0

    while x < 10 and true == 0:

        x = x + 1

        if char == str(x):
            true = 1

    return true

def readSensorNum(path):
    #returns sensor number from path

    counter = 0

    counterAnt = 0

    sensor = 0

    pos = []

    x = 0

    for x in path:

        counter = counter + 1

        if x == "/":

            pos.append(counter)

    x = 0

    for x in range(int(pos[0]), int(pos[len(pos) - 1]) - 1):

            sensor = sensor*10 + int(path[x])

    return sensor

def getFeatures(data):
    ## given a df such ['HORA','Timestamp','Accel_x','Accel_y','Accel_z','Gyro_x','Gyro_y','Gyro_z','Magnet_x','Magnet_y','Magnet_z']
    ## computes preprocesing and feature extraction

    low = signal.butter(3, 0.5, 'low', output='sos')
    hp = signal.butter(3, 0.5, 'hp', output='sos',fs = 20)

    Accel_x = data.iloc[:,2]
    Accel_y = data.iloc[:,3]
    Accel_z = data.iloc[:,4]

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

            
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # ax1.plot(AccelX_int)
    # ax1.set_title('before')

    # MEdian filter
    Accel_x = signal.medfilt(AccelX_int)
    Accel_y = signal.medfilt(AccelY_int)
    Accel_z = signal.medfilt(AccelZ_int)

    # ax2.plot(Accel_x)
    # ax2.set_title('median')

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

    # ax3.plot(Accel_x_hp)
    # ax3.set_title('butter')

    #plt.show()
    y = 0

    for y in range(Accel_x_hp.shape[0]):

        SMA = abs(Accel_x_hp[y]) + abs(Accel_y_hp[y]) + abs(Accel_z_hp[y])
        SVM = math.sqrt(pow(Accel_x_hp[y],2) + pow(Accel_y_hp[y],2) + pow(Accel_z_hp[y],2))


    SMA = round(SMA/y, 4)
    SVM = round(SVM/y, 4)

    features = pd.DataFrame({"SMA": [SMA], "SVM": [SVM]})

    return features

def readDataFrame(path, file, df, lastDay):
    ##reads dataframe from ".csv" file

    count = 0
    

    results = pd.DataFrame()
    data = pd.DataFrame

    Sensor = readSensorNum(path) ##gets sensor Num
    

    for x in range(df.shape[0]):  ##goes through the index file (df.index)

        newMov = df.iloc[x,0]

        # if x == 1000: #-------------------------------------------------------------
        #     break

        # print(newMov[0:5])

        if newMov[0:5] == 'HORA_': #gets next new movement

            # print(newMov)

            count = count + 1
            time = newMov[5:]

            date = readDate(file)
            hour = readHour(time)
            min = readMinutes(time)
            sec = readSeconds(time)

            if(int(date[:2]) > 20):

                currentDay = [date,hour,min,sec]

                if isAfter(currentDay,lastDay):

                    data = pd.DataFrame(df.loc[(x+1):(x+200)])


                    features = getFeatures(data)

                    data = pd.DataFrame({'Sensor': [Sensor], 'Date': [date], 'Hour': [hour], 'Minutes': [min], "Seconds": [sec]})

                    data = pd.concat([data, features], axis=1, join='inner')

                    # print("Data")
                    # print(data)

                    # type(data)

                    if x == 0:
                        results = pd.DataFrame(data)
                    else:
                        results = results.append(data, ignore_index=True)
                        # print(results)
                else:
                    print("Análisis ya realizado")

    # print("resultados")
    # print(results)

    return results

def readSensor(sensor, raw_path, features_path):

    print("SENSOR: ")
    print(sensor)

    path2 = raw_path + sensor + "/"

    files = os.listdir(path2)

    content = pd.DataFrame()

    existFile = 0

    lastDay = ["00_00_00","0","0","0"]

    try:
        sensor_features = pd.read_csv(features_path + sensor + ".csv")
        sensor_features = sensor_features.iloc[:, 1:]
        existFile = 1
        lastDay = sensor_features[["Date","Hour","Minutes","Seconds"]]
        lastDay = (lastDay.iloc[sensor_features.shape[0] - 1,:])

    except:
        print("No hay .csv previo generado para el Sensor: "+sensor)

    # checking all the csv files in the
    # specified path
    for file in files:

        # reading content of csv file
        # content.append(filename)
        # df_firstn = pd.read_csv(filename, nrows=0, sep=';')

        if file.endswith(".csv"):

            df = pd.read_csv(path2 + file, sep=';', names = ['HORA','Timestamp','Accel_x','Accel_y','Accel_z','Gyro_x','Gyro_y','Gyro_z','Magnet_x','Magnet_y','Magnet_z'])

            if content.empty :

                content = readDataFrame(path2, file, df, lastDay)

            else:

                content = content.append(readDataFrame(path2, file, df, lastDay),  ignore_index=True)

            # break

            print("Sensor: "+sensor+" file: " + file)

    # print(content)
    if existFile == 1:
        sensor_features = sensor_features.append(content)
        sensor_features.to_csv('CSVs/features_' + str(sensor) + ".csv")
    else:
        content.to_csv('CSVs/features_' + str(sensor) + ".csv")


start = time.time()

raw_path  = 'Medidas/'

features_path = 'CSVs/features_'

n_jobs = multiprocessing.cpu_count() - 1

Parallel(n_jobs=n_jobs)(delayed(readSensor)(sensor, raw_path, features_path) for sensor in os.listdir(raw_path))


end = time.time()
print(end - start)




