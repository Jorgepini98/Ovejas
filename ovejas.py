from numpy import true_divide
import pandas as pd
import os
import glob
import math
import multiprocessing
from joblib import Parallel, delayed

def isAfter(date1,date2):
    #format date: yy_mm_dd
    
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

    pos = path.find("_")

    year = path[pos - 12] + path[pos - 11]
    month = path[pos - 10] + path[pos - 9]
    day = path[pos - 8] + path[pos - 7]

    return year + "_" + month + "_" + day

def posTime(time):

    counter = 0

    pos = []

    for x in time:

        counter = counter + 1

        if x == ":":

            pos.append(counter)

    return pos

def readHour(time):

    pos = posTime(time)

    return time[0:(pos[0] - 1)]

def readMinutes(time):

    pos = posTime(time)

    return time[(pos[0]):(pos[1] - 1)]

def readSeconds(time):

    pos = posTime(time)

    return time[(pos[1]):]

def isNum(char):

    true = 0

    x = 0

    while x < 10 and true == 0:

        x = x + 1

        if char == str(x):
            true = 1

    return true

def readSensorNum(path):

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

    # Accel_x = data.iloc[:,2]
    # Accel_y = data.iloc[:,3]
    # Accel_z = data.iloc[:,4]

    # Gyro_x = data.iloc[:,5]
    # Gyro_y = data.iloc[:,6]
    # Gyro_z = data.iloc[:,7]
    
    # Magnet_x = data.iloc[:,8]
    # Magnet_y = data.iloc[:,9]
    # Magnet_z = data.iloc[:,10]

    acel = 0
    gyro = 0

    for y in range(data.shape[0]):

        if y != 0:

            acel = acel + math.sqrt(pow(int(data.iloc[y,2]),2)) + math.sqrt(pow(int(data.iloc[y,3]),2)) + math.sqrt(pow(int(data.iloc[y,4]),2))
                    # print(acel)

    acel = acel/data.shape[0]

    features = pd.DataFrame({"Acel_1": [acel], "Acel_2": [acel]})

    return features

def readDataFrame(path, file, df, lastDay):

    count = 0
    

    results = pd.DataFrame()
    data = pd.DataFrame

    Sensor = readSensorNum(path)
    #Sensor = sensor

    for x in range(df.shape[0]):

        newMov = df.iloc[x,0]

        # if x == 1000: #-------------------------------------------------------------
        #     break

        # print(newMov[0:5])

        if newMov[0:5] == 'HORA_':

            # print(newMov)

            count = count + 1
            time = newMov[5:]

            date = readDate(file)
            hour = readHour(time)
            min = readMinutes(time)
            sec = readSeconds(time)

            currentDay = [date,hour,min,sec]

            if isAfter(currentDay,lastDay):

                data = pd.DataFrame(df.loc[(x+1):(x+20)])


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



raw_path  = 'PrimerasMedidas/'

features_path = 'CSVs/features_'

n_jobs = multiprocessing.cpu_count() - 1

Parallel(n_jobs=n_jobs)(delayed(readSensor)(sensor, raw_path, features_path) for sensor in os.listdir(raw_path))








