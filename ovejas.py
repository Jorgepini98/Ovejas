from numpy import true_divide
import pandas as pd
import os
import glob
import math
import multiprocessing
from joblib import Parallel, delayed

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

def findPos(df, date, hour, min, sec):

    length = df.shape[0]

    x = 0

    featurePos = -1

    while (x < length):

        #Sensor,Date,Hour,Minutes,Seconds,...

        checkDate = df.iloc[x,1]

        dateReached = 0
        hourReached = 0
        minReached = 0

        if(checkDate == date):

            dateReached = 1

            checkHour = df.iloc[x,2]

            if(str(checkHour) == hour):

                hourReached = 1

                checkMin = df.iloc[x,3]

                if(str(checkMin) == min):

                    minreached = 1

                    checkSec = df.iloc[x,4]

                    if(str(checkSec) == sec):

                        featurePos = x

                        x = length
                        
                else:
                    if(minReached == 1):
                        x = length
                    
            else:
                if(hourReached == 1):
                    x = length

        else:
            if(dateReached == 1):
                x = length

        x = x + 1

    return featurePos


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

def getFeatures(data, dataFeatures):

    # Accel_x = data.iloc[:,2]
    # Accel_y = data.iloc[:,3]
    # Accel_z = data.iloc[:,4]

    # Gyro_x = data.iloc[:,5]
    # Gyro_y = data.iloc[:,6]
    # Gyro_z = data.iloc[:,7]
    
    # Magnet_x = data.iloc[:,8]
    # Magnet_y = data.iloc[:,9]
    # Magnet_z = data.iloc[:,10]

    Acel_1 = 0
    Acel_2 = 0

    desiredColumns = ["Acel_1","Acel_2"] 
    
    columnsToProcess = desiredColumns
    
    currentColumns = dataFeatures.columns
    
    for x in range(len(currentColumns) - 1):
        for y in range(len(desiredColumns) - 1):
        
            if str(currentColumns[x]) == desiredColumns[y]:
                
                columnsToProcess.pop(columnsToProcess.index(desiredColumns[y]))

    
   
    for y in range(data.shape[0]):

        if y != 0:
        
            if  "Acel_1" in columnsToProcess:

                Acel_1 = Acel_1+ math.sqrt(pow(int(data.iloc[y,2]),2)) + math.sqrt(pow(int(data.iloc[y,3]),2)) + math.sqrt(pow(int(data.iloc[y,4]),2))
                    # print(acel)
                    
            else:
                
                Acel_1 = int(dataFeatures["Acel_1"])
                
            # print(Acel_1)

            if  "Acel_2" in columnsToProcess:      
            
                Acel_2 = Acel_1
                
            else:
                
                Acel_2 = int(dataFeatures["Acel_2"])

            # print(Acel_2)



    features = pd.DataFrame({Acel_1,Acel_2},columns = desiredColumns)

    return features

def readDataFrame(path, file, df, sensor_features):

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

            featurePos = findPos(sensor_features,date,hour,min,sec)

            if(featurePos >= 0):

                dataFeatures = pd.DataFrame(data = sensor_features.iloc[featurePos,:]).T

            else:
                dataFeatures = pd.DataFrame()

            # print(dataFeatures)

            data = pd.DataFrame(df.loc[(x+1):(x+20)])


            features = getFeatures(data,dataFeatures)

            data = pd.DataFrame({'Sensor': [Sensor], 'Date': [date], 'Hour': [hour], 'Minutes': [min], "Seconds": [sec]})

            data = pd.concat([data, features], axis=1, join='inner')

            # print("Data")
            # print(data)

            # type(data)

            if x == 0:
                results = pd.DataFrame(data)
            else:
                results = results.append(data, ignore_index=True) #, ignore_index=True
                # print(results)

    # print("resultados")
    # print(results)

    return results

def readSensor(sensor,  raw_path, features_path):

    print("SENSOR: ")
    print(sensor)

    sensor_path = raw_path + sensor + "/"

    files = os.listdir(sensor_path)

    content = pd.DataFrame()

    

    try:

        sensor_features = pd.read_csv(features_path + sensor + ".csv")
        sensor_features = sensor_features.iloc[:, 1:]

    except:
        print("No hay .csv generado para el Sensor: "+sensor)
        sensor_features = pd.DataFrame()

    # checking all the csv files in theS
    # specified path
    for file in files:

        # reading content of csv file
        # content.append(filename)
        # df_firstn = pd.read_csv(filename, nrows=0, sep=';')

        if file.endswith(".csv"):

            raw = pd.read_csv(sensor_path + file, sep=';', names = ['HORA','Timestamp','Accel_x','Accel_y','Accel_z','Gyro_x','Gyro_y','Gyro_z','Magnet_x','Magnet_y','Magnet_z'])

            if content.empty :

                content = readDataFrame(sensor_path, file, raw, sensor_features)

            else:

                content = content.append(readDataFrame(sensor_path, file, raw, sensor_features), ignore_index=True) #, ignore_index=True

            # break
            print(sensor +":  " + file)

    # print(content)
    content.to_csv('CSVs/features_' + str(sensor) + ".csv")



raw_path = 'PrimerasMedidas/'

features_path = 'CSVs/features_'

n_jobs = multiprocessing.cpu_count() - 1

Parallel(n_jobs=n_jobs)(delayed(readSensor)(sensor, raw_path, features_path) for sensor in os.listdir(raw_path))








