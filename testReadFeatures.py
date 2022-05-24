from numpy import true_divide
import numpy as np
import pandas as pd
import os
import glob
import math
import multiprocessing
from joblib import Parallel, delayed
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import datetime



def lastPos(df,pos):
    return pos == len(df)

def readDate(date):

    year = date[0] + date[1]
    month = date[3] + date[4]
    day = date[6] + date[7]

    return int(year),int(month),int(day)

#sensor_features[["Date","Hour","Minutes","Seconds"]]
def isNextDay(date1,date2):
    #format yy_mm_dd

    year1,month1,day1 = readDate(date1)
    year2,month2,day2 = readDate(date2)

    oneDay = datetime.timedelta(days=1)

    date1 = datetime.date(year1, month1, day1)
    date2 = datetime.date(year2, month2, day2)

    return (date1 + oneDay) == date2
    
def isNextHour(hour1,hour2):

    hour1 = int(hour1)
    hour2 = int(hour2)

    return (hour1 + 1)%24 == hour2

def daysApart(date1,date2):

    year1,month1,day1 = readDate(date1)
    year2,month2,day2 = readDate(date2)

    date1 = datetime.date(year1, month1, day1)
    date2 = datetime.date(year2, month2, day2)

    return (date2 - date1).days

def hoursApart(hour1,hour2):
    hour1 = int(hour1)
    hour2 = int(hour2)

    return max([hour1,hour2]) - min ([hour1,hour2])

def readSensor(sensor, raw_path):

    previousDay = "00_00_00"
    previousHour = "00"

    datePos = 0
    hourPos = 0

    Day_SMA = []
    Hour_SMA = []

    Day_SVM = []
    Hour_SVM = []

    DateRecord = []
    HourRecord = []
    HourDateRecord = []

    SMA_day = 0
    SVM_day = 0

    SMA_hour = 0
    SVM_hour = 0

    path = raw_path + "features_" + str(sensor) + ".csv"

    df = pd.read_csv(path).iloc[:,1:]

    for i in df.index:

        SMA_day += df["SMA"][i]
        SVM_day += df["SVM"][i]

        SMA_hour += df["SMA"][i]
        SVM_hour += df["SVM"][i]

        if (previousDay != df["Date"][i]) and (i > 0):

            if isNextDay(previousDay,df["Date"][i]):

                DateRecord.append(previousDay)

                Day_SMA.append(round(SMA_day,2))
                Day_SVM.append(round(SVM_day,2))

            else:
                if lastPos(df["Date"],i):

                    DateRecord.append(df["Date"][i])

                    Day_SMA.append(round(SMA_day,2))
                    Day_SVM.append(round(SVM_day,2))

                else:
                    for days in range(daysApart(previousDay,df["Date"][i])):
                        if(days != 0):
                            Day_SMA.append(0)
                            Day_SVM.append(0)  

            SMA_day = 0
            SVM_day = 0  

        
        if (previousHour != df["Hour"][i]) and (i > 0):

            if isNextHour(previousHour,df["Hour"][i]):

                HourRecord.append(previousHour)
                HourDateRecord.append(df["Date"][i])

                Hour_SMA.append(round(SMA_hour,2))
                Hour_SVM.append(round(SVM_hour,2))

            else:
                if lastPos(df["Hour"],i):

                    HourRecord.append(df["Hour"][i])
                    HourDateRecord.append(df["Date"][i])

                    Hour_SMA.append(round(SMA_hour,2))
                    Hour_SVM.append(round(SVM_hour,2))

                else:
                    for hours in range(hoursApart(previousHour,df["Hour"][i])):
                        if(hours != 0):
                            Hour_SMA.append(0)
                            Hour_SVM.append(0)

            SMA_hour = 0
            SVM_hour = 0   


        previousDay = df["Date"][i]
        previousHour = df["Hour"][i]

    

    df_days = pd.DataFrame({"Date":[DateRecord],"SMA": [Day_SMA], "SVM": [Day_SVM]},columns = ["Date","SMA","SVM"])
    df_hour = pd.DataFrame({"Date":[HourDateRecord],"Hour":[HourRecord],"SMA": [Hour_SMA], "SVM": [Hour_SVM]})


    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(Day_SMA[1:])
    ax1.set_title('Day_SMA')

    ax2.plot(Day_SVM[1:])
    ax2.set_title('Day_SVM')

    fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(Hour_SMA[1:])
    ax1.set_title('Hour_SMA')

    ax2.plot(Hour_SVM[1:])
    ax2.set_title('Hour_SVM')
    plt.show()

    return df_days,df_hour


path = "CSVs/"

sensor = 5

df1,df2 = readSensor(sensor,path)

print(df1)

# date1 = "22_01_31"
# date2 = "22_02_11"

# print(daysApart(date1,date2))

# print(Day_SMA)

# fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# ax1.plot(Day_SMA[1:])
# ax1.set_title('Day_SMA')

# ax2.plot(Day_SVM[1:])
# ax2.set_title('Day_SVM')

# fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# ax1.plot(Hour_SMA[1:])
# ax1.set_title('Hour_SMA')

# ax2.plot(Hour_SVM[1:])
# ax2.set_title('Hour_SVM')
# plt.show()