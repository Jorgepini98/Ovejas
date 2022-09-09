
from multiprocessing.resource_sharer import stop
import numpy as np
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from datetime import date, timedelta
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import sys 
import os 
import multiprocessing
from joblib import Parallel, delayed

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def dateFormatChange(date):
        year = 20*100 + int(date[:2])
        month = int(date[3:5])
        day = int(date[6:])
        return year,month,day


def label(file):

    sensorlabel = pd.read_csv(pathOut + file)

    labels = sensorlabel["label"]

    dates = sensorlabel["Date"]

    numOfLabels = set(labels)

    df = pd.DataFrame()

    start_date = date(2022, 4, 1)
    end_date = date(2022, 9, 1)

    pos = 0
    datePos = 0
    date1 = dates[pos]

    for single_date in daterange(start_date, end_date):
            datePos += 1


    df["Date"] = [0]*datePos

    for label in numOfLabels:
        df[str(int(label))] = [0]*datePos

    datePos = 0

    # for label in labels:

    #         for column in df.columns:

    #             if column == str(int(label)):
    #                 df[column] += 1

            # if (int(label) == 0):
            #     label0 += 1
            # elif (int(label) == 1):
            #     label1 += 1
            # elif (int(label) == 2):
            #     label2 += 1
            # elif (int(label) == 3):
            #     label3 += 1

    for single_date in daterange(start_date, end_date):

        year,month,day = (dateFormatChange(date1))
        comparingDate = date(year,month,day,0)
        df.at[datePos,"Date"] = single_date
        #print(comparingDate)

        while comparingDate == single_date and pos < len(dates) - 1:
            pos += 1
            date1 = dates[pos]
            year,month,day = dateFormatChange(date1)
            comparingDate = date(year,month,day)


            for column in df.columns:
                if column == str(int(labels[datePos])):
                    df.at[datePos,column] += 1

        datePos += 1

    oveja = file[6:10]

    fig = px.line(df, x="Date", y=df.columns,
              hover_data={"Date": "|%B %d, %Y"},
              title=oveja + ' labels')
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    #fig.show()

    

    fig.write_html(directorio + "figuras/" + file + ".html")

    print(file)
    #print(df)


n_jobs = multiprocessing.cpu_count() - 1

print(n_jobs)

directorio = "D:/DatosOvejas/"

pathIn = directorio + "OvejasFeaturesOver/"
pathOut = directorio + "clusteringHDbscanOver/"

files = os.listdir(pathOut)

#label(files[0])
Parallel(n_jobs=n_jobs)(delayed(label)(file) for file in os.listdir(pathOut))