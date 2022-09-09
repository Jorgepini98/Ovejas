
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

def dateFormatChange(date):
        year = 20*100 + int(date[:2])
        month = int(date[3:5])
        day = int(date[6:])
        return year,month,day

def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

directorio = "D:/DatosOvejas/"

path = directorio + "OvejasFeaturesOver/"
files = os.listdir(path)
df = pd.DataFrame()

for file in files:

    oveja = file[0:4]

    print(oveja)

    sheep = pd.read_csv(path + file)

    sma = sheep["SMA"]
    svm = sheep["SVM"]

    dates = sheep["Date"]

    start_date = date(2022, 4, 1)
    end_date = date(2022, 9, 1)

    pos = 0
    datePos = 0
    date1 = dates[pos]

    for single_date in daterange(start_date, end_date):
            datePos += 1

    totalSMA = [0]*datePos
    totalSVM = [0]*datePos
    axisDates = [0]*datePos

    datePos = 0

    for single_date in daterange(start_date, end_date):

        year,month,day = (dateFormatChange(date1))
        comparingDate = date(year,month,day)
        axisDates[datePos] = single_date
        #print(comparingDate)

        while comparingDate == single_date and pos < len(dates) - 1:
            pos += 1
            date1 = dates[pos]
            year,month,day = dateFormatChange(date1)
            comparingDate = date(year,month,day)

            totalSMA[datePos] += sma[pos]
            totalSVM[datePos] += svm[pos]

        datePos += 1

    print(len(axisDates))

    # fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

    # plt.plot(axisDates,counter0)
    # plt.gcf().autofmt_xdate()
    # plt.show()
    
    df[str(oveja)] = totalSMA
    #df["SVM"] = totalSVM

df["dates"] = axisDates

import plotly.express as px

fig = px.line(df, x="dates", y=df.columns,
              hover_data={"dates": "|%B %d, %Y"},
              title='custom tick labels')
fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")
fig.show()

fig.write_html(directorio + "figuras/" + "sheepEnergy.html")