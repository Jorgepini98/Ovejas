from tracemalloc import stop
from numpy import true_divide, vstack
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
from datetime import datetime, timedelta

def isAfter(date1,date2):
    #format date: yy_mm_dd
    #returns 1 if date1 is after date2, 0 otherwise
    
    true = 0
    
    year1 = int(date1[0])*10 + int(date1[1])
    month1 = int(date1[3])*10 + int(date1[4])
    day1 = int(date1[6])*10 + int(date1[7])
    
    year2 = int(date2[0])*10 + int(date2[1])
    month2 = int(date2[3])*10 + int(date2[4])
    day2 = int(date2[6])*10 + int(date2[7])
    
    if year1 > year2:
        true = 1
    elif year1 == year2:
    
        if month1 > month2:
            true = 1
        elif month1 == month2:
        
            if day1 > day2:
                true = 1
                            
    return true

# result = pd.read_csv("resultado/" + "4990" + ".csv")

# result = result.iloc[:,1:]

# result = result[["Date","Hour","Minutes","Seconds"]].iloc[1,:]

# print(result)

# dictionaryTest = {"02_02_2022":[4,5],"03_02_2022":[6,7]}

# date = "04_02_2022"


# dictionaryTest[date] = [0,9]

# x = [2,4]

# x = np.vstack((x,[4,5]))

# print(x[:,1])


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


end_date = datetime(2022,8,1,1)

start_date = datetime(2022,4,1)

box = { "box1" : ["4990","5005","5017","5041","5056","5060","5063","5221"],
        "box2" : ["5059","4988","5018","5023","5024","5044","5054","5078"],
        "box3" : ["4996","4999","5012","5014","5026","5038","5046","4995"]
    }

box1 = pd.DataFrame({"Box":["1","1","1","1","1","1","1","1"],"Sheep":["4990","5005","5017","5041","5056","5060","5063","5221"]})
box2 = pd.DataFrame({"Box":["2","2","2","2","2","2","2","2"],"Sheep":["5059","4988","5018","5023","5024","5044","5054","5078"]})
box3 = pd.DataFrame({"Box":["3","3","3","3","3","3","3","3"],"Sheep":["4996","4999","5012","5014","5026","5038","5046","4995"]})

sheep_box = pd.concat([box1, box2, box3], axis=0)

#print(box)

# sheep_data = pd.read_csv(path + sheep_box.iloc[sheep_pos,1] + '.csv')

path  = 'resultado/'

out_path = 'estadistico/'

pos = 0

sheep_pos = 0
        
for sheep_pos in range(sheep_box.shape[0]):

    print("Sheep: " + sheep_box.iloc[sheep_pos,1])

    #se indica a que box corresponde cada oveja
    # if sheep_box.iloc[sheep_pos,0] == "1":
    #     box = "1"
    # elif sheep_box.iloc[sheep_pos,0] == "2":
    #     box = "2"
    # else:
    #     box = "3"

    skip = False

    try:
        sheep_data = pd.read_csv(path + sheep_box.iloc[sheep_pos,1] + '.csv')
    except:
        skip = True

    hour_features = pd.DataFrame()
    day_features = pd.DataFrame()
    for feature_type in sheep_data.columns[6:]:
        hour_features[feature_type] = 1
        day_features[feature_type] = 1

    skip = False

    #En caso de que el archivo este vacio, se evita entrar al calculo
    try:
        dates = sheep_data["Date"]
        hours = sheep_data["Hour"]
    except:
        skip = True

    if not skip:

        cont = False

        day_median_features = pd.DataFrame()
        day_25Q_features = pd.DataFrame()
        day_75Q_features = pd.DataFrame()
        day_SD_features = pd.DataFrame()

        total_hour_median_features = pd.DataFrame()
        total_hour_25Q_features = pd.DataFrame()
        total_hour_75Q_features = pd.DataFrame()
        total_hour_SD_features = pd.DataFrame()

        # se define un rango de fechas sobre las que se va trabajar
        for single_date in daterange(start_date, end_date): 
            #print (single_date.strftime("%Y-%m-%d"))

            date = single_date.strftime("%y_%m_%d")

            #print(date)

            if not cont:
                pos = 0
            else:
                pos = pos - 10 #por si acaso me salto alguna

            sheep_date = dates[pos] 

            #busca la fecha
            while isAfter(date,sheep_date) and pos < len(dates) - 1:

                pos += 1 #lo pongo antes dado que ya cojo el primer valor justo antes del while
                sheep_date = dates[pos]
                #print("Position: " + str(pos))
                
            cont = False
            if (sheep_date == date):
                cont = True

                day_features = day_features.iloc[0:0]

                hour_median_features = pd.DataFrame()
                hour_SD_features = pd.DataFrame()
                hour_25Q_features = pd.DataFrame()
                hour_75Q_features = pd.DataFrame()

                for hour in range(0,24):

                    hour_features = hour_features.iloc[0:0]
                    sheep_hour = hours[pos]
                    prev_pos = pos

                    while sheep_hour < hour and sheep_date == date and pos < len(dates) - 1:

                        pos += 1
                        sheep_date = dates[pos]
                        sheep_hour = hours[pos]

                    if sheep_hour == hour and sheep_date == date:

                        n_hour_sample = 0

                        #numero de medidas posible totales en una hora -> 4 por minuto, 240 por hora, sino llega a 240, 
                        # para calcular la mediana

                        while sheep_hour == hour and sheep_date == date and pos < len(dates) - 1:

                            for n_feature in range(0,hour_features.shape[1]):

                                hour_features.at[n_hour_sample, hour_features.columns[n_feature]] = sheep_data.iloc[pos,n_feature + 6]

                            prev_hour = sheep_hour
                            pos += 1
                            n_hour_sample += 1
                            sheep_hour = hours[pos]
                            sheep_date = dates[pos]
                        
                            if prev_hour == hour and sheep_hour != hour:

                                for n_hour_sample in range(n_hour_sample,240):
                                    for n_feature in range(0,hour_features.shape[1]):
                                        hour_features.at[n_hour_sample, hour_features.columns[n_feature]] = 0

                                for feature in hour_features.columns:
                                
                                    hour_median_features.at[hour,feature] = np.percentile(hour_features[feature],50)
                                    hour_25Q_features.at[hour,feature] = np.percentile(hour_features[feature],25) 
                                    hour_75Q_features.at[hour,feature] = np.percentile(hour_features[feature],75) 
                                    hour_SD_features.at[hour,feature] = np.std(hour_features[feature])                        

                    else:
                        pos = prev_pos

                        sheep_date = dates[pos]
                        sheep_hour = hours[pos]

                        for feature in hour_features.columns:
                            hour_median_features.at[hour,feature] = 0
                            hour_25Q_features.at[hour,feature] = 0
                            hour_75Q_features.at[hour,feature] = 0
                            hour_SD_features.at[hour,feature] = 0
                        #prev_hour = sheep_hour

                    day_features = day_features.append(hour_features)

                addDate = pd.DataFrame()
                addDate['Date'] = [[single_date.strftime("%y_%m_%d")] for _ in range(24)]

                total_hour_median_features = total_hour_median_features.append(pd.concat([addDate,hour_median_features], axis=1))
                total_hour_25Q_features = total_hour_25Q_features.append(pd.concat([addDate,hour_25Q_features], axis=1))
                total_hour_75Q_features = total_hour_75Q_features.append(pd.concat([addDate,hour_75Q_features], axis=1))
                total_hour_SD_features = total_hour_SD_features.append(pd.concat([addDate,hour_SD_features], axis=1))

                for feature in day_features.columns:
                    day_median_features.at[single_date.strftime("%y_%m_%d"),feature] = np.percentile(day_features[feature],50)
                    day_25Q_features.at[single_date.strftime("%y_%m_%d"),feature] = np.percentile(day_features[feature],25) 
                    day_75Q_features.at[single_date.strftime("%y_%m_%d"),feature] = np.percentile(day_features[feature],75) 
                    day_SD_features.at[single_date.strftime("%y_%m_%d"),feature] = np.std(day_features[feature])

        
                # for feature in features.columns:
                #     day_median_features[feature] = [np.percentile(median_features[feature],50)] ##inicializar a cero features
                # print(day_median_features)
                
            #print("Position: " + str(pos))

    with pd.ExcelWriter(out_path + str(sheep_box.iloc[sheep_pos,1]) + '_day.xlsx') as writer:  
        day_median_features.to_excel(writer,sheet_name='median')
        day_25Q_features.to_excel(writer,sheet_name='25Q')
        day_75Q_features.to_excel(writer,sheet_name='75Q')
        day_SD_features.to_excel(writer,sheet_name='SD')

    with pd.ExcelWriter(out_path + str(sheep_box.iloc[sheep_pos,1]) + '_hour.xlsx') as writer:  
        total_hour_median_features.to_excel(writer,sheet_name='median')
        total_hour_25Q_features.to_excel(writer,sheet_name='25Q')
        total_hour_75Q_features.to_excel(writer,sheet_name='75Q')
        total_hour_SD_features.to_excel(writer,sheet_name='SD')

            



    

#creamos un dataframe para cada box, pensar como se hace....

                    # if box == "1":

                    #     box1_day = np.array([date])

                    # elif box == "2":
                    #     box = "2"
                    # else:
                    #     box = "3"