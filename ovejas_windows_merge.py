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
import tsfresh

def isAfter(date1,date2):
    #format date: [yy_mm_dd,hh,mm,ss]
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

def getTime1(string):
    hour = string[0:2]
    min = string[3:5]
    sec = string[6:8]
    date = string[9:17]
    return date,hour,min,sec

def getTime2(string):
    hour = string[18:20]
    min = string[21:23]
    sec = string[24:26]
    date = string[27:35]
    return date,hour,min,sec

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

    for x in range(int(pos[len(pos) - 2]), int(pos[len(pos) - 1]) - 1):

            sensor = sensor*10 + int(path[x])

    return sensor

def getFeatures(data):

    low = signal.butter(3, 0.5, 'low', output='sos')
    hp = signal.butter(3, 0.5, 'hp', output='sos',fs = 20)

    Accel_x = data.iloc[:,2]
    Accel_y = data.iloc[:,3]
    Accel_z = data.iloc[:,4]

    #print(Accel_x)

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

    total_Accel = []

   
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

    for y in range(50): ##para evitar pico del filtro replico el pirmer valor 50 veces y posteriormente cogere los ultimos 200 valores correpsondientes a las posiciones de los valores originales
        modified_Accel_x.append(Accel_x[0])
        modified_Accel_y.append(Accel_y[0])
        modified_Accel_z.append(Accel_z[0])


    modified_Accel_x = np.append(modified_Accel_x,Accel_x)
    modified_Accel_y = np.append(modified_Accel_y,Accel_y)
    modified_Accel_z = np.append(modified_Accel_z,Accel_z)


    #low pass filter
    Accel_x_low = signal.sosfilt(low,modified_Accel_x)[50:250] - np.mean(modified_Accel_x)
    Accel_y_low = signal.sosfilt(low,modified_Accel_y)[50:250] - np.mean(modified_Accel_y)
    Accel_z_low = signal.sosfilt(low,modified_Accel_z)[50:250] - np.mean(modified_Accel_z)

    #high pass filter
    Accel_x_hp = signal.sosfilt(hp,modified_Accel_x)[50:250]
    Accel_y_hp = signal.sosfilt(hp,modified_Accel_y)[50:250]
    Accel_z_hp = signal.sosfilt(hp,modified_Accel_z)[50:250]

    # ax3.plot(Accel_x_hp)
    # ax3.set_title('butter')

    #plt.show()
    y = 0

    t = time.time()

    for y in range(Accel_x_hp.shape[0]):

        SMA = abs(Accel_x_hp[y]) + abs(Accel_y_hp[y]) + abs(Accel_z_hp[y])
        SVM = math.sqrt(pow(Accel_x_hp[y],2) + pow(Accel_y_hp[y],2) + pow(Accel_z_hp[y],2))
        total_Accel.append(abs(Accel_x_hp[y]) + abs(Accel_y_hp[y]) + abs(Accel_z_hp[y]))


    #SMA = round(SMA/y, 4)
    #SVM = round(SVM/y, 4)

    a_abs_energy = tsfresh.feature_extraction.feature_calculators.abs_energy(total_Accel)

    a_maximum = tsfresh.feature_extraction.feature_calculators.absolute_maximum(Accel_x_hp)

    a_entropy = tsfresh.feature_extraction.feature_calculators.approximate_entropy(total_Accel,10,3)

    a_x_autocorrelation = tsfresh.feature_extraction.feature_calculators.autocorrelation(Accel_x_hp,2) ##lag
    a_y_autocorrelation = tsfresh.feature_extraction.feature_calculators.autocorrelation(Accel_y_hp,2)
    a_z_autocorrelation = tsfresh.feature_extraction.feature_calculators.autocorrelation(Accel_z_hp,2)
    a_autocorrelation = tsfresh.feature_extraction.feature_calculators.autocorrelation(total_Accel,2)

    uno = {"a_x_autocorrelation":[a_x_autocorrelation],"a_y_autocorrelation":[a_y_autocorrelation],"a_z_autocorrelation":[a_z_autocorrelation],"a_autocorrelation":[a_autocorrelation]}

    a_x_benfordCorrelation = tsfresh.feature_extraction.feature_calculators.benford_correlation(Accel_x_hp)
    a_y_benfordCorrelation = tsfresh.feature_extraction.feature_calculators.benford_correlation(Accel_y_hp)
    a_z_benfordCorrelation = tsfresh.feature_extraction.feature_calculators.benford_correlation(Accel_z_hp)
    a_benfordCorrelation = tsfresh.feature_extraction.feature_calculators.benford_correlation(total_Accel)

    dos = {"a_x_benfordCorrelation":[a_x_benfordCorrelation],"a_y_benfordCorrelation":[a_y_benfordCorrelation],"a_z_benfordCorrelation":[a_z_benfordCorrelation],"a_benfordCorrelation":[a_benfordCorrelation]}

    a_x_c3 = tsfresh.feature_extraction.feature_calculators.c3(Accel_x_hp,10)
    a_y_c3 = tsfresh.feature_extraction.feature_calculators.c3(Accel_y_hp,10)
    a_z_c3 = tsfresh.feature_extraction.feature_calculators.c3(Accel_z_hp,10)
    a_c3 = tsfresh.feature_extraction.feature_calculators.c3(total_Accel,10)

    tres = {"a_x_c3":[a_x_c3],"a_y_c3":[a_y_c3],"a_z_c3":[a_z_c3],"a_c3":[a_c3]}

    a_x_cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce(Accel_x_hp,False)
    a_y_cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce(Accel_y_hp,False)
    a_z_cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce(Accel_z_hp,False)
    a_cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce(total_Accel,False)

    cuatro = {"a_x_cid_ce":[a_x_cid_ce],"a_y_cid_ce":[a_y_cid_ce],"a_z_cid_ce":[a_z_cid_ce],"a_cid_ce":[a_cid_ce]}

    ##cwt_coeficients

    # s = {"aggtype": {"centroid":None}}

    # x_ftt_centroid = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_x_hp, ["centroid", "variance", "skew", "kurtosis"])
    # y_ftt_centroid = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_y_hp,"centroid")
    # z_ftt_centroid = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_z_hp,"centroid")
    # ftt_centroid = tsfresh.feature_extraction.feature_calculators.fft_aggregated(total_Accel,"centroid")

    # x_ftt_variance = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_x_hp,"variance")
    # y_ftt_variance = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_y_hp,"variance")
    # z_ftt_variance = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_z_hp,"variance")
    # ftt_variance = tsfresh.feature_extraction.feature_calculators.fft_aggregated(total_Accel,"variance")

    # x_ftt_skew = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_x_hp,"skew")
    # y_ftt_skew = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_y_hp,"skew")
    # z_ftt_skew = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_z_hp,"skew")
    # ftt_skew = tsfresh.feature_extraction.feature_calculators.fft_aggregated(total_Accel,"skew")

    # x_ftt_kurtosis = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_x_hp,"kurtosis")
    # y_ftt_kurtosis = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_y_hp,"kurtosis")
    # z_ftt_kurtosis = tsfresh.feature_extraction.feature_calculators.fft_aggregated(Accel_z_hp,"kurtosis")
    # ftt_kurtosis = tsfresh.feature_extraction.feature_calculators.fft_aggregated(total_Accel,"kurtosis")

    a_x_fourier_entropy = tsfresh.feature_extraction.feature_calculators.fourier_entropy(Accel_x_hp,10) ##bins
    a_y_fourier_entropy = tsfresh.feature_extraction.feature_calculators.fourier_entropy(Accel_y_hp,10)
    a_z_fourier_entropy = tsfresh.feature_extraction.feature_calculators.fourier_entropy(Accel_z_hp,10)
    a_fourier_entropy = tsfresh.feature_extraction.feature_calculators.fourier_entropy(total_Accel,10)

    cinco = {"a_x_fourier_entropy":[a_x_fourier_entropy],"a_y_fourier_entropy":[a_y_fourier_entropy],"a_z_fourier_entropy":[a_z_fourier_entropy],"a_fourier_entropy":[a_fourier_entropy]}

    a_x_kurtosis = tsfresh.feature_extraction.feature_calculators.kurtosis(Accel_x_hp)
    a_y_kurtosis = tsfresh.feature_extraction.feature_calculators.kurtosis(Accel_y_hp)
    a_z_kurtosis = tsfresh.feature_extraction.feature_calculators.kurtosis(Accel_z_hp)
    a_kurtosis = tsfresh.feature_extraction.feature_calculators.kurtosis(total_Accel)

    seis = {"a_x_kurtosis":[a_x_kurtosis],"a_y_kurtosis":[a_y_kurtosis],"a_z_kurtosis":[a_z_kurtosis],"a_kurtosis":[a_kurtosis]}

    a_x_skewness = tsfresh.feature_extraction.feature_calculators.skewness(Accel_x_hp)
    a_y_skewness = tsfresh.feature_extraction.feature_calculators.skewness(Accel_y_hp)
    a_z_skewness = tsfresh.feature_extraction.feature_calculators.skewness(Accel_z_hp)
    a_skewness = tsfresh.feature_extraction.feature_calculators.skewness(total_Accel)

    siete = {"a_x_skewness":[a_x_skewness],"a_y_skewness":[a_y_skewness],"a_z_skewness":[a_z_skewness],"a_skewness":[a_skewness]}

    a_x_last_location_of_maximum = tsfresh.feature_extraction.feature_calculators.last_location_of_maximum(Accel_x_hp)
    a_y_last_location_of_maximum = tsfresh.feature_extraction.feature_calculators.last_location_of_maximum(Accel_y_hp)
    a_z_last_location_of_maximum = tsfresh.feature_extraction.feature_calculators.last_location_of_maximum(Accel_z_hp)
    a_last_location_of_maximum = tsfresh.feature_extraction.feature_calculators.last_location_of_maximum(total_Accel)

    ocho = {"a_x_last_location_of_maximum":[a_x_last_location_of_maximum],"a_y_last_location_of_maximum":[a_y_last_location_of_maximum],"a_z_last_location_of_maximum":[a_z_last_location_of_maximum],"a_last_location_of_maximum":[a_last_location_of_maximum]}

    a_x_last_location_of_minimum = tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(Accel_x_hp)
    a_y_last_location_of_minimum = tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(Accel_y_hp)
    a_z_last_location_of_minimum = tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(Accel_z_hp)
    a_last_location_of_minimum = tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(total_Accel)

    nueve = {"a_x_last_location_of_minimum":[a_x_last_location_of_minimum],"a_y_last_location_of_minimum":[a_y_last_location_of_minimum],"a_z_last_location_of_minimum":[a_z_last_location_of_minimum],"a_last_location_of_minimum":[a_last_location_of_minimum]}

    a_x_mean = tsfresh.feature_extraction.feature_calculators.mean(Accel_x_hp)
    a_y_mean = tsfresh.feature_extraction.feature_calculators.mean(Accel_y_hp)
    a_z_mean = tsfresh.feature_extraction.feature_calculators.mean(Accel_z_hp)
    a_mean = tsfresh.feature_extraction.feature_calculators.mean(total_Accel)

    diez = {"a_x_mean":[a_x_mean],"a_y_mean":[a_y_mean],"a_z_mean":[a_z_mean],"a_mean":[a_mean]}

    a_x_mean_abs_change = tsfresh.feature_extraction.feature_calculators.mean_abs_change(Accel_x_hp)
    a_y_mean_abs_change = tsfresh.feature_extraction.feature_calculators.mean_abs_change(Accel_y_hp)
    a_z_mean_abs_change = tsfresh.feature_extraction.feature_calculators.mean_abs_change(Accel_z_hp)
    a_mean_abs_change = tsfresh.feature_extraction.feature_calculators.mean_abs_change(total_Accel)

    once = {"a_x_mean_abs_change":[a_x_mean_abs_change],"a_y_mean_abs_change":[a_y_mean_abs_change],"a_z_mean_abs_change":[a_z_mean_abs_change],"a_mean_abs_change":[a_mean_abs_change]}

    a_x_median = tsfresh.feature_extraction.feature_calculators.median(Accel_x_hp)
    a_y_median = tsfresh.feature_extraction.feature_calculators.median(Accel_y_hp)
    a_z_median = tsfresh.feature_extraction.feature_calculators.median(Accel_z_hp)
    a_median = tsfresh.feature_extraction.feature_calculators.median(total_Accel)

    doce = {"a_x_median":[a_x_median],"a_y_median":[a_y_median],"a_z_median":[a_z_median],"a_median":[a_median]}

    a_x_minimum = tsfresh.feature_extraction.feature_calculators.minimum(Accel_x_hp)
    a_y_minimum = tsfresh.feature_extraction.feature_calculators.minimum(Accel_y_hp)
    a_z_minimum = tsfresh.feature_extraction.feature_calculators.minimum(Accel_z_hp)
    a_minimum = tsfresh.feature_extraction.feature_calculators.minimum(total_Accel)

    trece = {"a_x_minimum":[a_x_minimum],"a_y_minimum":[a_y_minimum],"a_z_minimum":[a_z_minimum],"a_minimum":[a_minimum]}

    a_x_number_cwt_peaks= tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(Accel_x_hp,10)
    a_y_number_cwt_peaks= tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(Accel_y_hp,10)
    a_z_number_cwt_peaks = tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(Accel_z_hp,10)
    a_number_cwt_peaks = tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(total_Accel,10)

    catorce = {"a_x_number_cwt_peaks":[a_x_number_cwt_peaks],"a_y_number_cwt_peaks":[a_y_number_cwt_peaks],"a_z_number_cwt_peaks":[a_z_number_cwt_peaks],"a_number_cwt_peaks":[a_number_cwt_peaks]}

    a_x_number_peaks= tsfresh.feature_extraction.feature_calculators.number_peaks(Accel_x_hp,int(SMA/5))
    a_y_number_peaks= tsfresh.feature_extraction.feature_calculators.number_peaks(Accel_y_hp,int(SMA/4))
    a_z_number_peaks = tsfresh.feature_extraction.feature_calculators.number_peaks(Accel_z_hp,int(SMA/4))
    a_number_peaks = tsfresh.feature_extraction.feature_calculators.number_peaks(total_Accel,int(SMA))

    quince = {"a_x_number_peaks":[a_x_number_peaks],"a_y_number_peaks":[a_y_number_peaks],"a_z_number_peaks":[a_z_number_peaks],"a_number_peaks":[a_number_peaks]}

    a_x_quantile = tsfresh.feature_extraction.feature_calculators.quantile(Accel_x_hp,0.25)
    a_y_quantile = tsfresh.feature_extraction.feature_calculators.quantile(Accel_y_hp,0.25)
    a_z_quantile = tsfresh.feature_extraction.feature_calculators.quantile(Accel_z_hp,0.25)
    a_quantile = tsfresh.feature_extraction.feature_calculators.quantile(total_Accel,0.25)

    dieciseis = {"a_x_quantile":[a_x_quantile],"a_y_quantile":[a_y_quantile],"a_z_quantile":[a_z_quantile],"a_quantile":[a_quantile]}

    a_x_root_mean_square= tsfresh.feature_extraction.feature_calculators.root_mean_square(Accel_x_hp)
    a_y_root_mean_square= tsfresh.feature_extraction.feature_calculators.root_mean_square(Accel_y_hp)
    a_z_root_mean_square = tsfresh.feature_extraction.feature_calculators.root_mean_square(Accel_z_hp)
    a_root_mean_square = tsfresh.feature_extraction.feature_calculators.root_mean_square(total_Accel)

    dieciseis = {"a_x_root_mean_square":[a_x_root_mean_square],"a_y_root_mean_square":[a_y_root_mean_square],"a_z_root_mean_square":[a_z_root_mean_square],"a_root_mean_square":[a_root_mean_square]}

    a_x_standard_deviation= tsfresh.feature_extraction.feature_calculators.standard_deviation(Accel_x_hp)
    a_y_standard_deviation= tsfresh.feature_extraction.feature_calculators.standard_deviation(Accel_y_hp)
    a_z_standard_deviation = tsfresh.feature_extraction.feature_calculators.standard_deviation(Accel_z_hp)
    a_standard_deviation = tsfresh.feature_extraction.feature_calculators.standard_deviation(total_Accel)

    diecisiete = {"a_x_standard_deviation":[a_x_standard_deviation],"a_y_standard_deviation":[a_y_standard_deviation],"a_z_standard_deviation":[a_z_standard_deviation],"a_standard_deviation":[a_standard_deviation]}

    a_x_variance= tsfresh.feature_extraction.feature_calculators.variance(Accel_x_hp)
    a_y_variance= tsfresh.feature_extraction.feature_calculators.variance(Accel_y_hp)
    a_z_variance = tsfresh.feature_extraction.feature_calculators.variance(Accel_z_hp)
    a_variance = tsfresh.feature_extraction.feature_calculators.variance(total_Accel)

    diecisiete = {"a_x_variance":[a_x_variance],"a_y_variance":[a_y_variance],"a_z_variance":[a_z_variance],"a_variance":[a_variance]}

    features = {"SMA": [SMA], "SVM": [SVM], "Energy": [a_abs_energy], "Maximum":[a_maximum], "Entropy":[a_entropy]}

    features = pd.DataFrame({**features, **uno, **dos, **tres, **cuatro, **cinco, **seis, **siete, **ocho, **nueve, **diez, **once, **doce, **trece, **catorce, **quince, **dieciseis, **diecisiete})

    elapsed = time.time() - t

    #print(elapsed)

    return features

def getData(data):

    #print(data.shape[0])

    low = signal.butter(3, 0.5, 'low', output='sos')
    hp = signal.butter(3, 0.5, 'hp', output='sos',fs = 20)

    Accel_x = data.iloc[:,2]
    Accel_y = data.iloc[:,3]
    Accel_z = data.iloc[:,4]

    #print(Accel_x)

    Gyro_x = data.iloc[:,5]
    Gyro_y = data.iloc[:,6]
    Gyro_z = data.iloc[:,7]

    Magnet_x = data.iloc[:,8]
    Magnet_y = data.iloc[:,9]
    Magnet_z = data.iloc[:,10]

    AccelX_int = []
    AccelY_int = []
    AccelZ_int = []

    GyroX_int = []
    GyroY_int = []
    GyroZ_int = []

    MagnetX_int = []
    MagnetY_int = []
    MagnetZ_int = []

    # in order to avoid spike at filtering stage
    modified_Accel_x=[]
    modified_Accel_y=[]
    modified_Accel_z=[]

    modified_Gyro_x=[]
    modified_Gyro_y=[]
    modified_Gyro_z=[]

    modified_Magnet_x=[]
    modified_Magnet_y=[]
    modified_Magnet_z=[]

    for y in range(data.shape[0]):

        if y != 0:

            AccelX_int.append(int((Accel_x.iloc[y])))
            AccelY_int.append(int((Accel_y.iloc[y])))
            AccelZ_int.append(int((Accel_z.iloc[y])))

            GyroX_int.append(int((Gyro_x.iloc[y])))
            GyroY_int.append(int((Gyro_y.iloc[y])))
            GyroZ_int.append(int((Gyro_z.iloc[y])))

            MagnetX_int.append(int((Magnet_x.iloc[y])))
            MagnetY_int.append(int((Magnet_y.iloc[y])))
            MagnetZ_int.append(int((Magnet_z.iloc[y])))

            
    #rawData
    rawAccel_x = (AccelX_int)
    rawAccel_y = (AccelY_int)
    rawAccel_z = (AccelZ_int)

    rawGyro_x = (GyroX_int)
    rawGyro_y = (GyroY_int)
    rawGyro_z = (GyroZ_int)

    rawMagnet_x = (MagnetX_int)
    rawMagnet_y = (MagnetY_int)
    rawMagnet_z = (MagnetZ_int)

    # MEdian filter
    Accel_x = signal.medfilt(AccelX_int)
    Accel_y = signal.medfilt(AccelY_int)
    Accel_z = signal.medfilt(AccelZ_int)

    Gyro_x = signal.medfilt(GyroX_int)
    Gyro_y = signal.medfilt(GyroY_int)
    Gyro_z = signal.medfilt(GyroZ_int)

    Magnet_x = signal.medfilt(MagnetX_int)
    Magnet_y = signal.medfilt(MagnetY_int)
    Magnet_z = signal.medfilt(MagnetZ_int)

    # ax2.plot(Accel_x)
    # ax2.set_title('median')

    y = 0

    for y in range(50): ##para evitar pico del filtro replico el pirmer valor 50 veces y posteriormente cogere los ultimos 200 valores correpsondientes a las posiciones de los valores originales
        modified_Accel_x.append(Accel_x[0])
        modified_Accel_y.append(Accel_y[0])
        modified_Accel_z.append(Accel_z[0])

        modified_Gyro_x.append(Gyro_x[0])
        modified_Gyro_y.append(Gyro_y[0])
        modified_Gyro_z.append(Gyro_z[0])

        modified_Magnet_x.append(Magnet_x[0])
        modified_Magnet_y.append(Magnet_y[0])
        modified_Magnet_z.append(Magnet_z[0])


    modified_Accel_x = np.append(modified_Accel_x,Accel_x)
    modified_Accel_y = np.append(modified_Accel_y,Accel_y)
    modified_Accel_z = np.append(modified_Accel_z,Accel_z)

    modified_Gyro_x = np.append(modified_Gyro_x,Gyro_x)
    modified_Gyro_y = np.append(modified_Gyro_y,Gyro_y)
    modified_Gyro_z = np.append(modified_Gyro_z,Gyro_z)

    modified_Magnet_x = np.append(modified_Magnet_x,Magnet_x)
    modified_Magnet_y = np.append(modified_Magnet_y,Magnet_y)
    modified_Magnet_z = np.append(modified_Magnet_z,Magnet_z)


    #low pass filter
    Accel_x_low = signal.sosfilt(low,modified_Accel_x)[50:250] - np.mean(modified_Accel_x)
    Accel_y_low = signal.sosfilt(low,modified_Accel_y)[50:250] - np.mean(modified_Accel_y)
    Accel_z_low = signal.sosfilt(low,modified_Accel_z)[50:250] - np.mean(modified_Accel_z)

    #high pass filter
    Accel_x_hp = signal.sosfilt(hp,modified_Accel_x)[50:250]
    Accel_y_hp = signal.sosfilt(hp,modified_Accel_y)[50:250]
    Accel_z_hp = signal.sosfilt(hp,modified_Accel_z)[50:250]

    #low pass filter
    Gyro_x_low = signal.sosfilt(low,modified_Gyro_x)[50:250] - np.mean(modified_Gyro_x)
    Gyro_y_low = signal.sosfilt(low,modified_Gyro_y)[50:250] - np.mean(modified_Gyro_y)
    Gyro_z_low = signal.sosfilt(low,modified_Gyro_z)[50:250] - np.mean(modified_Gyro_z)

    #high pass filter
    Gyro_x_hp = signal.sosfilt(hp,modified_Gyro_x)[50:250]
    Gyro_y_hp = signal.sosfilt(hp,modified_Gyro_y)[50:250]
    Gyro_z_hp = signal.sosfilt(hp,modified_Gyro_z)[50:250]

    #low pass filter
    Magnet_x_low = signal.sosfilt(low,modified_Magnet_x)[50:250] - np.mean(modified_Magnet_x)
    Magnet_y_low = signal.sosfilt(low,modified_Magnet_y)[50:250] - np.mean(modified_Magnet_y)
    Magnet_z_low = signal.sosfilt(low,modified_Magnet_z)[50:250] - np.mean(modified_Magnet_z)

    #high pass filter
    Magnet_x_hp = signal.sosfilt(hp,modified_Magnet_x)[50:250]
    Magnet_y_hp = signal.sosfilt(hp,modified_Magnet_y)[50:250]
    Magnet_z_hp = signal.sosfilt(hp,modified_Magnet_z)[50:250]


    # AccelX_int = modified_Accel_x[50:250]
    # AccelY_int = modified_Accel_y[50:250]
    # AccelZ_int = modified_Accel_z[50:250]

    # GyroX_int = modified_Gyro_x[50:250]
    # GyroY_int = modified_Gyro_y[50:250]
    # GyroZ_int = modified_Gyro_z[50:250]

    # MagnetX_int = modified_Magnet_x[50:250]
    # MagnetY_int = modified_Magnet_y[50:250]
    # MagnetZ_int = modified_Magnet_z[50:250]
    rawAccel_x = (AccelX_int)
    rawAccel_y = (AccelY_int)
    rawAccel_z = (AccelZ_int)

    rawGyro_x = (GyroX_int)
    rawGyro_y = (GyroY_int)
    rawGyro_z = (GyroZ_int)

    rawMagnet_x = (MagnetX_int)
    rawMagnet_y = (MagnetY_int)
    rawMagnet_z = (MagnetZ_int)


    #features = pd.DataFrame({"Acel_x": AccelX_int, "Acel_y": AccelY_int, "Acel_z": AccelZ_int, "Gyro_x": GyroX_int, "Gyro_y": GyroY_int, "Gyro_z": GyroZ_int, "Magnet_x":MagnetX_int, "Magnet_y": MagnetY_int, "Magnet_z": MagnetZ_int})
        
    features = pd.DataFrame({'R_Accel_x':rawAccel_x,'R_Accel_y':rawAccel_y,'R_Accel_z':rawAccel_z,'R_Gyro_x':rawGyro_x,'R_Gyro_y':rawGyro_y,'R_Gyro_z':rawGyro_z,'R_Magnet_x':rawMagnet_x,'R_Magnet_y':rawMagnet_y,'R_Magnet_z':rawMagnet_z,'l_Accel_x':Accel_x_low,'l_Accel_y':Accel_y_low,'l_Accel_z':Accel_z_low,'l_Gyro_x':Gyro_x_low,'l_Gyro_y':Gyro_y_low,'l_Gyro_z':Gyro_z_low,'l_Magnet_x':Magnet_x_low,'l_Magnet_y':Magnet_y_low,'l_Magnet_z':Magnet_z_low,'h_Accel_x':Accel_x_hp,'h_Accel_y':Accel_y_hp,'h_Accel_z':Accel_z_hp,'h_Gyro_x':Gyro_x_hp,'h_Gyro_y':Gyro_y_hp,'h_Gyro_z':Gyro_z_hp,'h_Magnet_x':Magnet_x_hp,'h_Magnet_y':Magnet_y_hp,'h_Magnet_z':Magnet_z_hp})


    #print(elapsed)

    return features

def readDataFrame(file, df, lastDay, windowDiv, overlapTime):
    ##reads dataframe from ".csv" file

    count = 0

    period = 0.05
    totalLength = 200
    windowLength = round(totalLength/windowDiv)
    overlapLength = round(overlapTime/period)
    totalWindows = round(totalLength/(windowLength - overlapLength) - 1)
    currentSec = 0
    results = pd.DataFrame()
    data = pd.DataFrame

    Sensor = readSensorNum(file)
    
    for x in range(df.shape[0]):  ##goes through the index file (df.index)

        newMov = df.iloc[x,0]

        if newMov[0:5] == 'HORA_': #gets next new movement

            # print(newMov)

            count = count + 1
            time = newMov[5:]

            date = readDate(file)
            hour = readHour(time)
            min = readMinutes(time)
            sec = readSeconds(time)

            if(int(date[:2]) > 20): #is verified that year is over 2020

                currentDay = [date,hour,min,sec]

                if isAfter(currentDay,lastDay):

                    #rawData = pd.DataFrame(df.loc[(x+1):(x+200)])

                    sec = int(sec)*100

                    for i in range(totalWindows):  ## it is iterated windowDiv times

                        rawDataDiv = pd.DataFrame(df.loc[(x + i*(windowLength - overlapLength)):(x + (i+2)*(windowLength - overlapLength))]) ## it is iterated windowDiv times

                        features = getFeatures(rawDataDiv)    #getFeatures(rawDataDiv)

                        if(i > 0):
                            sec += round((windowLength - overlapLength)*period*100) ##adjusts the current time

                        if(sec >= 6000 - 250): 
                            sec = sec - (6000 - 250)
                            min = str(int(min) + 1)

                        heading = pd.DataFrame({'Sensor': [Sensor], "Lenght": [windowLength], 'Date': [date], 'Hour': [hour], 'Minutes': [min], "Seconds": [str(sec)]})
                        
                        ##when raw data
                        # currentSec = sec

                        # for j in range(features.shape[0]):
                        #     if j > 0:
                        #         currentSec += period*100
                        #         heading = heading.append(pd.DataFrame({'Sensor': [Sensor], "Lenght": [windowLength], 'Date': [date], 'Hour': [hour], 'Minutes': [min], "Seconds": [str(currentSec)]}))
                            
                        # heading = heading.reset_index()

                        # heading = heading.drop(columns = ["index"])

                        # print(heading.shape[0])
                        # print(features.shape[0])
                        # print(features)
                        # print(heading)
                        ##when raw data

                        data = pd.concat([heading, features], axis=1) #, join='inner'

                        # print("Data")
                        #print(data)

                        # type(data)

                        if x == 0:
                            results = pd.DataFrame(data)
                        else:
                            results = results.append(data, ignore_index=True)
                    #print(results.to_string())
                        
                else:
                    print("Análisis ya realizado")

    # print("resultados")
    # print(results)

    return results

def readData(crotal, raw_path, features_path, orga):

    crotales = orga['crotal']

    print("CROTAL: ")
    print(crotal)

    pos1 = 0
    ## cant use df1.loc[df1.index[crotales.astype('str').str.contains(crotal)].tolist()]
    while crotal != crotales[pos1]:
        pos1 += 1

    sensors = orga.loc[pos1]
    fechas = sensors.index
    pos2 = 0
    finalFiles = []

    ##  Goes through all diferent dates' groups which are assigned to a specific sensor
    for sensor in sensors:
        if pos2 > 0:

            if sensor > 0:

                date1,hour1,min1,sec1 = getTime1(fechas[pos2])
                date2,hour2,min2,sec2 = getTime2(fechas[pos2])

                filePath = raw_path + str(pos2) + "/" + str(sensor) + "/"

                try:
                    iniFiles = os.listdir(filePath)
                except:
                    iniFiles = []

                for file in iniFiles:

                    if file.endswith(".csv"):

                        date = readDate(file)
                        hour = "00"
                        min = "00"
                        sec = "00"
                        date = [date,hour,min,sec]

                        if isAfter(date,[date1,hour1,min1,sec1]) and isAfter([date2,hour2,min2,sec2],date):
                            string = filePath + str(file)
                            
                            finalFiles.append(string)
    
        pos2 += 1
    #print(finalFiles)
    print("Files appended Crotal: "+ str(crotal))

    content = pd.DataFrame()
    existFile = 0
    lastDay = ["00_00_00","0","0","0"]

    filesRead = 0

    try:
        crotal_features = pd.read_csv(features_path + str(crotal) + ".csv")
        crotal_features = crotal_features.iloc[:, 1:]
        existFile = 1
        lastDay = crotal_features[["Date","Hour","Minutes","Seconds"]]
        lastDay = (lastDay.iloc[crotal_features.shape[0] - 1,:])
        crotal_features = pd.DataFrame()

    except:
        print("No hay .csv previo generado para el Crotal: " + str(crotal))
    

    # checking all the csv files in the
    # specified path
    for file in finalFiles:

        # reading content of csv file, which has been already checked that is .csv
        # content.append(filename)
        # df_firstn = pd.read_csv(filename, nrows=0, sep=';')

        df = pd.read_csv(file, sep=';', names = ['HORA','Timestamp','Accel_x','Accel_y','Accel_z','Gyro_x','Gyro_y','Gyro_z','Magnet_x','Magnet_y','Magnet_z'])

        if content.empty :
            content = readDataFrame(file, df, lastDay, 4, 1.25) #4 , 1.25
        else:
            content = content.append(readDataFrame(file, df, lastDay, 4, 1.25),  ignore_index=True) #4 , 1.25
        #print(content)
        # break
        print("Crotal: "+str(crotal)+" file: " + file)
        filesRead += 1

        if filesRead >= 24:
            filesRead = 0

            if existFile == 1:
                content.to_csv(features_path + str(crotal) + ".csv", mode='a', index=False, header=False)
                content = pd.DataFrame()
            else:
                if not content.empty:
                    content.to_csv(features_path + str(crotal) + ".csv", index=False)
                    existFile = 1
                    content = pd.DataFrame()

    print('#################################### ' + 'End of crotal: ' + str(crotal))
    # if existFile == 1:
    #     sensor_features = sensor_features.append(content)
    #     sensor_features.to_csv(features_path + str(sensor) + ".csv")
    # else:
    #     content.to_csv(features_path + str(sensor) + ".csv")
    



    


start = time.time()

organizadorPath  = 'Organizador.xlsx'

orga = pd.DataFrame()

orga = pd.read_excel(organizadorPath)

crotales = orga['crotal']

raw_path  = 'MedidasRaw/'

directorio = 'D:/DatosOvejas/'

features_path = directorio + 'OvejasFeaturesOver/' #D:/DatosOvejas/

n_jobs = multiprocessing.cpu_count() - 4

print(n_jobs)

#crotal = 4990
#readData(crotal, raw_path, features_path, orga)           ##   Debug Line
Parallel(n_jobs=n_jobs)(delayed(readData)(crotal, raw_path, features_path, orga) for crotal in crotales)

end = time.time()
print(end - start)




