import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt  
from sklearn.utils import shuffle

def getName(filePath):
    return filePath.split("\\")[-1]

def importDataInfo(path):
    coloumns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names = coloumns)
    #print(data.head())
    #print(data["Center"][0])
    #print(getName(data["Center"][0]))
    data["Center"] = data["Center"].apply(getName)
    #print(data.head())
    print("total images imported:", data.shape[0])
    return data

def balanceData(data, display = True):
    nBins = 31  # tek sayı olmalı
    samplesPerBin = 500
    hist, bins = np.histogram(data["Steering"], nBins)
    #print(bins)
    center = (bins[:-1] + bins[1:])*0.5
    if display: 
        #print(center)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1),(samplesPerBin, samplesPerBin))
        plt.show()
    
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data["Steering"])):
            if data["Steering"][i] >= bins[j] and data["Steering"][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[500:]
        removeIndexList.extend(binDataList)
    print("removed images:", len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print("remaining images:", len(data))
   
    if display: 
        hist, _ = np.histogram(data["Steering"], nBins)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1),(samplesPerBin, samplesPerBin))
        plt.show()
    return data