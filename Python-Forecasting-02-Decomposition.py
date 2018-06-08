############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Forecasting
# Lesson: Decomposition

# Citation: 
# PEREIRA, V. (2018). Project: Association Rules, File: Python-Forecasting-02-Decomposition.py, GitHub repository: <https://github.com/Valdecy/Forecasting-02-Decomposition>

############################################################################

# Installing Required Libraries
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# Function: MA 
def moving_average(timeseries, n = 2):
    
    name = 'MA(' + str(n) + ')'
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    ma = pd.DataFrame(np.nan, index = timeseries.index, columns = [name])
    adjustment = 0
    vector = timeseries.isnull()
    for i in range(0, len(timeseries)):
        if (vector.iloc[i,0] == True):
            adjustment = adjustment + 1
        else:
            break
    
    center = int(n/2) + adjustment   
    start  = 0 + adjustment
    finish = n + adjustment
    
    if(n % 2 == 0):
        weights = [None]*(n)
        for i in range(0, n):
            weights[i] = 1/n
   
        for i in range(center, len(timeseries) - center):
            ma.iloc[i, 0] = 0
            for j in range(0, len(weights)):
                ma.iloc[i, 0]  = float(ma.iloc[i, 0] + timeseries.iloc[start + j,:]*weights[j])
            start  = start  + 1        
        last = ma.iloc[(start - 1),0]  
    else:
        for i in range(center, len(timeseries) - center):
            ma.iloc[i,0]  = float(timeseries.iloc[start:finish,:].sum()/n)
            start  = start  + 1
            finish = finish + 1         
        last = float(timeseries.iloc[(start-1):(finish-1),:].sum()/n)
    
    timeseries = timeseries.iloc[:,0]
    ma = ma.iloc[:,0] 
    
    return ma, last

def henderson_seasonal_adjustment(timeseries):
    n = 13
    name = 'HSA(' + str(n) + ')'
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    ma = pd.DataFrame(np.nan, index = timeseries.index, columns = [name])
    adjustment = 0
    vector = timeseries.isnull()
    for i in range(0, len(timeseries)):
        if (vector.iloc[i,0] == True):
            adjustment = adjustment + 1
        else:
            break
    
    center = int(n/2) + adjustment   
    start  = 0 + adjustment
    weights = [None]*(n)
    
    weights[0]  = -0.019
    weights[1]  = -0.028
    weights[2]  =  0.0
    weights[3]  =  0.066
    weights[4]  =  0.147
    weights[5]  =  0.214
    weights[6]  =  0.240
    weights[7]  =  0.214
    weights[8]  =  0.147
    weights[9]  =  0.066
    weights[10] =  0.0
    weights[11] = -0.028
    weights[12] = -0.019
   
    for i in range(center, len(timeseries) - center):
        ma.iloc[i, 0] = 0
        for j in range(0, len(weights)):
            ma.iloc[i, 0]  = float(ma.iloc[i, 0] + timeseries.iloc[start + j,:]*weights[j])
        start  = start  + 1        
    last = ma.iloc[(start - 1),0]  
    
    timeseries = timeseries.iloc[:,0]
    ma = ma.iloc[:,0]  
    
    return ma, last
    ############### End of Function ##############
    
# Function: X-11
def x_11(X, graph = True): 
    
    step_00     = X.copy(deep = True)
    step_01, _  = moving_average(step_00, n = 12)
    step_02, _  = moving_average(step_01, n = 2)
    step_03     = step_00 / step_02
    step_04, _  = moving_average(step_03, n = 3)
    step_05, _  = moving_average(step_04, n = 3)
    step_06, _  = moving_average(step_05, n = 12)
    step_07, _  = moving_average(step_06, n = 2)
    step_08     = step_05 / step_07
    step_09     = step_00 / step_08
    step_10, _  = henderson_seasonal_adjustment(step_09)
    step_11     = step_00 / step_10
    step_12, _  = moving_average(step_11, n = 5)
    step_13, _  = moving_average(step_12, n = 3)
    step_14     = step_11 / step_13
    step_15     = step_00 / step_14
    
    
    step_16, _  = moving_average(step_15, n = 12)
    step_17, _  = moving_average(step_16, n = 2)
    step_18     = step_15 / step_17
    step_19     = step_00 - step_15

    if graph == True:
        style.use('ggplot')        
        
        ax1 = plt.subplot(4, 1, 1)
        plt.plot(X, color = 'black')
        plt.ylabel('Timeseries')
        plt.title('Decomposition')
        
        plt.subplot(4, 1, 2, sharex = ax1)
        plt.plot(step_18, color = 'blue')
        plt.ylabel('Seasonality')
        
        plt.subplot(4, 1, 3, sharex = ax1)
        plt.plot(step_17, color = 'green')
        plt.ylabel('Trend')
        
        plt.subplot(4, 1, 4, sharex = ax1)
        plt.bar(X.index, step_19, width = 7, color = 'red')
        plt.ylabel('Error')
        plt.show()
        
        step_17 = step_17.rename("Trend")
        step_18 = step_18.rename("Seasonality")
        step_19 = step_19.rename("Error")
  
    return step_18, step_17, step_19
    
######################## Part 2 - Usage ####################################

# Load Dataset    
df = pd.read_csv('Python-Forecasting-02-Dataset.txt', sep = '\t')

# Transforming the Dataset to a Time Series
X = df.iloc[:,:]
X = X.set_index(pd.DatetimeIndex(df.iloc[:,0])) # First column as row names
X = X.iloc[:,1]

# Calling Functions
decomposition = x_11(X, graph = True)

########################## End of Code #####################################