#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:12:57 2022

@author: chloedearman
"""

data.head()

# Drop the date column

data_1 = data.drop(['date'], axis=1)
data_1.head()

# Compute daily returns (percentage changes in price) for SPY, AAPL  
# Be sure to drop the first row of NaN  
# Hint: pandas has functions to easily do this

data_1['spy_change']=data_1['spy_adj_close'].pct_change()*100 # referenced pct_change documentation
data_1['aapl_change']=data_1['aapl_adj_close'].pct_change()*100
data_2 = data_1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) # reference drop.na documentation

#### 1. (1 PT) Print the first 5 rows of returns

data_2.head()

#Save AAPL, SPY returns into separate numpy arrays  
#### 2. (1 PT) Print the first five values from the SPY numpy array, and the AAPL numpy array

spy_array = data_2.iloc[:,2].values # referenced https://stackoverflow.com/questions/31789160/convert-select-columns-in-pandas-dataframe-to-numpy-array

aapl_array = data_2.iloc[:,3].values

print(spy_array[:5])
print(aapl_array[:5])

##### Compute the excess returns of AAPL, SPY by simply subtracting the constant *R_f* from the returns.
##### Specifically, for the numpy array containing AAPL returns, subtract *R_f* from each of the returns. Repeat for SPY returns.

#NOTE:  
#AAPL - *R_f* = excess return of Apple stock  
#SPY - *R_f* = excess return of stock market


aapl_excess = aapl_array - R_f
spy_excess = spy_array - R_f

#### 3. (1 PT) Print the LAST five excess returns from both AAPL, SPY numpy arrays


print(aapl_excess[-5:])
print(spy_excess[-5:])

#### 4. (1 PT) Make a scatterplot with SPY excess returns on x-axis, AAPL excess returns on y-axis####
#Matplotlib documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

import matplotlib 
from matplotlib import pyplot
matplotlib.pyplot.scatter(spy_excess,aapl_excess)

#### 5. (3 PTS) Use Linear Algebra (matrices) to Compute the Regression Coefficient Estimate, \\(\hat\beta_i\\)

#Hint 1: Here is the matrix formula where *x′* denotes transpose of *x*.

#\begin{aligned} \hat\beta_i=(x′x)^{−1}x′y \end{aligned} 

#Hint 2: consider numpy functions for matrix multiplication, transpose, and inverse. Be sure to review what these operations do, and how they work, if you're a bit rusty.

spy_trans = np.transpose(spy_excess, axes=None) # referenced np.transpose documentation
spy_matmul = np.matmul(spy_trans, spy_excess) # referenced https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

spy_inv = 1/spy_matmul

outer = spy_inv*spy_trans
coeff = np.matmul(outer, aapl_excess)
coeff

#You should have found that the beta estimate is greater than one.  
#This means that the risk of AAPL stock, given the data, and according to this particular (flawed) model,  
#is higher relative to the risk of the S&P 500.




#### Measuring Beta Sensitivity to Dropping Observations (Jackknifing)

#Let's understand how sensitive the beta is to each data point.   
#We want to drop each data point (one at a time), compute \\(\hat\beta_i\\) using our formula from above, and save each #measurement.

#### 6. (3 PTS) Write a function called `beta_sensitivity()` with these specs:

#- take numpy arrays x and y as inputs
#- output a list of tuples. each tuple contains (observation row dropped, beta estimate)

Hint: **np.delete(x, i).reshape(-1,1)** will delete observation i from array x, and make it a column vector

#from numpy import linalg 
def beta_sensivity(x,y):
    betas = []
    for i in range(len(x)):
        newx = np.array(x)
        newy= np.array(y)
        removex =  np.delete(newx, i).reshape(-1,1)
        removey =  np.delete(newy, i).reshape(-1,1)
        x_trans = np.transpose(removex, axes=None) 
        x_matmul = np.matmul(x_trans, removex) 

        x_inv = 1/x_matmul

        outer = x_inv*x_trans
        coeff = np.matmul(outer, removey)
        betas.append((i,coeff[0][0]))
    return betas

# worked with classmates

#### Call `beta_sensitivity()` and print the first five tuples of output.

beta_sensivity(spy_excess, aapl_excess)