# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:54:37 2017

@author: Crissy
"""

import numpy as np
import matplotlib.pyplot as plt
import urllib
from scipy.integrate import odeint
from scipy.optimize import curve_fit

#imports data
bacteria_data= urllib.request.urlopen("http://nemenmanlab.org/~ilya/images/d/d8/Bacteria.txt")
data_set=np.loadtxt(bacteria_data, delimiter=',')
data_set_sorted = data_set[data_set[:, 0].argsort()]

#splits up the data into an array of time values and an 
#array of cooresponding bacteria concentration
N=np.size(data_set)
Time=data_set_sorted[0:N:1,0]
Bacteria_Concentration=data_set_sorted[0:N:1,1]
tf = 150

#graphs the bacteria concentration over time
plt.subplot(2,1,1)
plt.scatter(Time,Bacteria_Concentration)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria[bacteria/ml]')
plt.title('Bacteria Concentration Over Time')
plt.show()
#graphs the log of the bacteria concentration over time
plt.subplot(2,1,2)
t_data = np.transpose(data_set_sorted)[0]
n_data = np.log(np.transpose(data_set_sorted)[1])
n_original = np.transpose(data_set_sorted)[1]
t = np.linspace(0, tf, tf +1).flatten()
plt.scatter(Time, n_data)
plt.xlabel('Time[Hours]')
plt.ylabel('Log of Bacteria [log bacteria/ml]')
plt.title('Log of Bacteria Concentration over Time')
plt.figure()

#initial glucose and bacteria concentrations
init = [0.2, 50]

# k = Monod constant = p*10^-100
k_guess = 0.007

# mu = bacteria death rate
B = t_data[41:52]
B = np.vstack([np.ones(B.size), B]).T
y_intercept, slope = np.linalg.lstsq(B, n_data[41:52])[0]
mu_guess= -slope

# gmax = maximum bacteria growth rate
C = t_data[0:29]
C = C.reshape((C.size, 1))
z = n_data[0:29]
z = z.reshape(z.size, 1)
slope2 = float(np.linalg.lstsq(C, z)[0])
gmax_guess = slope2 + mu_guess

# a = CFU/mg*ml

a_guess = (mu_guess * Bacteria_Concentration[41] / (0.2/38))

timearray = np.arange(0, 52, 1).flatten()
# Monod law = bacteria growth rate
def diff_eq(values, timearray, gmax, mu, k, a):
    p = values[0]
    n = values[1]
            # n = bacteria concentration
    dn_dt = n * gmax * p / (p+k) - mu*n
            # p = glucose concentration
    dp_dt = - n * gmax * p / (p+k) * (1/a)
    return dp_dt, dn_dt
    
def odeint_diff_eq(t, gmax, mu, k, a):
    return odeint(diff_eq, init, timearray, args = (gmax, mu, k, a))[0:N:1,1]

def log_odeint_diff_eq(timearray, gmax, mu, k, a):
    return np.log(odeint(diff_eq, init, timearray, args = (gmax, mu, k, a)))[0:N:1,1]
    
gmax1, mu1, k1, a1 = curve_fit(log_odeint_diff_eq, t_data, n_data, p0 = (gmax_guess, mu_guess, k_guess, a_guess), bounds=([0,0,0,0], [1, 1, 1, 10000000000]))[0]

#equations redefined to fit a larger time period
timearray2 = np.arange(4.5, 108.5, 1).flatten()
def diff_eq2(values, t, gmax, mu, k, a):
    p = values[0]
    n = values[1]
            # n = bacteria concentration
    dn_dt = n * gmax * p / (p+k) - mu*n
            # p = glucose concentration
    dp_dt = - n * gmax * p / (p+k) * (1/a)
    return dp_dt, dn_dt
    
def odeint_diff_eq2(t, gmax, mu, k, a):
    return odeint(diff_eq, init, timearray2, args = (gmax, mu, k, a))[0:N:1,1]

def log_odeint_diff_eq2(t, gmax, mu, k, a):
    return np.log(odeint(diff_eq, init, timearray2, args = (gmax, mu, k, a)))[0:N:1,1]

#graph odeint with parameter values from curve fit for bacteria vs. time graph
plt.plot(timearray2, log_odeint_diff_eq2(t, gmax1, mu1, k1, a1), label='Log fit')
plt.xlim([0, 100])
plt.title('Log of Bacteria Concentration over Time')
plt.xlabel('Time[Hours]')
plt.ylabel('Log of Bacteria Concentration [log bacteria/ml]')
plt.scatter(Time, n_data, label ='Log-Original data')
plt.legend()
plt.show()
plt.figure()

#polynomial fit
plt.plot(t_data, n_data, 'o', label='Log-Original data', markersize=4)
A = np.vstack([np.ones(len(t_data)), t_data, t_data**2, t_data**3, t_data**4, t_data**5]).T
a, b, c, d, e, f = np.linalg.lstsq(A, n_data)[0]
plt.plot(t, a+b*t+c*t**2+d*t**3+e*t**4+f*t**5, 'r', label='Fitted line')
plt.legend(loc=4)
plt.title('Log of Bacteria Concentration over Time')
plt.xlabel('Time[Hours]')
plt.ylabel('Log of Bacteria Concentration [log bacteria/ml]')
plt.show()
plt.figure()

#model verification
plt.subplot(2,3,1)
plt.plot(timearray2, odeint_diff_eq2(t, gmax1, mu1, k1, a1), label='Growth Model')
plt.xlim([0, 100])
plt.title('Bacteria Concentration over Time with the Growth Model', size=12)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria Concentration [bacteria/ml]')
plt.scatter(Time, Bacteria_Concentration, label = "Bacteria Data")
plt.legend()
plt.show()
#gmax
plt.subplot(2,3,2)
plt.plot(timearray2, odeint_diff_eq2(t, 100*gmax1, mu1, k1, a1), label='Growth Model')
plt.xlim([0, 100])
plt.title('Bacteria Concentration over Time with gmax multiplied by 100', size=10)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria Concentration [bacteria/ml]')
plt.scatter(Time, Bacteria_Concentration, label = "Bacteria Data")
plt.legend()
plt.show()
#mu
plt.subplot(2,3,3)
plt.plot(timearray2, odeint_diff_eq2(t, gmax1, 100*mu1, k1, a1), label='Growth Model')
plt.xlim([0, 100])
plt.title('Bacteria Concentration over Time with mu multiplied by 100', size=10)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria Concentration [bacteria/ml]')
plt.scatter(Time, Bacteria_Concentration, label = "Bacteria Data")
plt.legend()
plt.show()
#k
plt.subplot(2,3,4)
plt.plot(timearray2, odeint_diff_eq2(t, gmax1, mu1, 100000*k1, a1), label='Growth Model')
plt.xlim([0, 100])
plt.title('Bacteria Concentration over Time with k multiplied by 100', size=10)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria Concentration [bacteria/ml]')
plt.scatter(Time, Bacteria_Concentration, label = "Bacteria Data")
plt.legend()
plt.show()
#a
plt.subplot(2,3,5)
plt.plot(timearray2, odeint_diff_eq2(t, gmax1, mu1, k1,100*a1), label='Growth Model')
plt.xlim([0, 100])
plt.title('Bacteria Concentration over Time with a multiplied by 100', size=10)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria Concentration [bacteria/ml]')
plt.scatter(Time, Bacteria_Concentration, label = "Bacteria Data")
plt.legend()
plt.show()
