# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:12:38 2017

@author: Crissy
"""

import numpy as np
import matplotlib.pyplot as plt
import urllib
from scipy.integrate import odeint
from scipy.optimize import curve_fit

bacteria_data= urllib.request.urlopen("http://nemenmanlab.org/~ilya/images/d/d8/Bacteria.txt")
data_set=np.loadtxt(bacteria_data, delimiter=',')

t_lag = 4.5
tf = 100
init_n = 50
init_p = 0
init_conc = [init_n, init_p]
 
t_data = np.transpose(data_set)[0]
n_data = np.log(np.transpose(data_set)[1])
n_original = np.transpose(data_set)[1]
t = np.linspace(0, tf, tf +1).flatten()

plt.plot(t_data, n_original, 'o', label='Original data', markersize=4)
A = np.vstack([np.ones(len(t_data)), t_data, t_data**2, t_data**3, t_data**4, t_data**5]).T
a, b, c, d, e, f = np.linalg.lstsq(A, n_original)[0]
plt.plot(t, a+b*t+c*t**2+d*t**3+e*t**4+f*t**5, 'r', label='Fitted line')
plt.legend(loc=4)
plt.show()

plt.figure()
plt.plot(t_data, n_data, 'o', label='Log-Original data', markersize=4)
A = np.vstack([np.ones(len(t_data)), t_data, t_data**2, t_data**3, t_data**4, t_data**5]).T
a, b, c, d, e, f = np.linalg.lstsq(A, n_data)[0]
plt.plot(t, a+b*t+c*t**2+d*t**3+e*t**4+f*t**5, 'r', label='Fitted line')
plt.legend(loc=4)
plt.show()


plt.figure()
N=np.size(data_set)
Time=data_set[0:N:1,0]
Bacteria_Concentration=data_set[0:N:1,1]
log_Bacteria_Concentration=np.log(Bacteria_Concentration)

plt.scatter(Time,log_Bacteria_Concentration)
plt.xlabel('Time[Hours]')
plt.ylabel('Log Bacteria[*10^3 CFU/ml]')
plt.title('Log Bacteria Concentration Over Time')



t_data = np.transpose(data_set)[0]
n_data = np.log(np.transpose(data_set)[1])
n_original = np.transpose(data_set)[1]
t = np.linspace(0, tf, tf +1).flatten()

B = t_data[36:44]
B = np.vstack([np.ones(B.size), B]).T
y_intercept, slope = np.linalg.lstsq(B, n_data[36:44])[0]
mu_guess= - slope

first_range=23
A=Time[0:first_range]
A = A.reshape((A.size, 1))
y = log_Bacteria_Concentration[0:first_range]
y = y.reshape(y.size, 1)
a = float(np.linalg.lstsq(A, y)[0])
v_guess = a

# Monod law = bacteria growth rate
# n = bacteria concentration
# gmax = maximum bacteria growth rate
# p = glucose concentration
# k = Monod constant = p*10^-100
# a = CFU/mg*ml
   
# mu = bacteria death rate
# bacteria concentration
# glucose concentration

def diff_eq(values, t, gmax, mu, k, a):
    n = values[0]
    p = values[1]
    for t in Time:
        if t < 4.5:
            dn_dt = 0
            dp_dt = 0
        else:
            dn_dt = n * gmax * p / (p+k) - mu*n
            dp_dt = n * gmax * p / (p+k) * (1/a)
    return [dp_dt, dn_dt]

#plt.figure()
init = [0.2, 50]
"""
curve fit needed to find parameters - google function
plug odeint into curve fit
graph odeint with parameter values from curve fit for bacteria vs. time graph
"""
state = odeint(diff_eq, init, Time, args = (1, -0.24, 0.002, 0.00002))
X=np.size(state)
Odient_Bacteria=state[0:X:1,1]
#plt.scatter(Time, Odient_Bacteria)

#curve_fit(diff_eq, Time, Bacteria_Concentration, )




