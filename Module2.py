#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:08:28 2017

@author: Erik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

x = 100
y = 1000
k2 = 0.001
k1 = 0.0001
d1 = 0
d2 = 0
r = 0
y0 = 10000
T= 0
dt = 1
Time = []
X = []
Y = []
while T<=200:
    dx=((k2*x*y)-(d2*x))*dt
    dy=(((r*y)*(1-(y/y0))) - (k1*x*y)-(d1*y))*dt
    x=x+dx
    y=y+dy
    Time.append(T)
    X.append(x)
    Y.append(y)
    print('Time = ' + str(T) + '; X = ' + str(x) + '; Y = ' + str(y))
    T=T+dt
    

plt.plot(Time, X, label= 'Predator')
plt.plot(Time, Y, label = 'Prey')
plt.title('Lotka-Volterra predator-prey system')
plt.xlabel("Time [Days]", size = 10)
plt.ylabel("Population", size = 10)
plt.legend()
plt.xticks()
plt.yticks()

def G(x,dt,k2,d2,y):
    
    return ((k2*x*y)-(d2*x))*dt
def F(y,x,y0,r,d1,k1, dt):
    return (((r*y)*(1-(y/y0))) - (k1*x*y)-(d1*y))*dt
PreyPopulation= odeint(F,y0,Time)