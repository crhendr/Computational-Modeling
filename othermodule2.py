# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:11:40 2017

@author: Crissy
"""

import numpy as np
import matplotlib.pyplot as plt
from imp import reload
import growthfunctions as gf
import integrators as integ
reload(gf)
reload(integ)

SimulationTime = 1.0
P0 = 1.0
GrowthRate = 1.0


dt = np.logspace(-1, -6, 10)
Population = np.zeros(dt.size)
t_end = np.zeros(dt.size)

for i in range(dt.size):
    t, P = integ.RK2(gf.Malthus, 0.0, P0, SimulationTime, dt[i])
    Population[i] = P[-1]
    t_end[i] = t[-1]

PopError = Population - P0*np.exp(GrowthRate*t_end)


plt.loglog(dt, np.abs(PopError))
plt.title('Error of the numerical simulation for GrowthRate=' + str(GrowthRate))
plt.xlabel('dt')
plt.ylabel('absolute error')
plt.show()

dt = 0.001
tE, PE = integ.Euler(gf.Malthus, 0.0, P0, SimulationTime, dt)
tRK2, PRK2 = integ.RK2(gf.Malthus, 0.0, P0, SimulationTime, dt)
print('Euler error = '+str(PE[-1]-P0*np.exp(GrowthRate*SimulationTime))+'.')
print('RK2 error = '+str(PRK2[-1]-P0*np.exp(GrowthRate*SimulationTime))+'.')