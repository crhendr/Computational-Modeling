# -*- coding: utf-8 -*-
"""
Created on Mon May  1 23:00:34 2017

@author: Crissy
"""

#Module1: Cyanobacteria growth under sunlight
import numpy as np
import matplotlib.pyplot as plt
# default parameters: final
large, small = 12, 10  #font sizes
simulationTime = 30.0 #days
dt = 0.2/24 # day
pinitial = 1.0 #*10**3 cells, the initinal population
pmax = 10000.0 #*10**3 cells, the carrying capacity
rmax = .99 #day^-1, the initial growth rate
#%%
def populationDisCont_rate(dt, pinitial, pmax, rmax):
#returns the values of time, and the corresponding exact and numerical
#solution, and the growth rate

    def growthRate(n,t):#defines the growth rate
        return (rmax - k * (n-pinitial)) * np.sin(np.pi * t)**2

    k = rmax / (pmax - pinitial) #(day*cells)^-1
    
    time = np.arange(0, simulationTime+dt, dt)
    #creates an array for the values of time when data is collected
    population = np.zeros(time.size)
    #creates an array to be filled with the population at a certain time
    population[0] = pinitial
    
    populationCont = (rmax+k*pinitial) / (k + rmax/pinitial* \
    np.exp((rmax+k*pinitial)/2* (np.sin(2*np.pi*time)/(2*np.pi) - time)))
    #gives the exact solution
    rate = growthRate(populationCont, time) # via *exact solution* and time
    for i in np.arange(1, time.size):
        population[i] = population[i-1] + dt * growthRate(population[i-1], \
        time[i-1]) * population[i-1]
        #creates an array filled with the values of the numerical solution
    
    return [time, populationCont,population, rate]
#%% default plot
time, populationCont,population, rate = populationDisCont_rate(dt, pinitial, \
                                                               pmax, rmax)

plt.subplot(2,1,1)
plt.plot(time, rate)
plt.title('(a): Growth rate of cyanobacteria population in 30 days', \
          fontsize = large)
plt.xlim(0,simulationTime)
plt.ylim(0,1.05*np.max(rate))
plt.xlabel("Time [Days]", size = large)
plt.ylabel("Growth rate [Day^(-1)]", size = large)
plt.xticks(np.linspace(0,simulationTime,6), fontsize = small)
plt.yticks(np.linspace(0, np.max(rate), 6), fontsize = small)

plt.subplot(2,1,2)
plt.plot(time, population, 'bo', label='Numerical solution')
plt.plot(time, populationCont,'r-', linewidth = 3, label='Exact solution')
plt.legend(fontsize = large, loc = 2)
plt.title('(b): Cyanobacteria population growth in a span of 30 days', \
          fontsize = large)
plt.xlim(0,simulationTime)
plt.ylim(0,1.05*pmax)
plt.xlabel("Time [Days]", size = large)
plt.ylabel("Population [*10^3 Cells]", size = large)
plt.xticks(np.linspace(0,simulationTime,6), fontsize = small)
plt.yticks(np.linspace(0, pmax, 6), fontsize = small)
#%% More plots to verify the solution
#rmax --> 0.002
time, populationCont1,population1, bla = populationDisCont_rate(dt, pinitial, pmax, 0.002)

plt.subplot(2,2,1)
plt.plot(time, population, 'bo', label='Numerical solution')
plt.plot(time, populationCont,'r-', linewidth = 3, label='Exact solution')
plt.legend(fontsize = large, loc = 2)
plt.title('(a): Population growth with r_0 = 0.02 day^(-1)', fontsize = large)
plt.xlim(0,simulationTime)
plt.ylim(0,1.05*pmax)
plt.xlabel("Time [Days]", size = large)
plt.ylabel("Population [*10^3 Cells]", size = large)
plt.xticks(np.linspace(0,simulationTime,6), fontsize = small)
plt.yticks(np.linspace(0, pmax, 6), fontsize = small)

#rmax --> 200
time, populationCont2,population2, bla = populationDisCont_rate(dt, pinitial, pmax, 200)

plt.subplot(2,2,2)
plt.plot(time, population2, 'bo', label='Numerical solution')
plt.plot(time, populationCont2,'r-', linewidth = 3, label='Exact solution')
plt.legend(fontsize = large, loc = 4)
plt.title('(b): Population growth with r_0 = 200 day^(-1)', fontsize = large)
plt.xlim(0,simulationTime)
plt.ylim(0,1.05*pmax)
plt.xlabel("Time [Days]", size = large)
plt.ylabel("Population [*10^3 Cells]", size = large)
plt.xticks(np.linspace(0,simulationTime,6), fontsize = small)
plt.yticks(np.linspace(0, pmax, 6), fontsize = small)

#pinitial --> 0
time, populationCont2,population2, bla = populationDisCont_rate(dt, 0.0000001, pmax, rmax)

plt.subplot(2,2,3)
plt.plot(time, population2, 'bo', label='Numerical solution')
plt.plot(time, populationCont2,'r-', linewidth = 3, label='Exact solution')
plt.legend(fontsize = large, loc = 2)
plt.title('(c): Population growth with n_0 = 1*10^(-4) bacterium', fontsize = large)
plt.xlim(0,simulationTime)
plt.ylim(0,1.05*pmax)
plt.xlabel("Time [Days]", size = large)
plt.ylabel("Population [*10^3 Cells]", size = large)
plt.xticks(np.linspace(0,simulationTime,6), fontsize = small)
plt.yticks(np.linspace(0, pmax, 6), fontsize = small)

#pinitial --> pmax
time, populationCont2,population2, bla = populationDisCont_rate(dt, 0.999*pmax, pmax, rmax)

plt.subplot(2,2,4)
plt.plot(time, population2, 'bo', label='Numerical solution')
plt.plot(time, populationCont2,'r-', linewidth = 3, label='Exact solution')
plt.legend(fontsize = large, loc = 4)
plt.title('(d): Population growth with n_0 = 99.9% n_max', fontsize = large)
plt.xlim(0,simulationTime)
plt.ylim(0,1.05*pmax)
plt.xlabel("Time [Days]", size = large)
plt.ylabel("Population [*10^3 Cells]", size = large)
plt.xticks(np.linspace(0,simulationTime,6), fontsize = small)
plt.yticks(np.linspace(0, pmax, 6), fontsize = small)
