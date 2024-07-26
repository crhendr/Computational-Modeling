# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:05:57 2017

@author: Crissy
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#in the simulation prey is grass *10000 females, 
#and predator is rabbits*10000 females
k2 = 0.01#the rate the predators interact with the prey 
#in (day^-1)*(*10000 females)^-1
k1 = 0.05#the rate the prey interact with the predators 
#in (day^-1)*(*10000 females)^-1
d1 = 0.1#death rate of the prey in day^-1
d2 = 0.2#death rate of the predator in day^-1
r = 0.7#growth rate of the prey in day^-1
y0 = 10000#carrying capacity of the prey in *10000 females
dt=.1#step size, days
xi=20#initial predator population, *10000 females
yi=20#initial prey population, *10000 females
Ti=0#model start time in days
Tf=100#model end time in days
#%%
#Odeint
#solves the system of differential equations numerically using Odeint
#returns the time and the populations
def Odeint(k1, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt):
    #defines a function Odeint
    t=np.arange(Ti,Tf +dt, dt)#creates an array of the values of time to use
    initial_conditions = [xi,yi]#defines the initinal conditions
    def dX_dt(X_odeint, t=0):#creates an array of the populations
        return np.array([(k2*X_odeint[0]*X_odeint[1])-(d2*X_odeint[0]), \
                          ((r*X_odeint[1])*(1-(X_odeint[1]/y0)))-\
                   (k1*X_odeint[0]*X_odeint[1])-(d1*X_odeint[1])])
    X_odeint, infodict =odeint(dX_dt,initial_conditions,t, full_output=True)
    return [t, X_odeint]

t, X_odeint = Odeint(k1, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,2,1)
plt.plot(t,X_odeint)#plots the populations v. time 
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with Odeint', size=14)

#%%
#R-K-2
#solves the system of differential equations numerically using R-K-2
#returns the time and the populations
def RK2(k1, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt):#defines a R-K-2 function
    X_RK2 = []
    #creates an empty array to fill with the values of the predator population
    T_RK2 = np.arange(Ti,Tf+dt,dt)
    #creates an array of the values of time to use
    Y_RK2= []
    #creates an empty array to fill with the values of the prey population
    x_RK2 = xi#defines the initinal predator population
    y_RK2 = yi#defines the initinal prey population
    for t in np.arange(Ti,Tf+dt,dt):#implements R-K-2
        dx=((k2*x_RK2*y_RK2)-(d2*x_RK2))*dt
        dy=(((r*y_RK2)*(1-(y_RK2/y0))) - (k1*x_RK2*y_RK2)-(d1*y_RK2))*dt
        xguess=x_RK2+dx
        yguess=y_RK2+dy
        fxdx = ((k2*xguess*yguess)-(d2*xguess))*dt
        fydy = (((r*yguess)*(1-(yguess/y0))) - (k1*xguess*yguess)-(d1*yguess))*dt
        x_RK2= x_RK2 + 0.5*(dx+fxdx)
        y_RK2= y_RK2 + 0.5*(dy+fydy)
        X_RK2.append(x_RK2)
        Y_RK2.append(y_RK2)
    return [T_RK2, X_RK2, Y_RK2]

plt.subplot(2,2,2) 
T_RK2, X_RK2, Y_RK2 = RK2(k1, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt)   
plt.plot(T_RK2,X_RK2, label='Predator')#plots the predator population v. time
plt.plot(T_RK2,Y_RK2, label='Prey')#plots the prey population v. time
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with R-K-2', size=14)
plt.legend()

#%%
#Euler Method
#solves the system of differential equations numerically using Euler
#returns the time and the populations
def Euler(k1, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt):#defines a Euler function
    X_Euler = []
    #creates an empty array to fill with the values of the predator population
    T_Euler = np.arange(Ti,Tf+dt,dt)
    #creates an array of the values of time to use
    Y_Euler= []
    #creates an empty array to fill with the values of the prey population
    x_Euler=xi#defines the initinal predator population
    y_Euler=yi#defines the initinal prey population
    for t in np.arange(Ti,Tf+dt,dt):#implements Euler
        dx=((k2*x_Euler*y_Euler)-(d2*x_Euler))*dt
        dy=(((r*y_Euler)*(1-(y_Euler/y0))) - \
        (k1*x_Euler*y_Euler)-(d1*y_Euler))*dt
        x_Euler=x_Euler+dx
        y_Euler=y_Euler+dy
        X_Euler.append(x_Euler)
        Y_Euler.append(y_Euler)
    return[T_Euler,X_Euler,Y_Euler]
T_Euler,X_Euler,Y_Euler=Euler(k1, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,2,3)
plt.plot(T_Euler,X_Euler, label="Predator")
#plots the predator population v. time
plt.plot(T_Euler,Y_Euler, label="Prey")#plots the prey population v. time
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with Euler', size=14)
plt.legend()
#%%
#makes a phase portrait of the system
def Phase_Portrait(k1, k2, d1, d2, r, y0):
    coords=np.linspace(0, 50, 26)
    xe, ye = np.meshgrid(coords, coords)
    dx=((k2*xe*ye)-(d2*xe))
    dy=(((r*ye)*(1-(ye/y0))) - (k1*xe*ye)-(d1*ye))
    Vectordxdt, Vectordydt= dx, dy
    return [xe, ye, Vectordxdt, Vectordydt]
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, k2, d1, d2, r, y0)
plt.figure()
plt.quiver(xe, ye, Vectordxdt, Vectordydt)#plots the phase portrait
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system', size=14)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
#%%
#Phase Portrait and cooresponding population graphs 
#for various parameter values
#k2=.4, r=.07
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, .4, d1, d2, .07, y0)
plt.subplot(2,3,1)
plt.quiver(xe, ye, Vectordxdt, Vectordydt)
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system with k2=.4 & r=.07', size=10)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
t, X_odeint = Odeint(k1, .4, d1, d2, .07, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,3,4)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time{Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=.4 & r=.07', size=12)
#k2=.01, r=.7
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, .01, d1, d2, .7, y0)
plt.subplot(2,3,2)
plt.quiver(xe, ye, Vectordxdt, Vectordydt)
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system with k1=.01 & r=.7', size=10)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
t, X_odeint = Odeint(.01, k2, d1, d2, .7, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,3,5)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=.01 & r=.7', size=12)
#k2=.2, r=1.0
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, 0.2, d1, d2, 1.0, y0)
plt.subplot(2,3,3)
plt.quiver(xe, ye, Vectordxdt, Vectordydt)
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system with k2=0.2 & r=1.0', size=8)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
t, X_odeint = Odeint(k1, .2, d1, d2, 1.0, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,3,6)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=0.2 & r=1.0', size=10)
#k2=.6, r=1.5
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, 0.6, d1, d2, 1.5, y0)
plt.figure()
plt.subplot(2,3,1)
plt.quiver(xe, ye, Vectordxdt, Vectordydt)
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system with k2=0.6 & r=1.5', size=8)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
t, X_odeint = Odeint(k1, 0.6, d1, d2, 1.5, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,3,4)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=0.6 & r=1.5', size=10)
#k2=.1, r=2.0
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, .1, d1, d2, 2.0, y0)
plt.subplot(2,3,2)
plt.quiver(xe, ye, Vectordxdt, Vectordydt)
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system with k2=.1 & r=2.0', size=8)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
t, X_odeint = Odeint(k1, .1, d1, d2, 2.0, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,3,5)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=.1 & r=2.0', size=10)
#k2=.5,r=2.5
xe, ye, Vectordxdt, Vectordydt = Phase_Portrait(k1, .5, d1, d2, 2.5, y0)
plt.subplot(2,3,3)
plt.quiver(xe, ye, Vectordxdt, Vectordydt)
plt.title('Phase Portrait for the Lotka-Volterra predator-prey system with k2=.5 & r=2.5', size=8)
plt.xlabel('Predator[x10000 Females]', size=12)
plt.ylabel('Prey[x10000 Females]', size=12)
plt.streamplot(xe, ye, Vectordxdt, Vectordydt)
plt.xlim(0,50)
plt.ylim(0,50)
t, X_odeint = Odeint(k1, .5, d1, d2, 2.5, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,3,6)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=.5 & r=2.5', size=10)
#%%
#Phase Diagram for different k2 and r values
plt.figure()
plt.scatter(.07,.4, marker='x', label='no oscillations')
plt.scatter(.7,.01, marker='o', label='oscillations')
plt.scatter(1.0,.2, marker='o')
plt.scatter(1.5,0.6, marker='x')
plt.scatter(2.0,.1, marker='o')
plt.scatter(2.5,.5, marker='x')
plt.legend()
plt.xlabel('r Value[Day^-1]', size=12)
plt.ylabel('k2 Value[(day^-1)*(*10000 females)^-1]', size=12)
plt.title('Phase Diagram for Lotka-Volterra Predator-Prey System', size=14)
#%%
#Verification of the Model
plt.figure()
#r=1.4
t, X_odeint = Odeint(k1, k2, d1, d2, 2, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,4,1)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with r=1.4', size=14)
#d1=100
t, X_odeint = Odeint(k1, k2, 100, d2, r, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,4,2)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with d1=100', size=14)
#d2=100
t, X_odeint = Odeint(k1, k2, d1, 100, r, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,4,3)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with d2=100', size=14)
#xi=2000
t, X_odeint = Odeint(k1, k2, d1, d2, r, y0, 2000, yi, Ti, Tf, dt)
plt.subplot(2,4,4)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with xi=2000', size=14)
#yi=2000
t, X_odeint = Odeint(k1, k2, d1, d2, r, y0, xi, 2000, Ti, Tf, dt)
plt.subplot(2,4,5)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with yi=2000', size=14)
#k2=1.0
t, X_odeint = Odeint(k1, 1.0, d1, d2, r, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,4,6)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k2=1.0', size=14)
#k1=1.0
t, X_odeint = Odeint(1.0, k2, d1, d2, r, y0, xi, yi, Ti, Tf, dt)
plt.subplot(2,4,7)
plt.plot(t,X_odeint)
plt.legend(["Predator", "Prey"])
plt.xlabel('Time[Days]', size=12)
plt.ylabel('Population[x10000 Females]', size=12)
plt.title('Lotka-Volterra predator-prey system with k1=1.0', size=14)

