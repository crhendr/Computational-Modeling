# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:46:53 2017

@author: Crissy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

bounds = 20

original_point = (bounds / 2, bounds / 2)

numpoints= 100 #number of total points in cluster

latt_points_x = np.zeros(numpoints + 1)
latt_points_y = np.zeros(numpoints + 1)


time_coords = np.zeros(numpoints + 1)
final_coords = np.zeros(numpoints + 1)
moves=np.zeros(numpoints+1)

radius=np.zeros(numpoints+1)
latt_points_x[0] = original_point[0]
latt_points_y[0] = original_point[1]

#possible neighbor points
nx = [-1, 0, 1, 0]
ny = [0, 1, 0, -1]

#random walk that tells if the particle moves up, down, left, or right and produces an output of the new particle location
# also checks to see if boundry and reflects if it will pass boundry

def random_walk(point_coords, bounds):
    x = point_coords[0]
    y = point_coords[1]
    
    rand_step_x = (2 * (np.random.rand() < .5)) - 1
    rand_step_y = (2 * (np.random.rand() < .5)) - 1

    if x + rand_step_x > bounds or x + rand_step_x < 0: #reflects if points pass boundries
        rand_step_x = rand_step_x * -1
    if y + rand_step_y > bounds or y + rand_step_y < 0:
        rand_step_y = rand_step_y * -1
    
    x = x + rand_step_x
    y = y + rand_step_y
    
    return (x, y)

#checks to see if point attaches and returns a point if it is supposed to be attached
#returns coordinates and whether it is near cluster

def point_check(point_coords):
    pt_attach = False
    
    for i in range(len(nx)):
        x = point_coords[0] + nx[i]
        y = point_coords[1] + ny[i]

        for n in range(len(latt_points_x)):
            if x == latt_points_x[n] and y == latt_points_y[n] and x != 0 and y != 0:
                pt_attach = True
                break
        
        if pt_attach == True:
            break
        
    return pt_attach, point_coords
    
#print (point_check((49,58)))

#produces random coordinate of point on the borders of lattice

def start_point_gen(lattice_bounds):
    
    initial_pt = np.random.rand()
    
    if initial_pt < .25:
        initial_x = np.random.randint(0,lattice_bounds)
        initial_y = 0
    elif initial_pt < .5 and initial_pt >= .25:
        initial_y = np.random.randint(0,lattice_bounds)
        initial_x = 0
    elif initial_pt < .75 and initial_pt >= .5:
        initial_x = np.random.randint(0,lattice_bounds)
        initial_y = lattice_bounds
    elif initial_pt >= .75:
        initial_y = np.random.randint(0,lattice_bounds)
        initial_x = lattice_bounds
    
    start_point = (initial_x, initial_y)
    
    return start_point
    
#%%
# for loop = for individual point movement
# whil loop = for number of total cluster points

def dla(total_cluster_pts, lattice_bounds):
    start_time = time.time()
    
    cluster_pts = 0
    pt_counter = 1 # used to add elements to lattice array
    
    while cluster_pts < total_cluster_pts:
        new_pt = start_point_gen(lattice_bounds)
        xint=new_pt[0]
        yint=new_pt[1]
        for num_moves in range(1500): #number of times random particle moves before being terminated = assumption
            new_pt = random_walk(new_pt, lattice_bounds)
            flag, coords = point_check(new_pt)

            if flag == True:
                latt_points_x[pt_counter] = coords[0]
                latt_points_y[pt_counter] = coords[1]
			
                radius[pt_counter]=np.abs((latt_points_x[pt_counter]-xint))**2 + (np.abs(latt_points_y[pt_counter]-yint)**2)	
                
                #time_coords[pt_counter] = np.abs((np.sort(latt_points_x)[-1]) - original_point[0])
                
                
                time_coords[pt_counter] = np.abs((coords[0] - original_point[0]))
                
                final_coords[pt_counter] = max(time_coords)
            
            
                moves[pt_counter]=num_moves
                pt_counter = pt_counter + 1
                cluster_pts = cluster_pts + 1
                plt.scatter(latt_points_x, latt_points_y)
                plt.xlim(1,bounds)
                plt.ylim(1,bounds)
                plt.title('DLA')
                
                print(str(num_moves)+'==number of moves    ', str((cluster_pts/numpoints)*100)+'% complete')
                break
            
            if cluster_pts == total_cluster_pts:
            	elapsed_time = time.time() - start_time
            	return elapsed_time
        
        
            
dla(numpoints, bounds)   

plt.scatter(latt_points_x, latt_points_y)
plt.figure()
plt.plot(np.arange(0,numpoints+1,1),time_coords)

#Time lapse
t1=numpoints/4
t2=numpoints/2
t3=numpoints*(3/4)
t4=numpoints
plt.scatter(latt_points_x[0:t1], latt_points_y[0:t1])
plt.scatter(latt_points_x[0:t2], latt_points_y[0:t2])
plt.scatter(latt_points_x[0:t3], latt_points_y[0:t3])
plt.scatter(latt_points_x[0:t4], latt_points_y[0:t4])

#movie
from matplotlib import animation
fig=plt.figure()
ax=plt.axes(xlim=(0, bounds), ylim=(0, bounds))

scat = ax.scatter([], [], s=60)

def init():
    scat.set_offsets([])
    return scat,

def animate(i):
    data = np.hstack((latt_points_x[:i,np.newaxis], latt_points_y[:i, np.newaxis]))
    scat.set_offsets(data)
    return scat,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(latt_points_x)+1, 
                               interval=200, blit=False, repeat=False)
            
  
plt.figure(2)
plt.scatter(np.arange(0,numpoints,1),radius[1:numpoints+1])
plt.title('Distance to Attach')
plt.xlabel('Time Point')
plt.ylabel('Distance')    
plt.figure(3)
plt.plot(np.arange(0,numpoints+1,1),time_coords)          
plt.title('Max Horizontal Dinstance') 
plt.xlabel('Time Point')
plt.ylabel('Max Distance')