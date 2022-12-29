# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:26:38 2022

@author: Martina
"""

"""
AIM:
Figure 25: Activities produced by Jansen's model for typical values of the 
parameter p.

""" 

import numpy as np
import matplotlib.pyplot as plt
#%%

def sigm(v,e0,r,v0):
    Sigm = (2*e0)/(1+np.exp(r*(v0-v)))
    return Sigm

#%%

#We solve the differential equations using an integration method based on the
#Runge Kutta 4
p_list=[]

#We define our general model
def jansenmodel(y,param,C,p_value):
 
#We define all parameters
    A = param['A']
    B = param['B']
    v0 = param['v0']
    a = param['a']
    b = param['b']
    r = param['r']
    e0 = param['e0']
    
    C1=C
    C2=0.8*C
    C3=0.25*C
    C4=0.25*C


#We define the 6 derivatives of the model.
#We have an array of 6 rows, one for each equation with a certain numbert of 
#columns that correspond to the number of steps we want to make. 
    
#Random noise
    p=p_value 
    
    ydot = np.array([y[3],
                     y[4],
                     y[5],
                     A*a*sigm(y[1]-y[2], e0, r, v0)-2*a*y[3]-a**2*y[0],
                     A*a*(p+C2*sigm(C1*y[0], e0, r, v0))-2*a*y[4]-a**2*y[1],
                     B*b*(C4*sigm(C3*y[0], e0, r, v0))-2*b*y[5]-b**2*y[2]])

#Add gaussian noise to mimic the signal
    return ydot

#%%

#We do the RK4
def myrk4(fun, y0, t0, tf, dt):
    t=np.arange(t0,tf,dt)
    
#We create an array with all the information
    t_size=t.size
    y0_size=y0.size
    
    y=np.zeros((y0_size, t_size))
    
#All raws of the first column
    y[:,0] = y0
    
    for k in range(0,t_size-1):
        
        k1 = dt*fun(t[k], y[:,k])
        k2 = dt*fun(t[k] + dt/2, y[:,k] + k1/2)
        k3 = dt*fun(t[k] + dt/2, y[:,k] + k2/2)
        k4 = dt*fun(t[k] +dt, y[:,k] + k3)
        
        dy = (k1 + 2*k2 + 2*k3 +k4)/6
        
        y[:,k+1] = y[:,k] + dy
        
    return y,t

#%%

#We characterise the rk4

param = {"A":3.25, "B":22, "v0":6, "a":100, "b":50, "r":0.56, "e0":2.5}

t0 = 0
tf = 10
dt = 0.0001

t=np.arange(t0,tf,dt)

y0 = np.array([0,0,0,0,0,0])
y0_e = np.array([0.1,18.0,12.0,0,0,0])

#We make a list in order to put all solutions for the different cnsts
C=135
p_values=[50,100,125,200]
sol=[]
sol_e=[]

for i in range(0,len(p_values)):
    print(i)
    fun = lambda t, y : jansenmodel(y,param, C, p_values[i])
    y, t=myrk4(fun, y0, t0, tf, dt)
    act = y[1]-y[2]
    y_e, t=myrk4(fun, y0_e, t0, tf, dt)
    act_e = y_e[1]-y_e[2]
    sol.append(act)
    sol_e.append(act_e)
    
#%%

plt.figure()
plt.suptitle('Figure 25. Grimbert')
for i in range(0,len(p_values)):
    plt.subplot(2,2,i+1)
    plt.title("p= " +str(p_values[i]), fontsize=10)
    plt.plot(t,sol[i], color='red')
    plt.plot(t,sol_e[i], color='green')
    plt.xlim(8,10)
    plt.ylim(-2,12)
    plt.xlabel('t (s)', fontsize=10)
    
plt.show()