# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:06:47 2022

@author: Martina
"""

"""
AIM:
Figure 12: Activities of a column as simulated by Jansed and Rit with a
uniform noise (ranging between 120 and 320 Hz) as input.
""" 

import numpy as np
import matplotlib.pyplot as plt
import random
#%%
#p(t): arbitrary average firing rate than can be a noise

#Three families-pyramidal neurons, excitatory and inhibitory inter-neurons and
#synaptic interactions between them are modeled by different boxes:
    #Post-synaptic: he(t) or hi(t) converts the average rate of action 
    #               potencials constituting the input of a pop. into an 
    #               average post-synaptic potential. It converts a firing freq.
    #               into an electric potential by a 2n linear ODE

    #Sigmoid: Transforms the average membrane potential of a neural population
    #         into an average firing rate. Represents the average cell body 
    #         action of a population by converting the membrane potential into
    #         a firing rate.

#Variables of the system: 
    #Outputs of the post-synaptic boxes: y0, y1, y2
    #Their derivatives: y3, y4, y5
    
#%%
#CONSTANTS

A=3.25
B=22
v0=6
a=100
b=50
r=0.56
e0=2.5

#C=[68,128,135,270,675]
C = 68

C1=C
C2=0.8*C
C3=0.25*C
C4=0.25*C

param = {"A":3.25, "B":22, "v0":6, "a":100, "b":50, "r":0.56, "e0":2.5}

#%%
#POST SYNAPTIC
    
#Excitatory case:
def he(t, A, a):
    
    he = A*a*t*np.exp(-a*t)
    
    return he
    
#Inhibitory case:
def hi(t, B, b):
    hi = B*b*t*np.exp(-b*t)
    
    return hi

#%%
#FIGURE 10
t=np.linspace(0,0.14,100)
A = param['A']
a = param['a']
B = param['B']
b = param['b']

plt.figure()
he = he(t, A, a)
plt.plot(t,he,label='PPSE')
hi = hi(t, B, b)
plt.plot(t,hi,label='PPSI')
plt.xlabel('t')
plt.legend()

#%%
#SIGMOID

def sigm(v,e0,r,v0):
    Sigm = (2*e0)/(1+np.exp(r*(v0-v)))
    return Sigm


#%%
#FIGURE 11
v=np.linspace(-10,20,100)
v0 = param['v0']
a = param['a']
B = param['B']
b = param['b']

plt.figure()
Sigm=sigm(v, e0, r, v0)
plt.plot(v,Sigm)

#%%
#OUTPUTS
#We solve the differential equations using an integration method based on the
#Runge Kutta 4
p_list=[]

#We define our general model
def jansenmodel(y,param,C):
 
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
    mean=random.uniform(120,320)
    std = 1 
    num_samples = 1
    noise = np.random.normal(mean, std, size=num_samples)
    p=noise[0]
    
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
tf = 4
dt = 0.0001

t=np.arange(t0,tf,dt)

y0 = np.array([0,0,0,0,0,0])

#We make a list in order to put all solutions for the different cnsts
cnst=[68,128,135, 270, 675, 1350]

sol_cnst=[]

for i in range(0,len(cnst)):
    print(i)
    fun = lambda t, y : jansenmodel(y,param, cnst[i])
    y, t=myrk4(fun, y0, t0, tf, dt)
    act = y[1]-y[2]
    sol_cnst.append(act)

    
#%%

plt.figure()
plt.suptitle('Figure 12. Grimbert')

for i in range(0,len(cnst)):
    plt.subplot(6,1,i+1)
    plt.plot(t,sol_cnst[i],label = str(cnst[i]))
    plt.axhline(y=0, xmin=2, xmax=4)
    plt.legend(loc = "upper right")
    plt.xlim(2,4)
    
    if i==0:
        plt.ylim(9.5,11)
    
    if i==1:
        plt.ylim(6.5,9.5)
        
    if i==2:
        plt.ylim(5,12)
    
    if i==3:
        plt.ylim(-30,30)
        
    if i==4:
        plt.ylim(-150,30)
    
    if i==5:
        plt.ylim(-12.5,-11.5)
        
    if i!=5:
        plt.xticks([])

plt.xlabel('t (s)', fontsize=10)
    
plt.show()





