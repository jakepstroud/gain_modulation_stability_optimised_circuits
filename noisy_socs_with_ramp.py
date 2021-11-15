# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:18:30 2021

@author: Jake
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['ggplot','paper_plots'])

np.random.seed(0) #Set random seed for reproduciblity
n_neurons = 50
initial_cond = np.random.normal(0,1,n_neurons)
W = np.random.normal(0,1/np.sqrt(n_neurons),(n_neurons,n_neurons))
noise_std = 0.05
tau = 200
dt = 1 #time step update size
    
ramp = False #With or without a ramping input
if ramp:
    t_tot = 2000 #ms
    T = int(t_tot/dt)
    T_half = int(T/2)
    h = np.linspace(-T_half,T_half,T) #-1000 to +1000 in steps of 1
    h[:T_half] = np.exp(h[:T_half]/400) #ramp input function
    h[T_half:] = np.exp(-h[T_half:]/2) #ramp input function
else:    
    t_tot = 1000 #ms
    T = int(t_tot/dt)
    h = np.zeros(T)

x = np.zeros((n_neurons,T)) #initialise neural activities to before simulation
if ramp == False:
    x[:,0] = initial_cond #set IC to 0 if not using ramp input
    
for t in range(T-1): #loop over time
    rand = np.random.normal(0,1,n_neurons) #generate random noise each time step
    x[:,t+1] = x[:,t] + (dt/tau)*(
        -x[:,t] + W@x[:,t] + h[t]*(initial_cond-W@initial_cond)) + noise_std*np.sqrt(dt/tau)*rand
    
#Plot
plt.plot(x[:10,:].T)
plt.xlabel('time (ms)')
plt.ylabel('neural activity (Hz)')