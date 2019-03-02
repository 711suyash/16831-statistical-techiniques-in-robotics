#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import random

def E1(t):
    return 1

def E2(t):
    return -1

def E3(t):
    if (t%2==0):
        return 1
    else:
        return -1

def SmartExpert(t, weather): #smart expert with three weather conditions leading to prediction of tartan win
    if t%2 != 0 and t%5==0:
        if weather=='sunny':
            return 1
        else:
            return -1
    if t%10 + t%3 < 9:
        if weather == 'cloudy':
            return 1
        else:
            return -1
        
    if (t%2 + t%5)%2==0:
        if weather == 'rainy':
            return -1
        else:
            return 1
    
    else:
        return -1
        
    
    
def stochastic(t):
    x= [-1,1]
    y= random.choice(x)
    return y

def deterministic(t):
    if t%2==0:
        return 1
    else:
        return -1
    
def adversarial(w,h):
    p = w.index(max(w))
    return -h[p]


def WMA(T, x):
    w = [1,1,1,1]
    #For Loss of Learner
    loss_algorithm = 0
    sum_loss = 0
    cum_loss = [0]*T

    #For Loss of Experts; cumulative loss is stored as list of lists
    loss_expert = [0]*4
    sum_loss_expert = [0]*4
    cum_loss_expert = [[0 for i in range (4)] for j in range(T)]
    n = 0.1

    #WEIGHTED MAJORITY ALGORITHM
    for t in range(T):
        weather = random.choice(['sunny','cloudy','rainy'])
        h = [E1(t),E2(t),E3(t), SmartExpert(t,weather)]
        y_pred = np.sign(np.dot(h,w))
        if x== adversarial:
            y_t = x(w,h)
        else:
            y_t= x(t)
       
        for i in range(4):
            w[i] = w[i]*(1-(n*(h[i]!=y_t)))

        #CALCULATE THE LEARNER'S CUMULATIVE LOSS
        loss_algorithm = 1 if y_pred!=y_t else 0
        sum_loss = sum_loss + loss_algorithm
        cum_loss[t] = sum_loss

        #CALCULATE EACH EXPERT'S CUMULATIVE LOSS
        for i in range(4):
            loss_expert[i] = 1 if h[i]!=y_t else 0

            sum_loss_expert[i] = sum_loss_expert[i] + loss_expert[i]
            cum_loss_expert[t][i] = sum_loss_expert[i]

    plot_loss(cum_loss, cum_loss_expert)
    plot_regret(cum_loss, cum_loss_expert,T)
    plt.show()

def plot_loss(cum_loss, cum_loss_expert):
    plt.figure(1)
   
    Expert1, = plt.plot([row[0] for row in cum_loss_expert],'r',label='Expert1')
    Expert2, = plt.plot([row[1] for row in cum_loss_expert],'b',label = 'Expert2')
    Expert3, = plt.plot([row[2] for row in cum_loss_expert],'k',label = 'Expert3')
    Expert4, = plt.plot([row[3] for row in cum_loss_expert],'c',label = 'Expert4')
   
    Learner, = plt.plot(cum_loss,'g',label = 'Learner')
    plt.legend([Expert1, Expert2, Expert3, Expert4, Learner],['Expert1','Expert2','Expert3','Expert4','Learner'])
    plt.xlabel('t')
    plt.ylabel('Cumulative Loss')
    plt.title('average cumulative loss')
 

def plot_regret(cum_loss,cum_loss_expert,T):
    
    best_loss_expert = [0]*T
    regret = [0]*T
    
    avg_regret = [0]*T
    
    for t in range(T):
        best_loss_expert[t] = min(cum_loss_expert[t])
        regret[t] = cum_loss[t] - best_loss_expert[t]
        
        avg_regret[t] = (100*regret[t])/(t+1)

   
    plt.figure(2)
    plt.plot(avg_regret,'r')
    plt.xlabel('t')
    plt.ylabel('Average Regret')
    plt.title('regret')
 


T = 100
print('stochastic \n')
WMA(T, stochastic)

print('deterministic \n')
WMA(T, deterministic)

print('adversarial \n')
WMA(T, adversarial)







