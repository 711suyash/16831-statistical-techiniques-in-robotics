#!/usr/bin/env python
# coding: utf-8

# In[21]:


import math
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


def RWMA(T, x):
    w = [1,1,1]
    #For Loss of Learner
    loss_algorithm = 0
    sum_loss = 0
    cum_loss = [0]*T

    #For Loss of Experts; cumulative loss is stored as list of lists
    loss_expert = [0]*3
    sum_loss_expert = [0]*3
    cum_loss_expert = [[0 for i in range (3)] for j in range(T)]
    n = 0.1

    #WEIGHTED MAJORITY ALGORITHM
    for t in range(T):
        
        h = [E1(t),E2(t),E3(t)]
        y_pred = randomize(w,h)
        if x== adversarial:
            y_t = x(w,h)
        else:
            y_t= x(t)
       
        for i in range(3):
            w[i] = w[i]*(1-(n*(h[i]!=y_t)))

        #CALCULATE THE LEARNER'S CUMULATIVE LOSS
        if y_pred!=y_t:
            loss_algorithm = 1
        else:
            loss_algorithm = 0
        
        sum_loss = sum_loss + loss_algorithm
        cum_loss[t] = sum_loss

        #CALCULATE EACH EXPERT'S CUMULATIVE LOSS
        for i in range(3):
            loss_expert[i] = 1 if h[i]!=y_t else 0

            sum_loss_expert[i] = sum_loss_expert[i] + loss_expert[i]
            cum_loss_expert[t][i] = sum_loss_expert[i]

    plot_loss(cum_loss, cum_loss_expert)
    plot_regret(cum_loss, cum_loss_expert,T)
    plt.show()
    
def randomize(w,h):
    sum_weights = 0
    p_weights = [0]*3
    sum_p = 0

    sum_weights = math.fsum(w)
    for i in range(3):
        p_weights[i] = 1.0*w[i]/sum_weights
        sum_p = sum_p + p_weights[i]
        p_weights[i] = sum_p

    y = random.random()
    for i in range(3):
        if y<p_weights[i]:
            return i

def plot_loss(cum_loss, cum_loss_expert):
    plt.figure(1)
   
    Expert1, = plt.plot([row[0] for row in cum_loss_expert],'r',label='Expert1')
    Expert2, = plt.plot([row[1] for row in cum_loss_expert],'b',label = 'Expert2')
    Expert3, = plt.plot([row[2] for row in cum_loss_expert],'k',label = 'Expert3')
   
    Learner, = plt.plot(cum_loss,'g',label = 'Learner')
    plt.legend([Expert1, Expert2, Expert3, Learner],['Expert1','Expert2','Expert3','Learner'])
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
RWMA(T, stochastic)

print('deterministic \n')
RWMA(T, deterministic)

print('adversarial \n')
RWMA(T, adversarial)


#with respect to WMA, there is considerable decline in average regret for adversarial in RWMA, though not significant in deterministic and stochastic







