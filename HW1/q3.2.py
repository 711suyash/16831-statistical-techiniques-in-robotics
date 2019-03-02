#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import random

def ExpertOne(t):
    return 1

def ExpertTwo(t):
    return -1

def ExpertThree(t):
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

T = 100
w=[1,1,1]    

n=0.5 
for t in range (T):
    h=[ExpertOne(i), ExpertTwo(i),  ExpertThree(i)]
    print('expert one prediction', ExpertOne(i), '\n')
    print ('expert two prediction', ExpertTwo(i), '\n')
    print('expert three prediction', ExpertThree(i), '\n')
    
    print('true stochastic label ', t,'is ', stochastic(i) )
    print('true deterministic label ', t, 'is', deterministic(i))
  
   

    yt= adversarial(w,h)   
    
    
    for i in range(3):
        w[i] = w[i]*(1-(n*(h[i]!=yt)))
    
    print('true adversarial label', t, 'is', -adversarial(w,h)) #adversarial knows the learning process of the experts and has access to
                                                                # output predictions of the experts, and gives the output label completely different as that predicted.







