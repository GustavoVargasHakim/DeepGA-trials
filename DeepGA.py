# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:43:55 2020

@author: user
"""

from Operators import *
from EncodingClass import *
from Decoding import *
from DataReader import *
from Training import *
from torchvision import utils
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torchvision import transforms
import math


'''Loading data'''
train_dl, test_dl = loading_data()

'''Defining CNN hyperparameters'''
#Defining loss function
loss_func = nn.NLLLoss(reduction = "sum")

#Defining optimizer
num_epochs = 10

#Defining learning rate
lr = 1e-4

#Maximun and minimum numbers of layers to initialize networks
min_conv = 2
max_conv = 4
min_full = 1
max_full = 4

'''Genetic Algorithm Parameters'''
cr = 0.7 #Crossover rate
mr = 0.3 #Mutation rate
N = 20 #Population size
T = 50 #Number of generations
t_size = 5 #tournament size
w = 0.1 #penalization weight
max_params = 1e6

'''Evaluating the objective function of an encoding (accuracy + w*No. Params)'''
def evaluate_individual(x):
    #Decoding the network
    network = decoding(x)
    
    #Creating the CNN (and obtaining number of parameters)
    cnn = CNN(x, network[0], network[1], network[2])
    parms = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    
    #Passing the CNN to a GPU (if only one GPU is available)
    cnn.to(decive, dtype = torch.float32)
    
    #Defining optimizer
    opt = optim.Adam(cnn.parameters(), lr = lr)
    
    #Training the network
    accuracy, _ = train_val(num_epochs, cnn, loss_func, opt, train_dl, test_dl)
    
    #Fitness function
    f = accuracy + w*(parms - max_params)/max_params
    
    return f, accuracy
    
'''Initialize population'''
pop = []
for n in range(N):
    #Creating a genome (genetic encoding)
    e = Encoding(min_conv,max_conv,min_full,max_full) 
    
    #Evaluate individual
    f, accuracy = evaluate_individual(e)
    
    pop.append([e, f, accuracy])

'''Genetic Algorithm'''
for t in range(T):
    print('Generation: ', t)
    
    offspring = []
    while len(offspring) < N:
        #Parents selection through tournament 
        tournament = random.sample(pop, t_size)
        p1 = selection(tournament, 'max')
        tournament = random.rample(pop, t_size)
        p2 = selection(tournament, 'max')
        while p1 == p2:
            tournament = random.rample(pop, t_size)
            p2 = selection(tournament, 'max')
        
        #Crossover + Mutation
        if random.uniform(0,1) >= cr: #Crossover
            c1, c2 = crossover(p1, p2)
            
            #Mutation
            if random.uniform(0,1) >= mr:
                mutation(c1)
            
            if random.uniform(0,1) >= mr:
                mutation(c2)
            
            f1, acc1 = evaluate_individual(c1)
            f2, acc2 = evaluate_individual(c2)
            
            offspring.append([c1, f1, acc1])
            offspring.append([c2, f2, acc2])
        
        #Sorting offspring based on fitness
        offspring.sort(key = lambda x: x[1])
        leader = max(pop, key = lambda x: x[1])
        
        print('Best fitness: ', leader[1])
        print('Best accuracy: ', leader[2])
        
        #Replacement with elitism
        pop = [leader] + offspring[:-1]
        
    
    
    

