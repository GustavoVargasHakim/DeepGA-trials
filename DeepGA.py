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
from DistributedTraining import *
import numpy as np
from torch import optim
import pandas as pd
import timeit
import torch
from torch import nn
from multiprocessing import Process, Manager

#Random seed
random.seed(0)
torch.manual_seed(0)

#Loading data
train_dl, test_dl = loading_data()

'''Defining CNN hyperparameters'''
#Defining loss function
loss_func = nn.NLLLoss(reduction = "sum")

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
N = 8 #Population size
T = 1 #Number of generations
t_size = 5 #tournament size
w = 0.3 #penalization weight
max_params = 1.5e6
num_epochs = 8


#print('GPUs: ', torch.cuda.device_count())

'''Evaluating the objective function of an encoding (accuracy + w*No. Params)'''
'''def evaluate_individual(x, dev):
    #Decoding the network
    #network = decoding(x)
    
    #Creating the CNN (and obtaining number of parameters)
    #cnn = CNN(x, network[0], network[1], network[2])
    #params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    
    #Passing the CNN to a GPU 
    #cnn = nn.DataParallel(cnn) #Uncomment this if more than one GPU is available
    #cnn.to(device, dtype = torch.float32)
    
    #Defining optimizer
    opt = optim.Adam(x.parameters(), lr = lr)
    
    #Training the network
    accuracy, _ = train_val(dev, num_epochs, x, opt, loss_func, train_dl, test_dl)
    
    params = sum(p.numel() for p in x.parameters() if p.requires_grad)
    
    #Fitness function
    f = abs(accuracy - w*(1 - abs((max_params - params)/max_params)))
    
    return f, accuracy'''

#Reading GPU
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")

'''Initialize population'''
print('Initialize population')
start = timeit.default_timer()
pop = []
bestAcc = []
bestF = []
manager = Manager()
while len(pop) < N:
    acc_list = manager.list()
    
    #Creating genomes (genetic encoding)
    e1 = Encoding(min_conv,max_conv,min_full,max_full) 
    e2 = Encoding(min_conv,max_conv,min_full,max_full)
    
    #Decoding the networks
    network1 = decoding(e1)
    network2 = decoding(e2)
    
    #Creating the CNNs 
    cnn1 = CNN(e1, network1[0], network1[1], network1[2])
    cnn2 = CNN(e2, network2[0], network2[1], network2[2])
       
    #Evaluate individuals
    training1 = Process(target = training, args = ('1', device1, cnn1, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
    
    training2 = Process(target = training, args = ('2', device2, cnn2, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
    
    training1.start()
    training2.start()
    training1.join()
    training2.join()
    
    if acc_list[0][0] == '1':
        pop.append([e1, acc_list[0][1], acc_list[0][2]])
        pop.append([e2, acc_list[1][1], acc_list[1][2]])
    else:
        pop.append([e2, acc_list[0][1], acc_list[0][2]])
        pop.append([e1, acc_list[1][1], acc_list[1][2]])

stop = timeit.default_timer()
execution_time = stop-start
#print('Training time of 4 Networks: ', execution_time)
#for p in pop:
#    print('Accuracy: ', p[2])
'''Genetic Algorithm'''
for t in range(T):
    print('Generation: ', t)
    
    #Parents Selection
    parents = []
    while len(parents) < int(N/2):
        #Tournament Selection
        tournament = random.sample(pop, t_size)
        p1 = selection(tournament, 'max')
        tournament = random.sample(pop, t_size)
        p2 = selection(tournament, 'max')
        while p1 == p2:
            tournament = random.sample(pop, t_size)
            p2 = selection(tournament, 'max')  
        
        parents.append(p1)
        parents.append(p2)
    
    #Reproduction
    offspring = []
    while len(offspring) < int(N/2):
        par = random.sample(parents, 2)
        #Crossover + Mutation
        if random.uniform(0,1) >= cr: #Crossover
            p1 = par[0]
            p2 = par[1]
            c1, c2 = crossover(p1[0], p2[0])
            
            #Mutation
            if random.uniform(0,1) >= mr:
                mutation(c1)
            
            if random.uniform(0,1) >= mr:
                mutation(c2)
            
            #Evaluate offspring
            acc_list = manager.list()
            
            #Decoding the network
            network1 = decoding(c1)
            network2 = decoding(c2)
    
            #Creating the CNN 
            cnn1 = CNN(e1, network1[0], network1[1], network1[2])
            cnn2 = CNN(e2, network2[0], network2[1], network2[2])
            
            #Evaluate individuals
            training1 = Process(target = training, args = ('1', device1, cnn1, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
    
            training2 = Process(target = training, args = ('2', device2, cnn2, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
            
            training1.start()
            training2.start()
            training1.join()
            training2.join()
            
            if acc_list[0][0] == '1':
                offspring.append([c1, acc_list[0][1], acc_list[0][2]])
                offspring.append([c2, acc_list[1][1], acc_list[1][2]])
            else:
                offspring.append([c2, acc_list[0][1], acc_list[0][2]])
                offspring.append([c1, acc_list[1][1], acc_list[1][2]])
       
    #Replacement with elitism
    pop = pop + offspring
    pop.sort(reverse = True, key = lambda x: x[1])
    pop = pop[:N]
    
    leader = max(pop, key = lambda x: x[1])
    bestAcc.append(leader[2])
    bestF.append(leader[1])
        
    print('Best fitness: ', leader[1])
    print('Best accuracy: ', leader[2])
    print('--------------------------------------------')

results = pd.DataFrame(list(zip(bestAcc, bestF)), columns = ['Accuracy', 'Fitness'])
pop = []
pop.append(Encoding(min_conv,max_conv,min_full,max_full))
final_networks = []
final_connections = []
for p in pop:
    n_conv = p.n_conv
    n_full = p.n_full
    description = 'The network has ' + str(n_conv) + ' convolutional layers ' + 'with: '
    for i in range(n_conv):
        nfilters = str(p.first_level[i]['nfilters'])
        fsize = str(p.first_level[i]['fsize'])
        pool = str(p.first_level[i]['pool'])
        psize = str(p.first_level[i]['psize'])
        layer = '(' + nfilters + ', ' + fsize + ', ' + pool + ', ' + psize + ') '
        description += layer
    description += 'and '
    description += str(n_full)
    description += ' '
    description += 'fully-connected layers with: '
    for i in range(n_conv, n_conv+n_full):
        neurons = str(p.first_level[i]['neurons'])
        layer = '(' + neurons + ')'
        description += layer
    description += ' neurons'
    final_networks.append(description)
    
    connections = ''
    for bit in p.second_level:
        connections += str(bit)
    final_connections.append(connections)
        
final_population = pd.DataFrame(list(zip(final_networks, final_connections)), columns = ['Network Architecture', 'Connections'])

'''Saving Results as CSV'''
final_population.to_csv('final_population.csv', index = False)
results.to_csv('Final_population.csv', index = False)      
stop = timeit.default_timer()
execution_time = stop-start
print("Execution time: ", execution_time)
    

