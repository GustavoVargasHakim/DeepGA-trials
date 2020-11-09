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


'''Loading data'''
def loading_data():
    #Loading Datasets
    covid_ds = CovidDataset(root = c_root, labels = c_labels, transform = transforms.Compose([ToTensor()]))
    normal_ds = CovidDataset(root = n_root, labels = n_labels, transform = transforms.Compose([ToTensor()]))
    pneumonia_ds = CovidDataset(root = p_root, labels = p_labels, transform = transforms.Compose([ToTensor()]))
    
    #Merging Covid, normal, and pneumonia Datasets
    dataset = torch.utils.data.ConcatDataset([covid_ds, normal_ds, pneumonia_ds])
    lengths = [int(len(dataset)*0.7), int(len(dataset)*0.3)+1]
    train_ds, test_ds = torch.utils.data.random_split(dataset = dataset, lengths = lengths)
    
    i = 1836
    #Testing
    print("Length of Training Dataset: {}".format(len(train_ds)))
    print("Length of Test Dataset: {}".format(len(test_ds)))
    print("Shape of images as tensors: {}".format(dataset[i]['image'].shape))
    print("Label of image i: {}".format(dataset[i]['label']))
    
    #Creating Dataloaders
    train_dl = DataLoader(train_ds, batch_size = 24, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = 24, shuffle = True)
    
    return train_dl, test_dl

train_dl, test_dl = loading_data()

#Iterate over batches
#for i_batch, sample_batched in enumerate(test_dl):
#  print(sample_batched['image'].shape)
#  print(sample_batched['label'])
#  break

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
N = 4 #Population size
T = 50 #Number of generations
t_size = 5 #tournament size
w = 0.3 #penalization weight
max_params = 1.5e6
num_epochs = 10


#print('GPUs: ', torch.cuda.device_count())

'''Evaluating the objective function of an encoding (accuracy + w*No. Params)'''
def evaluate_individual(x, dev):
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
    
    return f, accuracy

'''Evaluating the objective function of an encoding (accuracy + w*No. Params)'''
def evaluate_individual2(x, y):
    #Decoding the network
    network1 = decoding(x)
    network2 = decoding(y)
    
    #Creating the CNN (and obtaining number of parameters)
    cnn1 = CNN(x, network1[0], network1[1], network1[2])
    cnn2 = CNN(y, network2[0], network2[1], network2[2])
    
    params1 = sum(p.numel() for p in cnn1.parameters() if p.requires_grad)
    params2 = sum(p.numel() for p in cnn2.parameters() if p.requires_grad)
    
    #Passing the CNN to a GPU 
    #cnn = nn.DataParallel(cnn) #Uncomment this if more than one GPU is available
    #cnn.to(device, dtype = torch.float32)
    
    #Defining optimizer
    opt = optim.Adam(cnn.parameters(), lr = lr)
    
    #Training the network
    accuracy1, accuracy2, _, _ = train_val2(num_epochs, cnn1, cnn2, loss_func, train_dl, test_dl)
    
    #Fitness function
    f = abs(accuracy - w*(1 - abs((max_params - params)/max_params)))
    
    return f, accuracy


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
    
    #Creating a genome (genetic encoding)
    e1 = Encoding(min_conv,max_conv,min_full,max_full) 
    e2 = Encoding(min_conv,max_conv,min_full,max_full)
    
    #Decoding the network
    network1 = decoding(e1)
    network2 = decoding(e2)
    
    #Creating the CNN (and obtaining number of parameters)
    cnn1 = CNN(e1, network1[0], network1[1], network1[2])
    cnn2 = CNN(e2, network2[0], network2[1], network2[2])
    
    #Passing to GPU
    #cnn1.to(device1, dtype = torch.float32)
    #cnn2.to(device2, dtype = torch.float32)
       
    #Evaluate individual
    #f, accuracy = evaluate_individual2(e1, e2)
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
        pop.append([e2, acc_list[1][1], acc_list[0][2]])
    else:
        pop.append([e2, acc_list[0][1], acc_list[0][2]])
        pop.append([e1, acc_list[1][1], acc_list[0][2]])

stop = timeit.default_timer()
execution_time = stop-start
print('Training time of 4 Networks: ', execution_time)
'''Genetic Algorithm'''
'''for t in range(T):
    print('Generation: ', t)
    
    offspring = []
    while len(offspring) < int(N/2):
        #Parents selection through tournament 
        tournament = random.sample(pop, t_size)
        p1 = selection(tournament, 'max')
        tournament = random.sample(pop, t_size)
        p2 = selection(tournament, 'max')
        while p1 == p2:
            tournament = random.sample(pop, t_size)
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
        
       
    #Replacement with elitism
    pop = pop + offspring
    pop.sort(key = lambda x: x[1])
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
    final_connections.append(connections)'''
        
#final_population = pd.DataFrame(list(zip(final_networks, final_connections)), columns = ['Network Architecture', 'Connections'])

'''Saving Results as CSV'''
#final_population.to_csv('final_population.csv', index = False)
#results.to_csv('Final_population.csv', index = False)      

    
    

