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
import numpy as np
from torch import optim
import pandas as pd


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
    #print("Length of Training Dataset: {}".format(len(train_ds)))
    #print("Length of Test Dataset: {}".format(len(test_ds)))
    print("Shape of images as tensors: {}".format(dataset[i]['image'].shape))
    print("Label of image i: {}".format(dataset[i]['label']))
    
    #Creating Dataloaders
    train_dl = DataLoader(train_ds, batch_size = 24, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = 24, shuffle = True)
    
    return train_dl, test_dl
train_dl, test_dl = loading_data()

#Iterate over batches
for i_batch, sample_batched in enumerate(test_dl):
  print(sample_batched['image'].shape)
  print(sample_batched['label'])
  break

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
w = 0.3 #penalization weight
max_params = 1.5e6
lr = 1e-4
num_epochs = 10

#Reading GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)

'''Evaluating the objective function of an encoding (accuracy + w*No. Params)'''
def evaluate_individual(x):
    #Decoding the network
    network = decoding(x)
    
    #Creating the CNN (and obtaining number of parameters)
    cnn = CNN(x, network[0], network[1], network[2])
    parms = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    
    #Passing the CNN to a GPU (if only one GPU is available)
    cnn.to(device, dtype = torch.float32)
    
    #Defining optimizer
    opt = optim.Adam(cnn.parameters(), lr = lr)
    
    #Training the network
    accuracy, _ = train_val(num_epochs, cnn, loss_func, opt, train_dl, test_dl)
    
    #Fitness function
    f = abs(accuracy - w*(1 - abs((max_params - params)/max_params)))
    
    return f, accuracy
    
'''Initialize population'''
pop = []
bestAcc = []
bestF = []
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
    final_connections.append(connections)
        
final_population = pd.DataFrame(list(zip(final_networks, final_connections)), columns = ['Network Architecture', 'Connections'])

'''Saving Results as CSV'''
final_population.to_csv('final_population.csv', index = False)
results.to_csv('Final_population.csv', index = False)      

    
    

