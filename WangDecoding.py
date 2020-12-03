# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:31:31 2020

@author: user
"""

from torch import nn
import math
import torch

def conv_out_size(W, K):
    return W - K + 3

def pool_out_size(W, K):
    return math.floor((W - K)/2) + 1

class DenseBlock(nn.Module):
    def __init__(self, input_channels, block, connections, init_weights = True):
        super(DenseBlock, self).__init__()
        n_filters = block['nfilters']
        n_conv = block['nconv']
        layers = []
        dense = [] 
        prev = -1
        pos = 0
        for i in range(n_conv):
          if i == 0 or i == 1:
            layers.append([nn.Conv2d(in_channels = input_channels, out_channels = n_filters, kernel_size = 3, padding = 1, stride = 1),
                           nn.BatchNorm2d(n_filters),
                           nn.ReLU(inplace = True)])
            dense += [nn.Conv2d(in_channels = input_channels, out_channels = n_filters, kernel_size = 3, padding = 1, stride = 1),
                           nn.BatchNorm2d(n_filters),
                           nn.ReLU(inplace = True)]
            input_channels = n_filters

          else:
            conn = connections[pos:pos+prev]
            new_inputs = 0
            for c in range(len(connections[pos:pos+prev])):
              if conn[c] == 1:
                new_inputs += n_filters
            layers.append([nn.Conv2d(in_channels = input_channels+new_inputs, out_channels = n_filters, kernel_size = 3, padding = 1, stride = 1),
                             nn.BatchNorm2d(n_filters),
                             nn.ReLU(inplace = True)])
            dense += [nn.Conv2d(in_channels = input_channels+new_inputs, out_channels = n_filters, kernel_size = 3, padding = 1, stride = 1),
                        nn.BatchNorm2d(n_filters),
                        nn.ReLU(inplace = True)]
            pos += prev
          prev += 1

        self.layers = layers
        self.connections = connections
        self.denseblock = nn.Sequential(*dense)
        self.n_conv = n_conv
        
    def forward(self, x):
        prev = -1
        pos = 0
        outputs = []
        layers = self.layers
        for i in range(self.n_conv):
            if i == 0 or i == 1:
                x = nn.Sequential(*layers[i])(x)
                outputs.append(x)
                
            else:
                connections = self.connections[pos:pos+prev]
                for c in range(len(connections)):
                    if connections[c] == 1:
                        x2 = outputs[c]
                        x = torch.cat((x, x2), axis = 1)
                x = nn.Sequential(*layers[i])(x)
                outputs.append(x)
                pos += prev
            prev += 1    

def decoding(encoding):
  n_block = encoding.n_block
  n_full = encoding.n_full
  first_level = encoding.first_level
  second_level = encoding.second_level

  '''Components'''
  features = []
  classifier = []
  in_channels = 1
  out_size = 256
  for i in range(n_block):
    block = first_level[i]
    connections = second_level[i]
    g_rate = block['nfilters']
    denseBlock = DenseBlock(in_channels, block, connections)
    features += [denseBlock, 
                 nn.Conv2d(in_channels = in_channels, out_channels = int(g_rate/2), kernel_size = 3, padding = 0, stride = 1),
                 nn.MaxPool2d(kernel_size = 2, stride = 2)]
    in_channels = int(g_rate/2) 
    out_size = conv_out_size(out_size, 3)
    out_size = pool_out_size(out_size, 2)

  in_size = out_size*out_size*in_channels  
  classifier = []
  for i in range(n_block,n_block+n_full):
    block = first_level[i]
    n_neurons = block['neurons']
    classifier += [nn.Linear(in_size, n_neurons)]
    in_size = n_neurons
  
  classifier += [nn.Linear(n_neurons, 3)]
  
  return features, classifier

'''Networks class'''
class CNN(nn.Module):
  def __init__(self, features, classifier, init_weights = True):
    super(CNN, self).__init__()
    self.extraction = nn.Sequential(*features)
    self.classifier = nn.Sequential(*classifier)
    
  def forward(self, x):
    '''Feature extraction'''
    x = self.extraction(x)
    x = torch.flatten(x,1)
    '''Classification'''
    x = self.classifier(x)

    return nn.functional.log_softmax(x, dim=1)