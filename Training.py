# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:43:11 2020

@author: user
"""

import torch
from torch import nn

#Helper function to compute de loss on a batch
def loss_batch(loss_func, xb, yb, yb_h, opt = None):
  #Obtain the loss
  loss = loss_func(yb_h, yb)
  #Obtain peformance metric
  metric_b = metrics_batch(yb, yb_h)
  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()
  
  return loss.item(), metric_b
  #return metric_b

#Helper function to compute the accuracy per mini_batch
def metrics_batch(target, output):
  #Obtain output class
  pred = output.argmax(dim=1, keepdim = True)
  #Compare output class with target class
  corrects = pred.eq(target.view_as(pred)).sum().item()

  return corrects

#Helper function to compute the loss and metric values for a dataset
def loss_epoch(device, model, loss_func, dataset_dl, opt = None):
  loss = 0.0
  metric = 0.0
  len_data = len(dataset_dl.dataset)
  for i, data in enumerate(dataset_dl, 0):
    #print('batch: ', i)
    xb, yb = data['image'], data['label']
    xb = xb.type(torch.double).to(device, dtype = torch.float32)
    yb = yb.to(device, dtype = torch.long)
    
    #Obtain model output
    yb_h = model(xb)

    loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
    #metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
    loss += loss_b
    if metric_b is not None:
      metric += metric_b
  
  loss /= len_data
  metric /= len_data

  return loss, metric
  #return metric

#Define the training function
def train_val(epochs, model, loss_func, train_dl, test_dl):
  lr = 1e-4
  #Reading GPU
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
  if torch.cude.device_count() > 1:
      model = nn.DataParallel(model)
  model.to(device)
  
  opt = optim.Adam(cnn.parameters(), lr = lr)
  
  for epoch in range(epochs):
    #print(epoch)
    model.train()
    train_loss, train_metric = loss_epoch(device, model, loss_func, train_dl, opt)
    #train_metric = loss_epoch(model, loss_func, train_dl, opt)
    model.eval()
    with torch.no_grad():
      val_loss, val_metric = loss_epoch(device, model, loss_func, test_dl)
      #val_metric = loss_epoch(model, loss_func, test_dl)
    accuracy = 100*val_metric

    #print("Epoch: %d, train loss: %.6f, val loss: %.6f, test accuracy: %.2f" %(epoch, train_loss, val_loss, accuracy))
  
  return accuracy, model