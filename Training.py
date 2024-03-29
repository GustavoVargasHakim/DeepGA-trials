# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:43:11 2020

@author: user
"""

import torch
from torch import nn
from torch import optim

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
def train_val(device, epochs, model, opt, loss_func, train_dl, test_dl):
  lr = 1e-4
  #Reading GPU
  #if torch.cuda.is_available():
  #device = torch.device("cuda:0")
  #print(device)
  #if torch.cuda.device_count() > 1:
  #model = nn.DataParallel(model, device_ids=[0,1], output_device = device).to(device)
  #model.to(device)
  
  #opt = optim.Adam(model.parameters(), lr = lr)
  
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Helper function to compute the loss and metric values for a dataset
def loss_epoch2(device1, device2, model1, model2, loss_func, dataset_dl, opt1 = None, opt2 = None):
  loss1 = 0.0
  loss2 = 0.0
  metric1 = 0.0
  metric2 = 0.0
  len_data = len(dataset_dl.dataset)
  for i, data in enumerate(dataset_dl, 0):
    #print('batch: ', i)
    xb, yb = data['image'], data['label']
    xb2, yb2 = xb.clone(), yb.clone()
    xb = xb.type(torch.double).to(device1, dtype = torch.float32)
    yb = yb.to(device1, dtype = torch.long)
    xb2 = xb2.type(torch.double).to(device2, dtype = torch.float32)
    yb2 = yb2.to(device2, dtype = torch.long)
    
    #Obtain model output
    yb_h = model1(xb)
    yb_h2 = model2(xb2)

    loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt1)
    loss_b2, metric_b2 = loss_batch(loss_func, xb2, yb2, yb_h2, opt2)
    #metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
    loss1 += loss_b
    loss2 += loss_b2
    if metric_b is not None:
      metric1 += metric_b
    if metric_b is not None:
      metric2 += metric_b2
  
  loss1 /= len_data
  loss2 /= len_data
  metric1 /= len_data
  metric2 /= len_data

  return loss1, metric1, loss2, metric2

#Define the training function
def train_val2(epochs, model1, model2, loss_func, train_dl, test_dl):
  lr = 1e-4
  #Reading GPU
  device1 = torch.device("cuda:0")
  device2 = torch.device("cuda:1")
  model1.to(device1)
  model2.to(device2)
  
  opt1 = optim.Adam(model1.parameters(), lr = lr)
  opt2 = optim.Adam(model2.parameters(), lr = lr)
  
  for epoch in range(epochs):
    #print(epoch)
    model1.train()
    model2.train()
    train_loss1, train_metric1, train_loss2, train_metric2 = loss_epoch(device1, device2, model1, model2, loss_func, train_dl, opt1, opt2)
    #train_metric = loss_epoch(model, loss_func, train_dl, opt)
    model1.eval()
    model2.eval()
    with torch.no_grad():
      val_loss1, val_metric1, val_loss2, val_metric2 = loss_epoch(device1, device2, model1, model2, loss_func, test_dl)
      #val_metric = loss_epoch(model, loss_func, test_dl)
    accuracy1 = 100*val_metric1
    accuracy2 = 100*val_metric2

    #print("Epoch: %d, train loss: %.6f, val loss: %.6f, test accuracy: %.2f" %(epoch, train_loss, val_loss, accuracy))
  
  return accuracy1, accuracy2, model1, model2