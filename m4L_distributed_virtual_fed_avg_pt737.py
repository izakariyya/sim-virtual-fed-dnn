# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:13:21 2020

@author: 1804499
"""

import syft as sy
import numpy as np
import time
import memory_profiler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
hook = sy.TorchHook(torch)

#Create couple of workers

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id='alice')
jake = sy.VirtualWorker(hook, id="jake")
jane = sy.VirtualWorker(hook, id='jane')
secure_worker = sy.VirtualWorker(hook, id="secure_worker")


def data():
    benign = np.loadtxt("benign_traffic.csv", delimiter = ",")
    mirai = np.loadtxt("mirai_traffic.csv", delimiter = ",")
    gafgyt = np.loadtxt("gafgyt_traffic.csv", delimiter = ",")
    alldata = np.concatenate((benign, gafgyt, mirai))
    j = len(benign[0])
    data = alldata[:, 1:j] 
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2, random_state = 42)
    return bendata, benmir, benlabel, benslabel


traind, testd, trainlbl, testlbl =  data()

traind = torch.FloatTensor(traind)

testd = torch.FloatTensor(testd)

trainlbl = torch.FloatTensor(trainlbl)

n = len(traind)
part1 = int(0.25*n)
part2 = int(0.50*n)
part3 = int(0.75*n)

#testlbl = torch.FloatTensor(testlbl)
torch.manual_seed(0)
# Define network dimensions
n_input_dim = traind.shape[1]
# Layer size
n_hidden1 = 83
n_hidden2 = 128 # Number of hidden nodes
n_output = 1 # Number of output nodes = for binary classifier


#Build and initialize network (model)
model = nn.Sequential(
    nn.Linear(n_input_dim, n_hidden1),
    nn.ReLU(),
    nn.Linear(n_hidden1, n_hidden2),
    nn.Linear(n_hidden2, n_hidden2),
    nn.Linear(n_hidden2, n_hidden1),
    nn.Linear(n_hidden1, n_output),
    nn.Sigmoid()) 

# Define the loss function
#loss_fn = torch.nn.BCELoss() 
learning_rate = 0.001
eps = 0.001
epochs = 4
worker_iter = 30
batch_size = 128
m_batch_size = 4

cross_entropy = nn.BCELoss()

# Cross Entropy Cost Function

#def cross_entropy(input, target, eps):
#    input = torch.clamp(input,min=1e-7,max=1-1e-7)
#    bce = - (target * torch.log(input + eps) + (1 - target + eps) * torch.log(1 - input))
#    return torch.mean(bce)

# Regularized Cost

#def cross_reg(input, target, eps, lambd):
#    rloss = cross_entropy(input, target, eps)
#    rloss = rloss * lambd
#    return rloss
         
#Full Training

def train_base(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size):
    #Create data for Bob and Alice
    #size = int(len(traind) / 2)
    
    bobs_data = traind[0:part1]
    bobs_target = trainlbl[0:part1]
    alices_data = traind[part1:part2]
    alices_target = trainlbl[part1:part2]
    jakes_data = traind[part2:part3]
    jakes_target = trainlbl[part2:part3]
    janes_data = traind[part3:]
    janes_target = trainlbl[part3:]
    
    
    #batch_number = bobs_data.size()[0] // batch_size
    
    for i in range(epochs):
        # X is a torch Variable
        #indices = epochs % batch_number
        permutation = torch.randperm(bobs_data.size()[0])
        indices = permutation[i:i+batch_size]
        bobs_data_batch, bobs_target_batch = bobs_data[indices].send(bob), bobs_target[indices].send(bob)
        alices_data_batch, alices_target_batch = alices_data[indices].send(alice), alices_target[indices].send(alice)
        jakes_data_batch, jakes_target_batch = jakes_data[indices].send(jake), jakes_target[indices].send(jake)
        janes_data_batch, janes_target_batch = janes_data[indices].send(jane), janes_target[indices].send(jane)
        
        #Send model to workers
        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)
        jakes_model = model.copy().send(jake)
        janes_model = model.copy().send(jane)
        boptim = torch.optim.SGD(bobs_model.parameters(), lr=learning_rate)
        aoptim = torch.optim.SGD(alices_model.parameters(), lr=learning_rate)
        jkoptim = torch.optim.SGD(jakes_model.parameters(), lr=learning_rate)
        jnoptim = torch.optim.SGD(janes_model.parameters(), lr=learning_rate)
        
        # Training virtual workers script
        for i in range(worker_iter):
            #Bobs Training
            boptim.zero_grad()
            b_yhat = bobs_model(bobs_data_batch) 
            bloss = cross_entropy(b_yhat.reshape(-1), bobs_target_batch)
            bloss.backward()
        
            boptim.step()
            bloss = bloss.get().data
                
            #Alices Training
            aoptim.zero_grad()
            a_yhat = alices_model(alices_data_batch)
            aloss = cross_entropy(a_yhat.reshape(-1), alices_target_batch)
            aloss.backward()
        
            aoptim.step()
            aloss = aloss.get().data
            
            #Jakes Training
            jkoptim.zero_grad()
            jk_yhat = jakes_model(jakes_data_batch)
            jkloss = cross_entropy(jk_yhat.reshape(-1), jakes_target_batch)
            jkloss.backward()
            
            jkoptim.step()
            jkloss = jkloss.get().data
            
            #Janes Training
            jnoptim.zero_grad()
            jn_yhat = janes_model(janes_data_batch)
            jnloss = cross_entropy(jn_yhat.reshape(-1), janes_target_batch)
            jnloss.backward()
            
            jnoptim.step()
            jnloss = jnloss.get().data
            
            
        
                  
        #Send Both Updated Models to a Secure Worker
   
        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)
        jakes_model.move(secure_worker)
        janes_model.move(secure_worker)
        #james_model.move(secure_worker)
    
        #obtaining model weights and averaging them
        paramb = []
        for param in bobs_model.parameters():
            paramb.append(param.view(-1))
        paramb = torch.cat(paramb)
        parama = []
        for param in alices_model.parameters():
            parama.append(param.view(-1))
        parama = torch.cat(parama)
        paramjk = []
        for param in jakes_model.parameters():
            paramjk.append(param.view(-1))
        paramjk = torch.cat(paramjk)
        paramjn = []
        for param in janes_model.parameters():
            paramjn.append(param.view(-1))
        paramjn = torch.cat(paramjn)
       
        
        
        #Averaging model weights
        (parama + paramb + paramjk + paramjn) / 4
    return bloss, aloss, jkloss, jnloss, model

def train_efficient(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size, m_batch_size):
    #Create data for Bob and Alice
    #size = int(len(traind) / 2)
    #W_c = 0.01
    #W_t = 0.01
    lambd = 0.01
    #lambi = 0.01

    bobs_data = traind[0:part1]
    bobs_target = trainlbl[0:part1]
    alices_data = traind[part1:part2]
    alices_target = trainlbl[part1:part2]
    jakes_data = traind[part2:part3]
    jakes_target = trainlbl[part2:part3]
    janes_data = traind[part3:]
    janes_target = trainlbl[part3:]
   
    
   
    for i in range(epochs):
        
        permutation = torch.randperm(bobs_data.size()[0])
        indices = permutation[i:i+batch_size]
        # Mini-Batch
        bobs_data_batch, bobs_target_batch = bobs_data[indices].send(bob), bobs_target[indices].send(bob)
        alices_data_batch, alices_target_batch = alices_data[indices].send(alice), alices_target[indices].send(alice)
        jakes_data_batch, jakes_target_batch = jakes_data[indices].send(jake), jakes_target[indices].send(jake)
        janes_data_batch, janes_target_batch = janes_data[indices].send(jane), janes_target[indices].send(jane)
        
        # Micro-Batch
        mindices = indices / m_batch_size
        
        bobs_data_batch, bobs_target_batch = bobs_data[mindices].send(bob), bobs_target[mindices].send(bob)
        alices_data_batch, alices_target_batch = alices_data[mindices].send(alice), alices_target[mindices].send(alice)
        jakes_data_batch, jakes_target_batch = jakes_data[mindices].send(jake), jakes_target[mindices].send(jake)
        janes_data_batch, janes_target_batch = janes_data[mindices].send(jane), janes_target[mindices].send(jane)
        
        #Send model to workers
        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)
        jakes_model = model.copy().send(jake)
        janes_model = model.copy().send(jane)
        boptim = torch.optim.SGD(bobs_model.parameters(), lr=learning_rate, weight_decay=lambd)
        aoptim = torch.optim.SGD(alices_model.parameters(), lr=learning_rate, weight_decay=lambd)
        jkoptim = torch.optim.SGD(jakes_model.parameters(), lr=learning_rate, weight_decay=lambd)
        jnoptim = torch.optim.SGD(janes_model.parameters(), lr=learning_rate, weight_decay=lambd)
        
        
        # Training virtual workers script
        for i in range(worker_iter):
            #Bobs Training
            for param in bobs_model.parameters():
                param.grad = None#   
            b_yhat = bobs_model(bobs_data_batch) 
            bloss = cross_entropy(b_yhat.reshape(-1), bobs_target_batch)
            bloss.backward()
        
            boptim.step()
            bloss = bloss.get().data
                
            #Alices Training
            for param in alices_model.parameters():
                param.grad = None#  
            a_yhat = alices_model(alices_data_batch)
            aloss = cross_entropy(a_yhat.reshape(-1), alices_target_batch)
            aloss.backward()
        
            aoptim.step()
            aloss = aloss.get().data
            
            #Jakes Training
            for param in jakes_model.parameters():
                param.grad = None#  
            jk_yhat = jakes_model(jakes_data_batch)
            jkloss = cross_entropy(jk_yhat.reshape(-1), jakes_target_batch)
            jkloss.backward()
            
            jkoptim.step()
            jkloss = jkloss.get().data
            
            #Janes Training
            for param in janes_model.parameters():
                param.grad = None#  
            jn_yhat = janes_model(janes_data_batch)
            jnloss = cross_entropy(jn_yhat.reshape(-1), janes_target_batch)
            jnloss.backward()
            
            jnoptim.step()
            jnloss = jnloss.get().data
            
            
        
            #if bloss <= bl:
            #    lambd = lambd + lambi
            #if aloss <= al:
            #    lambd = lambd + lambi
            #if jkloss <= jkl:
            #    lambd = lambd + lambi
            #if jnloss <= jnl:
            #    lambd = lambd + lambi
            
        #Send Both Updated Models to a Secure Worker
        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)
        jakes_model.move(secure_worker)
        janes_model.move(secure_worker)
       
    
        #obtaining model weights and averaging them
        paramob = []
        for param in bobs_model.parameters():
            paramob.append(param.view(-1))
        paramob = torch.cat(paramob)
        paramoa = []
        for param in alices_model.parameters():
            paramoa.append(param.view(-1))
        paramoa = torch.cat(paramoa)
        paramojk = []
        for param in jakes_model.parameters():
            paramojk.append(param.view(-1))
        paramojk = torch.cat(paramojk)
        paramojn = []
        for param in janes_model.parameters():
            paramojn.append(param.view(-1))
        paramojn = torch.cat(paramojn)
        
        #Averaging model weights
        (paramoa + paramob + paramojk + paramojn) / 4
    
    return bloss, aloss, jkloss, jnloss, model

#Baseline Computational Resources
    
starttbase = time.time()
startmbase = memory_profiler.memory_usage()

bl, al, jkl, jnl,modelb = train_base(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size)

endtbase =time.time()
endmbase = memory_profiler.memory_usage()
traintime_base = endtbase - starttbase
train_memory_base = endmbase[0] - startmbase[0]

print("Training time base: {:2f} sec".format(traintime_base))
print("Training memory base: {:2f} mb".format(train_memory_base))


#Optimized Computational Resources

starttefi = time.time()
startmefi = memory_profiler.memory_usage()

obl, oal, ojkl, ojnl, modelo = train_efficient(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size, m_batch_size)

endtefi = time.time()
endmefi = memory_profiler.memory_usage()
traintime_efi = endtefi - starttefi
train_memory_efi = endmefi[0] - startmefi[0]

print("Training time optimize: {:2f} sec".format(traintime_efi))
print("Training memory optimize: {:2f} mb".format(train_memory_efi))


def predict(model, X, Y):
    y_hat = model(X)
    y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
    accuracy = np.sum(Y.reshape(-1,1) ==y_hat_class) / len(Y)
    return accuracy
 
acc_b = predict(modelb, testd, testlbl)

acc_o = predict(modelo, testd, testlbl)

print("Test accuracy base: {:2f}".format(acc_b))
print("Test accuracy optimize: {:2f}".format(acc_o))