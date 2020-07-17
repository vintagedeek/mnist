#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[3]:


train_url = "https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/train.csv"
df_train = pd.read_csv(train_url)
df_train.info()


# In[4]:


test_url = 'https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/test.csv'
df_test = pd.read_csv(test_url)
df_test.info()


# In[5]:


train_set = df_train[0:29400]
train = torch.tensor(np.array(train_set.iloc[:, 1:]), dtype=torch.torch.float32)
train_labels = torch.tensor(np.array(train_set.iloc[:, 0]), dtype=torch.int64)
val_set = df_train[29400:]
val = torch.tensor(np.array(val_set.iloc[:, 1:]), dtype=torch.float32)
val_labels = torch.tensor(np.array(val_set.iloc[:, 0]), dtype=torch.int64)


# In[24]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[25]:


device


# In[26]:


train_labels.shape


# In[35]:




df_train_copy = df_train.copy()
df_train_copy = torch.tensor(np.array(df_train_copy), dtype=torch.float32)
pixels_only = torch.narrow(df_train_copy, 1, 1, 784) # start at 1 to 1 +  = 42000 total samples
df_train_copy_mean = pixels_only.sum()/(42000*28*28)
df_train_copy_var = (1 / (42000 * 28 * 28)) * ((pixels_only - df_train_copy_mean)**2).sum()
df_train_copy_std = df_train_copy_var.sqrt()

train_set = df_train_copy[0:29400]
train = torch.narrow(train_set, 1, 1, 784)
train = (train - df_train_copy_mean) / df_train_copy_std
train_labels = torch.narrow(train_set, 1, 0, 1)
train_labels = torch.tensor(train_labels, dtype=torch.int64).squeeze() # they are 2D if you don't squeeze

val_set = df_train_copy[29400:]
val = torch.narrow(val_set, 1, 1, 784)
val = (val - df_train_copy_mean) / df_train_copy_std
val_labels = torch.narrow(val_set, 1, 0, 1)
val_labels = torch.tensor(val_labels, dtype=torch.int64).squeeze() # they are 2D if you don't squeeze

batch_size = 20
epochs = 15
learning_rate = 0.1
lmbda = 0
hidden_neurons_size = 100
in_features = 784 # image is shape (28, 28)


network = nn.Sequential(
    nn.Linear(in_features=in_features, out_features=hidden_neurons_size),
    nn.Sigmoid(),
    nn.BatchNorm1d(100),
    nn.Linear(in_features=hidden_neurons_size, out_features=10),
    nn.Sigmoid(),
    nn.BatchNorm1d(10)))

train_labels.shape
# In[37]:


network[0].weight


# In[38]:


train.shape


# In[39]:


network = network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, weight_decay=lmbda)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for j in range(epochs):
    num_correct = 0
    val_number_correct = 0
    mini_batches = [train[k:k + batch_size] for k in range(0, train_labels.shape[0], batch_size)]
    mini_labels = [train_labels[k:k + batch_size] for k in range(0, train_labels.shape[0], batch_size)]
    for b in range(0, len(mini_batches)):
        images = mini_batches[b].to(device)
        #print(images.shape)
        labels = mini_labels[b].to(device)
        #print(labels.shape)
        #print(labels.dtype)
        #print(images.dtype)
        preds = network(images)
        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_correct += preds.argmax(dim=1).eq(labels).sum().item()

    validation_set = val.to(device)
    validation_labels = val_labels.to(device)
    validation_preds = network(validation_set)
    val_number_correct += validation_preds.argmax(dim=1).eq(validation_labels).sum().item()

    print('Completed epoch: ', j)
    print('Train accuracy: ', num_correct/len(train))
    print('Validation Accuracy: ', val_number_correct/len(validation_set))

    scheduler.step()


# In[13]:


# Train accuracy
num_correct/len(train)


# In[14]:


len(train)


# In[15]:


# val accuracy
val = val.to(device)
val_labels = val_labels.to(device)
val_preds = network(val)
val_num_correct = val_preds.argmax(dim=1).eq(val_labels).sum().item()
val_num_correct/len(val_preds)


# ![image.png](attachment:image.png)

# In[ ]:
