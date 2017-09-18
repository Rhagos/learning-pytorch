import torch
import torchvision
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


input_size = 1
output_size = 1
epoch = 1000
learning_rate = 0.025

seed = 7734
torch.manual_seed(seed)

dataset = pd.read_csv('Iris_Dataset.csv')
dataset = pd.get_dummies(dataset, columns=['Species'])
values = list(dataset.columns.values)
print(values)

X = dataset[values[1:-3]]
X = np.array(X, dtype = 'float32')
Y = dataset[values[-3:]]
Y = np.array(Y, dtype = 'float32')

#Shuffle the data
indices = np.random.choice(len(X), len(X), replace = False)
X_values = X[indices]
Y_values = Y[indices]

#X_values = X
#Y_values = Y
#Training & Testing dataset creation
test_size = int(-1 * len(X_values) * 0.2)
X_test = X_values[test_size:]
X_train = X_values[:test_size]
Y_test = Y_values[test_size:]
Y_train = Y_values[:test_size]
#Define NN model

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=4, n_hidden=8, n_output=3)


#Loss & Optimizer
criterion = nn.MSELoss()
prev_err = 10000
optimal_learning = learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

for i in range(epoch):
    inputs = Variable(torch.from_numpy(X_train))
    targets = Variable(torch.from_numpy(Y_train))
    
    #Forward, Backwards, optimize
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
#    print('Epoch [%d/%d], Loss: %.4f' %(i+1, epoch, loss.data[0]))
def iris_parser(one_hot_data):
    a,b = one_hot_data.max(0)
    if b.data[0] == 0:
        return 'Iris-setosa'
    elif b.data[0] == 1:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'
errors = 0
for i in range(len(X_test)):
    inputs = Variable(torch.from_numpy(X_test))
    outputs = net(inputs)
    real_result = Variable(torch.from_numpy(Y_test))
    print(iris_parser(real_result[i]), " <--ACTUAL PREDICTED--> ", 
            iris_parser(outputs[i]))
    if not (iris_parser(real_result[i]) == iris_parser(outputs[i])):
        errors += 1
#    if i % 5:
#        print(real_result[i], " vs. ", outputs[i])

print(1- errors/len(X_test))

if errors < prev_err:
    prev_err = errors
    optimal_learning = learning_rate

learning_rate += 0.005
