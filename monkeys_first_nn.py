"""
torch.nn depends on autograd to define and differentiate models
nn.Module contains layers and forward(input) returns output


Theory: forward() progresses forwards through one layer of the network, and then returns the resulting output after the layer(s)



Digit image classifier:

32x32 Input --> C1: feature maps (6@28x28)?? S2: f.maps 6@14x14 C3: f.maps 16@10x10, S4: f, maps 16@5x5 C5: layer 120 F6: layer 84 Output 10

Convulutions = C, Subsampling = S, full connection = F, gaussian connections = output




Typical training procedure for nn is:
    1. Define the network that has learnable params
    2. Iterate over a dataset of inputs
    3. Process input through the network
    4. Compute loss (Error)
    5. Propagate gradients back into network
    6. Update weights, usually with a simple rule:
        weight = weight - learning_rate * gradient
"""



import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# DEFINE the network

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #1 input image channel, 6 output channels, 5x5 square convolution
        #Kernel
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #An affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        #Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        #If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] #All dimensions EXCEPT the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()


#Once the forward function is defined, the backwards function is automatically
# defined with autograd, and any Tensor operations can be used in forward


#The learnable parameters of a model are returned by net.parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())


#The input and output to and from forward is an autograd.Variable
input = Variable(torch.randn(1,1,32,32))
out = net(input)
print(out)

#Zero the gradient buffers of all params/backdrops with random gradients

net.zero_grad()
out.backward(torch.randn(1,10))



#Loss function
#nn.MSELoss = mean-squared error between in and target

output = net(input)
target = Variable(torch.arange(1,11)) #Dummy target
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


#Backpropagation

net.zero_grad() #zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


#Simplest update rule is Stochastic Gradient Descent (SGD)

learning_rate = 0.01

for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
