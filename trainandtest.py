'''
Created on Apr 3, 2018

@author: fox
'''

import torch
import torchvision
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import capsule
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default="/home/fox/MNIST_data")
parser.add_argument('-batch', type=int, default=4)
parser.add_argument('-iter', type=int, default=1)
parser.add_argument('-display', type=int, default=50)
args = parser.parse_args()
# configurations
data_dir = args.data
width = 28
height = 28
A = 32
B = 16
C = 16
D = 16
E = 10
iterations = args.iter 
batch_size = args.batch
learning_rate = 1.0e-3
training_iters = 30000
display_step = args.display

# loading the mnist data
train_data = torchvision.datasets.MNIST(root=data_dir, 
                                        train=True, 
                                        transform=torchvision.transforms.ToTensor(),
                                        download=False
                                        )

test_data = torchvision.datasets.MNIST(root=data_dir, 
                                       train = False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False
                                       )


train_loader = Data.DataLoader(train_data, batch_size, shuffle=False, num_workers=1)
test_loader = Data.DataLoader(test_data, batch_size, shuffle=False, num_workers=1)

model = capsule.myCuda(capsule.CapsNet(A, B, C, D, E, iterations))


optimizer = optim.Adam(model.parameters(), lr=learning_rate)




for iter in range(training_iters):
    
    model.train()
    for step, (batch_x, batch_y) in enumerate(train_loader): 
        # (bs, 1, 28, 28)
        batch_x = capsule.myCuda(autograd.Variable(batch_x))
        batch_y = capsule.myCuda(autograd.Variable(batch_y))
        
        
        pred = model.forward(batch_x)
        
        cost = model.spread_loss(pred, batch_y)
        
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
 
        
        if (step + 1) % display_step == 0:
            print("Iter " + str(iter + 1) + ", Step "+str(step+1)+", Avarage Loss: " + "{:.6f}".format(cost.data[0]))
            pred_y = torch.max(pred, 1)[1].data.squeeze()
            accuracy = sum(pred_y == batch_y.data) / float(batch_y.size(0))
            print("Iter " + str(iter + 1) + ", Step "+str(step+1)+", Train Batch Accuarcy: " + "{:.6f}".format(accuracy))
        
    # validation performance
    model.eval()
    total_correct = 0
    total_gold = 0
    for batch_x, batch_y in test_loader:
        batch_x = capsule.myCuda(autograd.Variable(batch_x, volatile=True))
        batch_y = capsule.myCuda(autograd.Variable(batch_y, volatile=True))
    
        test_output = model.forward(batch_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        total_correct += sum(pred_y == batch_y.data) 
        total_gold += batch_y.size(0)
        
    print("========> Test Accuarcy: {:.6f}".format(total_correct/total_gold))
         



