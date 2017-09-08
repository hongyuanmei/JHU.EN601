from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


start = time.time()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

end = time.time()
timetaken =  (end - start) #seconds

print("Loading samples:")
print("Number of samples in training: %d" % len(train_loader.dataset))
print("Number of samples in testing: %d" % len(test_loader.dataset))
print("Total loading time taken: %.3fs" % timetaken)
print("Total loading time per sample: %.10fs\n\n" % (timetaken/70000.0))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    start = time.time()
    total_training_time = 0 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss.data[0]))

            end = time.time()
            time_per_b = end - start
            start = time.time()
            total_training_time += time_per_b
    avg_time_per_batch = total_training_time/len(train_loader)
    print("Batch size: %s" % args.batch_size)
    print("Average time/batch: %ss" % avg_time_per_batch)
    print("Average time/batch/sample: %.10fs" % (avg_time_per_batch / args.batch_size))
    print("Total training time: %.10fs" % total_training_time)
                

def test():

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

tot_train_time = 0
tot_test_time = 0

for epoch in range(1, args.epochs + 1):
    tot_train_start = time.time()
    print("Iteration %s...."%epoch)
    start = time.time()
    train(epoch)
    end = time.time()
    tot_train_end = time.time()

    tot_train_time += tot_train_end - tot_train_start

    start = time.time()
    test()
    end = time.time()
    print("Total time taken for evaluation: %.10fs"%(end - start))
    print("Average time per sample: %.10fs\n\n\n\n"%((end-start)/len(test_loader.dataset)))

    tot_test_time += end - start



print("Total training time for %s iteratons: %.10fs"%(args.epochs, tot_train_time))

tot_time_per_it = tot_train_time / args.epochs
print("Average training time/iteration: %.10fs" % tot_time_per_it)

tot_time_per_it_per_batch = tot_time_per_it / len(train_loader)
print("Average training time/iteration/batch: %.10fs" % tot_time_per_it_per_batch)

tot_time_per_it_per_batch_per_sample = tot_time_per_it_per_batch / args.batch_size
print("Average training time/iteration/batch/sample: %.10fs\n\n" % tot_time_per_it_per_batch_per_sample)


print("Total testing time for %s iteratons: %.10fs"%(args.epochs, tot_test_time))

tot_time_per_it = tot_test_time / args.epochs
print("Average testing time/iteration: %.10fs" % tot_time_per_it)

tot_time_per_it_per_sample = tot_time_per_it / len(test_loader.dataset)
print("Average testing time/iteration/sample: %.10fs" % tot_time_per_it_per_sample)

