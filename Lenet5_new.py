import gzip, struct
import math
import numpy as np
from torch.nn.modules.activation import SELU
from torch.nn.modules.batchnorm import BatchNorm2d


def _read(image, label):
    minist_dir = './MNIST_data/'

    # 使用gzip模块完成对文件的解压
    with gzip.open(minist_dir+label) as flabel:

        # struct提供用format specifier方式对数据进行打包和解包（Packing and Unpacking）
        magic, num=  struct.unpack(">II", flabel.read(8))
        label =np.fromstring(flabel.read(), dtype=np.int8)

    with gzip.open(minist_dir+image, 'rb') as fimg:

        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)

    return image, label


def get_data():

    train_img, train_label = _read(
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz')

    test_img,test_label = _read(
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz')

    return [train_img, train_label, test_img, test_label]


from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import torch

class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d( F.relu(self.conv2(x)),(2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features*=s
        
        return num_features



#使用pytorch封装的dataloader进行训练和预测
from torch.utils.data import TensorDataset,DataLoader, dataloader, dataset
from  torchvision import transforms


def custom_normalization(data, std, mean):
    return  (data-mean)/std

use_gpu = torch.cuda.is_available()

batch_size = 256

kwargs = {'num_workers':2,  'pin_memory':True} if use_gpu else {}

X, y, Xt, yt = get_data()


# 主要进行标准化处理
# mean, std = X.mean(), X.std()
# X = custom_normalization(X, mean, std)
# Xt = custom_normalization(Xt, mean, std)

train_x, train_y = torch.from_numpy(X.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y.astype(int))
test_x, test_y = [
        torch.from_numpy(Xt.reshape(-1,1,28,28)).float(),
        torch.from_numpy(yt.astype(int))
        ]

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, shuffle= True, batch_size=batch_size, **kwargs)
test_loader = DataLoader(dataset= test_dataset, shuffle = True, batch_size= batch_size, **kwargs)

model = LeNet5()

if use_gpu:
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')


criterion = nn.CrossEntropyLoss(size_average=False)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, betas=(0.9, 0.99))

def  weight_init(m):

    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0]*m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2./n))

    elif isinstance(m, nn.BatchNorm2d):
    # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

model.apply(weight_init)

def train(epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        if use_gpu:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)

        target = target.long()
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx %90 ==0:

            print('Train Epoch : {} [{}/{} ({:.0f})%]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 
                100.*batch_idx/len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        target1 = target.long()
        test_loss += criterion(output, target1).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



for epoch in range(1, 501):
    train(epoch)
    test()           



