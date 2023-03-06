import argparse
import os

import torch
import torch.nn as nn

from redunet import *
import evaluate
import functional as F
import load as L
import utils
import plot

#参数
num_epochs=95
classes=5
channels=35

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,default=r'saved_models\forward\mnist2d_5class+lift2d_channels35_layers20\samples10', help='model directory')
parser.add_argument('--loss', default=False, action='store_true', help='set to True if plot loss')
parser.add_argument('--trainsamples', type=int, default=5, help="number of train samples in each class")
parser.add_argument('--testsamples', type=int, default=5, help="number of train samples in each class")
parser.add_argument('--translatetrain', default=False, action='store_true', help='set to True if translation train samples')
parser.add_argument('--translatetest', default=False, action='store_true', help='set to True if translation test samples')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for evaluation')
args = parser.parse_args()


#定义网络
class MyNet(nn.Module):
    def __init__(self, base_model):  ####经常改
        super(MyNet,self).__init__()
        self.model = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),  # 0.75
            nn.Linear(in_features=channels*28*28, out_features=classes)
            ) #nn.Softmax() 因为在交叉熵中已经对输入input做了softmax了。

    def forward(self,input):
        out = self.model.forward(input, loss=args.loss) #参数写哪？
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## Setup
eval_dir = os.path.join(args.model_dir,
                        f'trainsamples{args.trainsamples}'
                        f'_testsamples{args.testsamples}'
                        f'_translatetrain{args.translatetrain}'
                        f'_translatetest{args.translatetest}')
params = utils.load_params(args.model_dir)

## Data
trainset, testset, num_classes = L.load_dataset(params['data'], data_dir=params['data_dir'])
X_train, y_train = F.get_samples(trainset, args.trainsamples)
X_test, y_test = F.get_samples(testset, args.testsamples)
if args.translatetrain:
    X_train, y_train = F.translate(X_train, y_train, stride=7)
if args.translatetest:
    X_test, y_test = F.translate(X_test, y_test, stride=7)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

## Architecture
net = L.load_architecture(params['data'], params['arch'])
net = utils.load_ckpt(args.model_dir, 'model', net)
net = MyNet(net)
net = net.to(device)

# 随机初始化
from torch.nn import init
for name, param in net.named_parameters():
    init.normal_(param, mean=0, std=0.01)

## Forward

print('train')
opt = torch.optim.SGD(net.parameters(), lr=0.001)  #parameters=net.parameters() optim？optimizer ?
opt.zero_grad()  # 清空梯度
for epoch in range(num_epochs):
    outputs = []
    loss_each_epoch = 0
    for i in range(0, classes*args.trainsamples, args.batch_size):
        batch_inputs = X_train[i:i + args.batch_size]
        batch_labels = y_train[i:i + args.batch_size]

        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        batch_outputs = net(batch_inputs)  #训练

        outputs.append(batch_outputs)
        my_loss = nn.functional.cross_entropy(batch_outputs, batch_labels) #计算Loss
        loss_each_epoch+=my_loss.item()
        my_loss.backward() # 反向传播
        opt.step()  # 更新参数

    outputs_cat=torch.cat(outputs) #之前的是[25, 35, 28, 28] 现在是25(trainsamples)x5(classes)

    pred_y = torch.argmax(outputs_cat, dim=1)

    train_correct = (pred_y == y_train).sum()

    print('epoch',epoch,'的准确率是',train_correct.item()/len(y_train))
    print('epoch',epoch,'的loss是',loss_each_epoch/(i/10+1))



print('test')
with torch.no_grad():
    net.eval()
    outputs = []
    loss_each_epoch = 0
    for i in range(0, classes * args.testsamples, args.batch_size):
        batch_inputs = X_test[i:i + args.batch_size]
        batch_labels = y_test[i:i + args.batch_size]
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_outputs = net(batch_inputs)  # 训练
        outputs.append(batch_outputs)
        my_loss = nn.functional.cross_entropy(batch_outputs, batch_labels)  # 计算Loss
        loss_each_epoch += my_loss.item()

    outputs_cat = torch.cat(outputs)

    pred_y = torch.argmax(outputs_cat, dim=1)

    test_correct = (pred_y == y_test).sum()

    print('测试集的准确率是', test_correct.item() / len(y_test))
    print('测试集的loss是', loss_each_epoch / (i / 10 + 1))









