import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
# バージョン確認 (Google Colab default)
# print(torch.__version__)  # 1.4.0
# print(torchvision.__version__)  # 0.5.0
# print(np.__version__)  # 1.18.2
# print(matplotlib.__version__)  # 3.2.1
# print(Image.__version__)  # 7.0.0

# データの読み込み
transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
data_set = {
  x: torchvision.datasets.STL10(root='./images', split=x, download=True, transform=transform)
  for x in ('train', 'test')
}
data_size = {
  x: len(data_set[x]) for x in ('train', 'test')
}
print(data_size)

data_loaders = {
  x[0]: torch.utils.data.DataLoader(data_set[x[0]], batch_size=4, shuffle=x[1], num_workers=1)
  for x in (('train', True), ('test', False))
}

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

# モデル構築
from torchvision import models
from torch import nn

net = models.vgg16(pretrained=True)
last_in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(last_in_features, 10)  # 最終層を10クラスに変更
print(net)

# 損失・最適関数の定義
from torch import optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 10エポックごとに学習率が1/10に更新

num_epochs = 2
def train_net(net, criterion, optimizer, scheduler, num_epochs):
  for epoch in range(num_epochs):
    print('Epoch %d/%d' % (epoch, num_epochs - 1))
    print('-' * 10)

    running_loss_sum = 0
    train_loss_sum = 0
    test_loss_sum = 0

    # モデルの更新
    net.train()
    for i, (inputs, labels) in enumerate(data_loaders['train']):
      # 勾配の初期化
      optimizer.zero_grad()
      # 予測
      outputs = net(inputs)
      # 損失の導出
      loss = criterion(outputs, labels)
      # 逆伝播
      loss.backward()
      # 勾配の更新
      optimizer.step()

      running_loss_sum += loss.item()
      train_loss_sum += loss.item()
      if i % 10 == 9:
        print('[%d] running_loss: %.3f' % (i + 1, running_loss_sum / 10))
        running_loss_sum = 0
    
    # モデルの評価
    net.eval()
    for i, (inputs, labels) in enumerate(data_loaders['test']):
      # 勾配の初期化
      optimizer.zero_grad()
      # 予測
      outputs = net(inputs)
      # 損失の導出
      loss = criterion(outputs, labels)

      test_loss_sum += loss.item()

    print('')
    print('train_loss_ave:\t%.3f' % (train_loss_sum / data_size['train']))
    print('test_loss_ave:\t%.3f' % (test_loss_sum / data_size['test']))

    net.train()
    scheduler.step()

train_net(net, criterion, optimizer, scheduler, num_epochs)

# 保存
PATH = './stl_net.pth'
torch.save(net.state_dict(), PATH)