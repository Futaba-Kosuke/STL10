import torch
import torchvision
from torchvision import transforms

# GPUの確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データの読み込み
transform = transforms.Compose(
  [transforms.Resize((300, 300)),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
data_set = torchvision.datasets.ImageFolder(
  root='./images',
  transform=transform
)
data_loader = torch.utils.data.DataLoader(
  data_set,
  batch_size=1,
  shuffle=False,
  num_workers=1
)

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

# モデルの読み込み
# アンサンブル学習を行う為、複数のモデルを読み込む
from torch import nn
from torchvision import models

vgg16 = models.vgg16()
last_in_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(last_in_features, 10)
vgg16 = vgg16.to(device)
state_dict = torch.load('./models/vgg16.pth', map_location=device)
vgg16.load_state_dict(state_dict)

densenet = models.densenet161()
last_in_features = densenet.classifier.in_features
densenet.classifier = nn.Linear(last_in_features, 10)
densenet = densenet.to(device)
state_dict = torch.load('./models/densenet.pth', map_location=device)
densenet.load_state_dict(state_dict)

wide_resnet = models.wide_resnet50_2()
last_in_features = wide_resnet.fc.in_features
wide_resnet.fc = nn.Linear(last_in_features, 10)
wide_resnet = wide_resnet.to(device)
state_dict = torch.load('./models/wide_resnet.pth', map_location=device)
wide_resnet.load_state_dict(state_dict)

nets = {
    'vgg16': vgg16,
    'densenet': densenet,
    'wide_resnet': wide_resnet
}
net_names = ('vgg16', 'densenet', 'wide_resnet')

# 検知開始
for net_name in net_names:
  nets[net_name].eval()

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
  for (inputs, label) in data_loader:
    inputs = inputs.to(device)
    label = label.to(device)

    outputs = torch.zeros([1, 10])
    outputs = outputs.to(device)
    # アンサンブル学習
    for net_name in net_names:
      outputs += nets[net_name](inputs)

    _, predicted = torch.max(outputs, 1)
    is_correct = (predicted == label)

    # 各クラスの枚数をカウント
    class_total[label] += 1
    if is_correct:
      # 各クラスの正解数をカウント
      class_correct[label] += 1

# 正解率の出力
for i in range(10):
  print('Accuracy of %5s : %02.01f %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]))
  
print('\nTop-1 Accuracy: %02.01f %%' % (100 * sum(class_correct) / sum(class_total)))