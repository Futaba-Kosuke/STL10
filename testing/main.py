import torch
import torchvision
from torchvision import transforms

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

# for i in range(len(data_set)):
#   print(data_set.__getitem__(i)[0].shape, data_set.__getitem__(i)[1])

data_loader = torch.utils.data.DataLoader(
  data_set,
  batch_size=1,
  shuffle=False,
  num_workers=1
)

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

# モデルの読み込み
from torch import nn
from torchvision import models

net = models.vgg16()
last_in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(last_in_features, 10)
model = torch.load('./models/vgg16_net_gpu.pth', map_location='cpu')
# model = torch.load('./models/vgg16_net_gpu.pth')

# net = models.densenet161()
# last_in_features = net.classifier.in_features
# net.classifier = nn.Linear(last_in_features, 10)
# model = torch.load('./models/dense_net_gpu.pth', map_location='cpu')
# model = torch.load('./models/vgg16_net_gpu.pth')

# net = models.wide_resnet50_2()
# last_in_features = net.fc.in_features
# net.fc = nn.Linear(last_in_features, 10)
# model = torch.load('./models/wide_resnet_gpu.pth', map_location='cpu')
# model = torch.load('./models/vgg16_net_gpu.pth')

net.load_state_dict(model)

# 検知開始
net.eval()

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
  for (inputs, label) in data_loader:

    outputs = net(inputs)
    
    _, predicted = torch.max(outputs, 1)
    is_correct = (predicted == label)

    class_total[label] += 1
    if is_correct:
      class_correct[label] += 1

for i in range(10):
  print('Accuracy of %5s : %02.01f %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]))
  
print('\nAccuracy Top-1: %02.01f %%' % (100 * sum(class_correct) / sum(class_total)))