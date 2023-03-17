import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
#加载数据集
batch_size = 10
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])

train_dataset = datasets.MNIST(root = './dataset/mnist', train = True, download = True, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.MNIST(root = './dataset/mnist/', train = False, download = True, transform = transform)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size)
#构建网络
class ConvolutionNet(torch.nn.Module):
    def __init__(self):
        super(ConvolutionNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10) 

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = ConvolutionNet()

#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
#构建训练和测试函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set:%d %% [%d/%d]' % (100 * correct / total, correct, total))
    return 100 * correct / total
if __name__ == '__main__':
    len = 10
    x = [0] * len
    y = [0] * len
    for epoch in range(10):
        train(epoch)
        y[epoch] = test()
        x[epoch] = epoch
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

