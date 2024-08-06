import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt

train_dataset = datasets.MNIST(root = 'data/', train = True, 
                               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'data/', train = False, 
                               transform = transforms.ToTensor(), download = True)

# ipdb.set_trace()

train_loader = DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size= 100, shuffle = True)

# # 简单查看一下数据集
# images, labels = next(iter(test_loader))
# img = torchvision.utils.make_grid(images, nrow = 10)
# img = img.numpy().transpose(1, 2, 0)
# print(type(test_loader),len(test_loader.dataset),len(test_loader))
# print(images.shape)
# print(img.shape)
# print(labels)
# # 归一化处理（可选，根据你的数据集是否已经归一化）
# # img = img * 0.5 + 0.5
# # 使用 matplotlib 显示并保存图片
# plt.imshow(img)
# plt.axis('on')  # 显示坐标轴
# plt.savefig('mnist_grid.png')  # 保存图片
# plt.show()  # 显示图片
# # cv2.imshow('img', img)
# # cv2.waitKey(0)

# 定义模型

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# 查看参数
# num_parameters = len(list(model.parameters()))
# print(f"Number of parameter tensors in the model: {num_parameters}")
# for name, param in model.named_parameters():
#     print(f"Name: {name}, Shape: {param.shape}")

# 设置优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 设置学习率调度器
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
# 设置loss计算方式
compute_loss = nn.CrossEntropyLoss()


for epoch in range(10):
    model.train()

    with tqdm(train_loader) as process_bar:
        for i, (images, labels) in enumerate(process_bar):

            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            pred = model(images)
            loss = compute_loss(pred, labels)
            loss.backward()
            optimizer.step()
            
            train_acc = (pred.argmax(dim=1) == labels).float().mean()
            process_bar.set_postfix(loss=loss.item(), 
                                    accuracy=train_acc.item(), 
                                    lr=optimizer.param_groups[0]['lr'])
            
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            test_loss += compute_loss(pred, labels)
            test_acc += (pred.argmax(dim=1) == labels).float().mean().item()
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        scheduler.step()
        # 打印当前epoch的验证损失和准确率
        print(f"Epoch {epoch + 1}, Val Loss: {test_loss}, Val Accuracy: {test_acc}")
