import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 数据增强与预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(30),  # 随机旋转
    transforms.Resize((224, 224)),  # 将图像尺寸调整为224x224
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载数据集
train_dataset = ImageFolder(root=r'F:/python/exam/ChestXRay2017/chest_xray/train', transform=transform)
test_dataset = ImageFolder(root=r'F:/python/exam/ChestXRay2017/chest_xray/test', transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义PneumoniaResNet模型
class PneumoniaResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 使用预训练的ResNet18模型
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # 修改最后一层为二分类

    def forward(self, x):
        return self.resnet(x)


# 初始化模型
model = PneumoniaResNet()

# 选择交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


# 训练函数（添加准确率与损失的记录）
def train_model(model, train_loader, num_epochs=10):
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 清除之前的梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            scheduler.step()  # 更新学习率

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # 训练完后绘制损失与准确率曲线
    plot_training_curve(epoch_losses, epoch_accuracies)


# 绘制训练损失和准确率的曲线图
def plot_training_curve(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss', color='b')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Training Accuracy', color='g')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()


# 测试模型并计算评价指标
def evaluate_model(model, test_loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # 计算评价指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


# 训练模型
train_model(model, train_loader, num_epochs=10)

# 评估模型
evaluate_model(model, test_loader)

