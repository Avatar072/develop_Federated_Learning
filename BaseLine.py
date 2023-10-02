import warnings
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

filepath = "D:\\Labtest20230911\\"
start_IDS = time.time()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 命令行参数解析器
parser = argparse.ArgumentParser(description='Federated Learning Client')

# 添加一个参数来选择数据集
parser.add_argument('--dataset', type=str, choices=['train_half1', 'train_half2'], default='train_half1',
                    help='Choose the dataset for training (train_half1 or train_half2)')

args = parser.parse_args()

# 根据命令行参数选择数据集
my_command = args.dataset
# python BaseLine.py --dataset train_half1
# python BaseLine.py --dataset train_half2

# 加载选择的数据集
if my_command == 'train_half1':
    x_train = np.load(filepath + "x_train_half1.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_half1.npy", allow_pickle=True)
    client_str = "client1"
    print("Training with train_half1")
elif my_command == 'train_half2':
    x_train = np.load(filepath + "x_train_half2.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_half2.npy", allow_pickle=True)
    client_str = "client2"
    print("Training with train_half2")

x_test = np.load(filepath + "x_test.npy", allow_pickle=True)
y_test = np.load(filepath + "y_test.npy", allow_pickle=True)  # Fixed variable name

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# Define your neural network model
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 15)
        #self.fc3 = nn.Linear(64, len(np.unique(y_train)))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        #實際矩陣相乘部分
        # x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    # ... Define your model architecture ...

# Define the training and testing functions (similar to your code)
def train(net, trainloader, epochs):
    # ... Your training logic ...
    print("train")
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()#類似覆盤回推結果會甚麼會差，
            optimizer.step()#使用優化器優化

def test(net, testloader):
    print("test")
    correct = 0
    total = 0
    loss = 0  # 初始化损失值为0
    ave_loss = 0
    # with torch.no_grad():
    #     for images, labels in tqdm(testloader):
    #         output = net(images)
    #         _, predicted = torch.max(output.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # 迭代测试数据集
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
            # 使用神经网络模型进行前向传播
            outputs = net(images)
        
            # 计算损失
            loss += criterion(outputs, labels).item()
        
         # 计算预测的类别
            _, predicted = torch.max(outputs.data, 1)
        
            # 统计总样本数和正确分类的样本数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            # 计算滑动平均损失
            ave_loss = ave_loss * 0.9 + loss * 0.1

            # 将标签和预测结果转换为 NumPy 数组
            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            # 计算每个类别的召回率
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # print("correct:\n",correct)
            # print("total:\n",total)
            accuracy = correct / total
            print("acc:\n",acc)
            # 将每个类别的召回率写入 "recall-baseline.csv" 文件
            # RecordRecall是用来存储每个类别的召回率（recall）值的元组
            # RecordAccuracy是用来存储其他一些数据的元组，包括整体的准确率（accuracy）
            #RecordRecall = []
            RecordRecall = ()
            RecordAccuracy = ()
            
            labelCount = len(np.unique(y_train))# label數量
            print("labelCount:\n",labelCount)
           
            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                #RecordRecall.append(acc[str(i)]['recall'])    
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_IDS,)
          

            RecordRecall = str(RecordRecall)[1:-1]

            # 标志来跟踪是否已经添加了标题行
            header_written = False
            with open(f"./single_AnalyseReportFolder/recall-baseline_{client_str}.csv", "a+") as file:
                # file.write(str(RecordRecall))
                # file.writelines("\n")
                # 添加标题行
                #file.write("Label," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                # 写入Recall数据
                file.write(f"Recall," + RecordRecall + "\n")
        
            # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
            with open(f"./single_AnalyseReportFolder/accuracy-baseline_{client_str}.csv", "a+") as file:
                # file.write(str(RecordAccuracy))
                # file.writelines("\n")
                # 添加标题行
                file.write(f"Accuracy,Time\n")
                # 写入Accuracy数据
                file.write(str(RecordAccuracy) + "\n")

            # 生成分类报告
            GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # 将字典转换为 DataFrame 并转置
            report_df = pd.DataFrame(GenrateReport).transpose()
            # 保存为 baseline_report 文件 "這邊會存最後一次的資料"
            report_df.to_csv(f"./single_AnalyseReportFolder/baseline_report_{client_str}.csv",header=True)
    accuracy = correct / total
    print("test_data:\n",len(test_data))
    print("train_data:\n",len(train_data))
    return accuracy

    # ... Your testing logic ...

# Create data loaders for training and testing
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
# trainloader = DataLoader(train_data, batch_size=20000, shuffle=True)
# trainloader = DataLoader(train_data, batch_size=2000, shuffle=True)
trainloader = DataLoader(train_data, batch_size=1000, shuffle=True)
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# Initialize the neural network model
net = Net().to(DEVICE)

# Train the model
train(net, trainloader, epochs=10)

# Evaluate the model
accuracy = test(net, testloader)
print("Test Accuracy:", accuracy)