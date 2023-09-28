import warnings
import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

filepath = "D:\\Labtest20230911\\"
start_IDS = time.time()
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 命令行参数解析器
parser = argparse.ArgumentParser(description='Federated Learning Client')

# 添加一个参数来选择数据集
parser.add_argument('--dataset', type=str, choices=['x_train_half1', 'x_train_half2'], default='x_train_half1',
                    help='Choose the dataset for training (x_train_half1 or x_train_half2)')

args = parser.parse_args()

# 根据命令行参数选择数据集
selected_dataset = args.dataset

# 加载选择的数据集
if selected_dataset == 'x_train_half1':
    x_train = np.load(filepath + "x_train_half1.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_half1.npy", allow_pickle=True)
    print("Training with x_train_half1")
elif selected_dataset == 'x_train_half2':
    x_train = np.load(filepath + "x_train_half2.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_half2.npy", allow_pickle=True)
    print("Training with x_train_half2")

x_test = np.load(filepath + "x_test.npy", allow_pickle=True)
y_test = np.load(filepath + "y_test.npy", allow_pickle=True)  # Fixed variable name

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

print("Minimum label value:", min(y_train))
print("Maximum label value:", max(y_train))
print(np.unique(y_train))
# 定义神经网络模型
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 15)
        #self.fc3 = nn.Linear(64, len(np.unique(y_train)))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义训练和评估函数
def train(net, trainloader, epochs):
    print("train")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

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
            with open("./my_AnalyseReportfolder/recall-baseline.csv", "a+") as file:
                # file.write(str(RecordRecall))
                # file.writelines("\n")
                # 添加标题行
                file.write("Label," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                # 写入Recall数据
                file.write("Recall," + RecordRecall + "\n")
        
            # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
            with open("./my_AnalyseReportfolder/accuracy-baseline.csv", "a+") as file:
                # file.write(str(RecordAccuracy))
                # file.writelines("\n")
                # 添加标题行
                file.write("Accuracy,Time\n")
                # 写入Accuracy数据
                file.write(str(RecordAccuracy) + "\n")

            # 生成分类报告
            GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # 将字典转换为 DataFrame 并转置
            report_df = pd.DataFrame(GenrateReport).transpose()
            # 保存为 baseline_report 文件
            report_df.to_csv("./my_AnalyseReportfolder/baseline_report.csv",header=True)
    accuracy = correct / total
    print("test_data:\n",len(test_data))
    print("train_data:\n",len(train_data))
    return accuracy

# 创建用于训练和测试的 DataLoader
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=20000, shuffle=True)  # 设置 shuffle 为 True
# test_data 的batch_size要設跟test_data(y_test)的筆數一樣 重要!!!
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
# #############################################################################
# 2. 使用 Flower 集成的代码
# #############################################################################

# 定义Flower客户端类
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = test(net, testloader)
        return accuracy, len(testloader.dataset), {"accuracy": accuracy}

# 初始化神经网络模型
net = Net().to(DEVICE)

# 启动Flower客户端
fl.client.start_numpy_client(
    server_address="127.0.0.1:53388",
    client=FlowerClient(),
    
)