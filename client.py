import warnings
import os
import seaborn as sns
import time
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
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
from sklearn.metrics import confusion_matrix
from mytoolfunction import generatefolder, ChooseLoadNpArray,ParseCommandLineArgs,ChooseTrainDatastes
from collections import Counter

filepath = "D:\\Labtest20230911\\"
start_IDS = time.time()
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#python client.py --dataset train_half1 --epochs 500 --method normal
#python client.py --dataset train_half2 --epochs 500 --method normal
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)

counter = Counter(y_train)
print(counter)
today = datetime.date.today()
today = today.strftime("%Y%m%d")
generatefolder(filepath, "\\FL_AnalyseReportfolder")
generatefolder(f"./FL_AnalyseReportfolder/", today)
generatefolder(f"./FL_AnalyseReportfolder/{today}/", client_str)
generatefolder(f"./FL_AnalyseReportfolder/{today}/{client_str}/", Choose_method)

x_test = np.load(filepath + "x_test_20231113.npy", allow_pickle=True)
y_test = np.load(filepath + "y_test_20231113.npy", allow_pickle=True)  # Fixed variable name

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# 将测试数据移动到GPU上
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

print("Minimum label value:", min(y_train))
print("Maximum label value:", max(y_train))




# def model using nn model ，和神經元使用512
# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(x_train.shape[1], 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, 15)
#         #self.fc3 = nn.Linear(64, len(np.unique(y_train)))

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         #實際矩陣相乘部分
#         # x = x.to(torch.float32)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(x_train.shape[1], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.layer5 = nn.Linear(512, 15)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.layer5(x)
        return x
    
# 定义训练和评估函数
def train(net, trainloader, epochs):
    print("train")
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)

    for epoch in range(epochs):
        print("epoch",epoch)
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(images)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            ###訓練的過程    
        test_accuracy = test(net, testloader, start_IDS, client_str,False)
        print(f"訓練週期 [{epoch+1}/{epochs}] - 測試準確度: {test_accuracy:.4f}")

def test(net, testloader, start_time, client_str,plot_confusion_matrix):
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
            #print("acc:\n",acc)
            # 将每个类别的召回率写入 "recall-baseline.csv" 文件
            # RecordRecall是用来存储每个类别的召回率（recall）值的元组
            # RecordAccuracy是用来存储其他一些数据的元组，包括整体的准确率（accuracy）
            #RecordRecall = []
            RecordRecall = ()
            RecordAccuracy = ()
            labelCount = 15
            # labelCount = len(np.unique(y_train))# label數量
            # print("labelCount:\n",labelCount)

            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                #RecordRecall.append(acc[str(i)]['recall'])    
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_time,)
          

            RecordRecall = str(RecordRecall)[1:-1]

            # 标志来跟踪是否已经添加了标题行
            header_written = False
            with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/recall-baseline_{client_str}.csv", "a+") as file:
                # file.write(str(RecordRecall))
                # file.writelines("\n")
                # 添加标题行
                #file.write("Label," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                # 写入Recall数据
                # file.write(f"{client_str}_Recall," + str(RecordRecall) + "\n")
                file.write(str(RecordRecall) + "\n")
        
            # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
            with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-baseline_{client_str}.csv", "a+") as file:
                # file.write(str(RecordAccuracy))
                # file.writelines("\n")
                # 添加标题行
                file.write(f"{client_str}_Accuracy,Time\n")
                # 写入Accuracy数据
                file.write(str(RecordAccuracy) + "\n")

            # 生成分类报告
            GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # 将字典转换为 DataFrame 并转置
            report_df = pd.DataFrame(GenrateReport).transpose()
            # 保存为 baseline_report 文件
            report_df.to_csv(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/baseline_report_{client_str}.csv",header=True)
    draw_confusion_matrix(y_true, y_pred,plot_confusion_matrix)
    accuracy = correct / total
    print(f"測試準確度: {accuracy:.4f}")
    return accuracy

# 畫混淆矩陣
def draw_confusion_matrix(y_true, y_pred, plot_confusion_matrix = False):
    #混淆矩陣
    if plot_confusion_matrix:
        # df_cm的PD.DataFrame 接受三個參數：
        # arr：混淆矩陣的數據，這是一個二維陣列，其中包含了模型的預測和實際標籤之間的關係，以及它們在混淆矩陣中的計數。
        # class_names：類別標籤的清單，通常是一個包含每個類別名稱的字串清單。這將用作 Pandas 資料幀的行索引和列索引，以標識混淆矩陣中每個類別的位置。
        # class_names：同樣的類別標籤的清單，它作為列索引的標籤，這是可選的，如果不提供這個參數，將使用行索引的標籤作為列索引
        arr = confusion_matrix(y_true, y_pred)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
        # class_names = ['0', '1', '2', '3']
        df_cm = pd.DataFrame(arr, class_names, class_names)
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.title(client_str +"_"+ Choose_method)
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/{client_str}_epochs_{num_epochs}_confusion_matrix.png")
        # plt.show()

# 创建用于训练和测试的 DataLoader
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=500, shuffle=True)  # 设置 shuffle 为 True
# test_data 的batch_size要設跟test_data(y_test)的筆數一樣 重要!!!
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
# #############################################################################
# 2. 使用 Flower 集成的代码
# #############################################################################

# 定义Flower客户端类
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):#是知識載入
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=num_epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}#step1上傳給權重，#step2在server做聚合，step3往下傳給server

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)#更新現有的知識#step4 更新model
        accuracy = test(net, testloader, start_IDS, client_str,True)
        return accuracy, len(testloader.dataset), {"accuracy": accuracy}

# 初始化神经网络模型
# net = Net().to(DEVICE)
net = MLP().to(DEVICE)

# 启动Flower客户端
fl.client.start_numpy_client(
    server_address="127.0.0.1:53388",
    client=FlowerClient(),
    
)