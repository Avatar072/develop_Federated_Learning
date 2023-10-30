import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")#https://blog.csdn.net/qq_43391414/article/details/120543028
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from mytoolfunction import SaveDataToCsvfile, generatefolder, mergeDataFrameAndSaveToCsv, ChooseLoadNpArray,ChooseTrainDatastes, ParseCommandLineArgs
from collections import Counter
####################################################################################################

filepath = "D:\\Labtest20230911\\"
start_IDS = time.time()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)

# python BaseLine.py --dataset train_half1 --epochs 100
# python BaseLine.py --dataset train_half2 --epochs 100
# python DoGAN.py --dataset train_half1
file, num_epochs = ParseCommandLineArgs(["dataset", "epochs"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")

# ChooseLoadNpArray function  return x_train、y_train 和 client_str
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file)

print(client_str)

# x_test = np.load(filepath + "x_test.npy", allow_pickle=True)
# y_test = np.load(filepath + "y_test.npy", allow_pickle=True)
x_test = np.load(filepath + "x_test_respilt.npy", allow_pickle=True)
y_test = np.load(filepath + "y_test_respilt.npy", allow_pickle=True)

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# 將測試數據移動到 GPU 上
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

# 定義你的神經網絡模型
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 15)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 定義訓練函數
def train(net, trainloader, epochs):
    print("訓練中")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)

    for epoch in range(epochs):
        print("epoch",epoch)
        net.train()# PyTorch 中的一個方法，模型切換為訓練模式
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

# 定義測試函數
def test(net, testloader, start_time, client_str,plot_confusion_matrix):
    # print("測試中")
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    ave_loss = 0.0  # 初始化 ave_loss

    net.eval()  #PyTorch 中的一個方法，用於將神經網絡模型設置為測試模式
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 計算滑動平均損失
            ave_loss = ave_loss * 0.9 + loss * 0.1

            # 將標籤和預測結果轉換為 NumPy 陣列
            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
        
            # 計算每個類別的召回率
            acc = classification_report(y_true, y_pred, digits=4, output_dict=True)
            accuracy = correct / total

            # 將每個類別的召回率寫入 "recall-baseline.csv" 檔案
            RecordRecall = ()
            RecordAccuracy = ()
            labelCount = 15
           
            for i in range(labelCount):
                RecordRecall = RecordRecall + (acc[str(i)]['recall'],)
                 
            RecordAccuracy = RecordAccuracy + (accuracy, time.time() - start_time,)
            RecordRecall = str(RecordRecall)[1:-1]

            # 標誌來跟踪是否已經添加了標題行
            header_written = False
            with open(f"./single_AnalyseReportFolder/recall-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(str(RecordRecall) + "\n")
        
            # 將總體準確度和其他信息寫入 "accuracy-baseline.csv" 檔案
            with open(f"./single_AnalyseReportFolder/accuracy-baseline_{client_str}.csv", "a+") as file:
                if not header_written:
                    # file.write("標籤," + ",".join([str(i) for i in range(labelCount)]) + "\n")
                    header_written = True
                file.write(f"精確度,時間\n")
                file.write(f"{accuracy},{time.time() - start_time}\n")

                # 生成分類報告
                GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(GenrateReport).transpose()
                report_df.to_csv(f"./single_AnalyseReportFolder/baseline_report_{client_str}.csv",header=True)

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
        df_cm = pd.DataFrame(arr, class_names, class_names)
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.title(client_str)
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.savefig(f"./single_AnalyseReportFolder/{client_str}_epochs_{num_epochs}_confusion_matrix.png")
        plt.show()

# 創建用於訓練和測試的數據加載器
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=500, shuffle=True)
testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# 初始化神經網絡模型
net = Net().to(DEVICE)

# 訓練模型
train(net, trainloader, epochs=num_epochs)

# 評估模型
test_accuracy = test(net, testloader, start_IDS, client_str,True)
print("測試數據量:\n", len(test_data))
print("訓練數據量:\n", len(train_data))
print(f"最終測試準確度: {test_accuracy:.4f}")
# print("Label 14 SMOTE", Counter(y_train))