import warnings
from collections import OrderedDict
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torch import Tensor
import torch.nn.functional as Fnc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
from pandas.core.frame import DataFrame
#############################################################################  variable  ###################
filepath = "D:\\Labtest20230911\\"
#############################################################################  variable  ###################
CUDA_LAUNCH_BLOCKING=1
global Round

Round = 0
start = time.time()
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
global c 
c = 0
warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA is available and there is a compatible GPU.")
else:
    print("CUDA is not available or no compatible GPU is found.")
    
def CR_mat(y_true,y_pred):
    import re
    from sklearn.metrics import classification_report
    report=classification_report(y_true, y_pred, digits=4)
    classification_list=[]
    lines=report.split("\n")[2:-5]
    for line in lines:
        line=re.sub(' +', ' ', line)
        words=line.split(" ")
        if len(words)<5:
            continue
        # print(words)
        classification_list.append(words[1:])
    classification_mat=np.array(classification_list).astype(np.float64)
    return classification_mat

class DNN(nn.Module):
   
    def __init__(self):

        super(DNN,self).__init__()
        
        nodes_layer_0 = 78
        nodes_layer_1 = 500
        nodes_layer_2 = 500
        nodes_layer_3 = 500
        nodes_layer_4 = 15
        # nodes_layer_6 = 15
       
        # input has two features and
        self.layer1 = nn.Linear(nodes_layer_0,nodes_layer_1)
        self.layer2 = nn.Linear(nodes_layer_1,nodes_layer_2)
        self.layer3 = nn.Linear(nodes_layer_2,nodes_layer_3)
        self.layer4 = nn.Linear(nodes_layer_3,nodes_layer_4)
        # self.layer5 = nn.Linear(nodes_layer_4,nodes_layer_6)
        
        
    #forward propagation    
    def forward(self,x):

        #output of layer 1
        z1 = self.layer1(x)
        a1 = Fnc.relu(z1)
        # output of layer 2
        z2 = self.layer2(a1)
        a2 = Fnc.relu(z2)
        z3 = self.layer3(a2)
        a3 = Fnc.relu(z3)
        z4 = self.layer4(a3)
        # a4 = Fnc.relu(z4)
        # result = self.layer5(a4)
        
        return z4
client = 1
try:
    client=sys.argv[3]
    client = int(client)
except:
    client
    
# 加载选择的数据集
if client == 1:
    x_train = np.load(filepath + "x_train_half1.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_half1.npy", allow_pickle=True)
    print("Training with x_train_half1")
elif client == 2:
    x_train = np.load(filepath + "x_train_half2.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_half2.npy", allow_pickle=True)
    print("Training with x_train_half2")

x_test = np.load(filepath + "x_test.npy", allow_pickle=True)
y_test = np.load(filepath + "y_test.npy", allow_pickle=True)  # Fixed variable name

cuda = True if torch.cuda.is_available() else False
if torch.cuda.is_available():
    print("use GPU")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)
    print(torch.cuda.get_device_name(0))  
class Data_Loader():
    
    def __init__(self,data_list):       
        self.data=data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index][0]
        img_tensor = Tensor(img).float()
        label = self.data[index][1]
        return (img_tensor, label)

#######################  Train and Test Data Set  ##########################

def load_data(x_data,y_data,batch,shuffle_flag = True):

    x_data = torch.from_numpy(x_data).type(torch.FloatTensor)
    y_data = torch.from_numpy(y_data).type(torch.LongTensor)

    data = [(x, y) for x, y in zip(x_data,y_data)]
    if shuffle_flag:   
        import random
        random.shuffle(data)
    DataSet=Data_Loader(data)
    #print(batch)
    tset = DataLoader(dataset= DataSet, batch_size=batch, shuffle= shuffle_flag )
    return tset

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)

# net = MLP().to(DEVICE)
maxEpochforIDS=100
try:
    maxEpochforIDS=sys.argv[1]
    maxEpochforIDS = int(maxEpochforIDS)
except:
    maxEpochforIDS

#target_state = nn.Linear(500,6).to(DEVICE)
#original_state = nn.Linear(500,3).to(DEVICE)
  
net = DNN().to(DEVICE)

strategy = []
try:
    strategy_num = sys.argv[2]
    
    strategy_num = int(strategy_num)
    if strategy_num > 0:
        #strategy.append("7")
        for i in range(2,strategy_num+1):
            #print(i)
            strategy.append(str(i))
except:
    strategy
print(strategy)

send_key=net.state_dict().keys()
remove_key = []
for keyword in strategy:
    for key in send_key:
        if keyword in key:
            remove_key.append(key)
send_key = {tempkey: value for tempkey, value in net.state_dict().items() if tempkey not in remove_key}

start_IDS = time.time()

# net= torch.nn.Sequential(*(list(net.children())[:]))
# FILE = "MonFriday.pth"
# net.load_state_dict(torch.load(FILE))
# net.layer1 = nn.Linear(67,50)
net.eval()
save_state = {}
print("Model's state_dict:")
"""for param_tensor in net.state_dict():
    freeze = False
    for keyword in strategy:
        if keyword in param_tensor:
            freeze = True
            continue
    if not freeze:        
        save_state.update({param_tensor:net.state_dict()[param_tensor]})
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
"""
#from imblearn.over_sampling import SMOTE
# from collections import Counter
# def smote_train_data(x_train,y_train,label=-1):
#     smo = SMOTE()
#     if label < 0:
#         smo = SMOTE( sampling_strategy ='auto',random_state=42)
#     else:
#         smo = SMOTE( sampling_strategy ={label: 717*6 },random_state=42)
#     #print(Counter(y_train))
#     X_smo, y_smo = smo.fit_resample(x_train, y_train)
#     #print(Counter(y_smo))
    
#     return X_smo, y_smo

#x_gan = np.loadtxt('./electra_modbus/gan100+smote400/xgan1.txt')
#y_gan = np.loadtxt('./electra_modbus/gan100+smote400/ygan1.txt')

# xtemp = []
# ytemp = []
# for i in range(len(y_train)):
#     xtemp.append(x_train[i])
#     ytemp.append(y_train[i])
# for i in range(len(y_gan)):
#     xtemp.append(x_gan[i])
#     ytemp.append(y_gan[i])

# x_train = np.asarray(xtemp)
# y_train = np.asarray(ytemp)

#x_smo,y_smo = smote_train_data(x_train,y_train,5)
trainloader = load_data(x_train,y_train,512)
#trainloader = load_data(x_smo,y_smo,512)
testloader = load_data(x_test,y_test,len(y_test),shuffle_flag = False)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
global class_names
class_names =  ['FORCE_ERROR_ATTACK','MITM_UNALTERED','NORMAL','READ_ATTACK','RECOGNITION_ATTACK','REPLAY_ATTACK','RESPONSE_ATTACK','WRITE_ATTACK','Reconn']
# class_names =  ['FORCE_ERROR','MITM_UNALTERED','NORMAL','READ','RECOGNITION','REPLAY','RESPONSE','WRITE']
# class_names =  ['Backdoor','CommInj','DoS','normal','Reconn']

# net.layer1 = nn.Linear(67,50)
net.to(DEVICE)
#net.train()
    
for epoch in range(maxEpochforIDS):
    total = 0
    #if epoch % 50 ==0 and epoch != 0 :
    print("epoch:",epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
         # zero the parameter gradients
        optimizer.zero_grad()
         # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
            
    # testing
    correct, total, loss = 0, 0, 0.0

    y_true = []
    y_pred = []
    accuracy = 0.0
    ave_loss = 0.0
    
    # Evaluate the network
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ave_loss = ave_loss * 0.9 + loss * 0.1

            y_true = labels.data.cpu().numpy()
            y_pred = predicted.data.cpu().numpy()
            
            acc = classification_report(y_true, y_pred, digits = 4, output_dict = True)
            accuracy = correct / total
            #print(accuracy)
            
            #global class_names

            a = ()
            b = ()
            c = ()
            d = ()
            e = ()
            # length = len(class_names)
            length = 15
            
            for i in range(length):
                a = a + (acc[str(i)]['precision'],)
#                try:
                b = b + (acc[str(i)]['recall'],)
 #               except:
  #              b = b + (-1,)    
                c = c + (acc[str(i)]['f1-score'],)

            d = d + (ave_loss,)
            e = e + (accuracy,time.time()-start_IDS,)

            #global recall_list
            recall_list = list(b)
            a = str(a)[1:-1]
            b = str(b)[1:-1]
            c = str(c)[1:-1]
            d = str(d)[1:-1]
            e = str(e)[1:-1]
            with open("./other_AnalyseReportFolder/precision-baseline_C{0}.csv".format(client), "a+") as file:
                file.write(str(a))
                file.writelines("\n")
            with open("./other_AnalyseReportFolder/recall-baseline_C{0}.csv".format(client), "a+") as file:
                file.write(str(b))
                file.writelines("\n")

            GenrateReport = classification_report(y_true, y_pred, digits=4, output_dict=True)
            # 将字典转换为 DataFrame 并转置
            report_df = pd.DataFrame(GenrateReport).transpose()
            # 保存为 baseline_report 文件 "這邊會存最後一次的資料"
            report_df.to_csv("./other_AnalyseReportFolder/baseline_report_C{0}.csv".format(client),header=True)

    print(torch.__version__)