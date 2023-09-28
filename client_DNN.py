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
from tqdm import tqdm
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
from sklearn.datasets import make_classification
from collections import Counter
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
global Round
Round = 0
global count
count = 0
global class_names
class_names =  ['FORCE_ERROR_ATTACK','MITM_UNALTERED','NORMAL','READ_ATTACK','RECOGNITION_ATTACK','REPLAY_ATTACK','RESPONSE_ATTACK','WRITE_ATTACK']
class_names =  ['FORCE_ERROR','MITM_UNALTERED','NORMAL','READ','RECOGNITION','REPLAY','RESPONSE','WRITE']
#CUDA_LAUNCH_BLOCKING=1
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
if torch.cuda.is_available():
    print("use GPU")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU") 
warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        
        nodes_layer_0 = 10
        nodes_layer_1 = 500
        nodes_layer_2 = 500
        nodes_layer_3 = 500
        nodes_layer_4 = 500
        nodes_layer_7 = 8
       
        # input has two features and
        self.layer1 = nn.Linear(nodes_layer_0,nodes_layer_1)
        self.layer2 = nn.Linear(nodes_layer_1,nodes_layer_2)
        self.layer3 = nn.Linear(nodes_layer_2,nodes_layer_3)
        self.layer4 = nn.Linear(nodes_layer_3,nodes_layer_4)
        self.layer5 = nn.Linear(nodes_layer_4,nodes_layer_7)
 
        
    #forward propagation    
    def forward(self,x):
        print(x.dtype)
        #output of layer 1       
        z1 = self.layer1(x)
        a1 = Fnc.relu(z1)
        # output of layer 2
        z2 = self.layer2(a1)
        a2 = Fnc.relu(z2)
        z3 = self.layer3(a2)
        a3 = Fnc.relu(z3)
        z4 = self.layer4(a3)
        a4 = Fnc.relu(z4)
        z5 = self.layer5(a4)
        return z5

def train(net, local_net, trainloader, epochs ,device: torch.device):
    """Train the model on the training set."""
    net.to(device)
    #check_alpha()
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        if epoch % 50 ==0 and epoch != 0 :
            print("epoch:",epoch)
        
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(images)
        
            loss = criterion(outputs, labels)
                
            loss.backward()
            
            optimizer.step()

def test(net, testloader, device: torch.device,flag):
    global Round
    if "before" in flag:
        Round = Round + 1
        print ("現在是第",Round,"Round")
    
    """Validate the model on the test set."""
    #criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    y_true = []
    y_pred = []
    accuracy = 0.0
    ave_loss = 0.0
    
    
    # Evaluate the network
    net.to(device)
    net.eval()
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()

            ave_loss = ave_loss * 0.9 + loss * 0.1
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    y_true = labels.data.cpu().numpy()
    y_pred = predicted.data.cpu().numpy()
    
    acc = classification_report(y_true, y_pred, digits = 4, output_dict = True)
    accuracy = correct / total
    print(flag,"  ",accuracy)

    global class_names
    a = ()
    z = ()
    #c = ()
    y = ()
    x = ()

    for i in range(len(class_names)):
        a = a + (acc[str(i)]['precision'],)
        try:
            z = z + (acc[str(i)]['recall'],)
        except:
            z = z + (-1,)    
        #c = c + (acc[str(i)]['f1-score'],)

    y = y + (ave_loss,)
    x = x + (accuracy,time.time()-start_IDS,)
    
    

    # b = b + (accuracy,ave_loss,time.time()-start_IDS)
    a = str(a)[1:-1]
    #c = str(c)[1:-1]
    z = str(z)[1:-1]
    y = str(y)[1:-1]
    x = str(x)[1:-1]
    if not flag == "none":
        with open("./TXT/recall_C{0}_{1}.csv".format(client,flag), "a+") as file:
            file.write(str(z))
            file.writelines("\n")
        with open("./TXT/accuracy_C{0}_{1}.csv".format(client,flag), "a+") as file:
            file.write(str(x))
            file.writelines("\n")
        with open("./TXT/loss_C{0}_{1}.csv".format(client,flag), "a+") as file:
            file.write(str(y))
            file.writelines("\n")
        with open("./TXT/precision_C{0}_{1}.csv".format(client,flag), "a+") as file:
            file.write(str(a))
            file.writelines("\n")
        """
        with open("./TXT/f1_C{0}_{1}.csv".format(client,flag), "a+") as file:
            file.write(str(c))
            file.writelines("\n")
        """

    report = DataFrame(classification_report(y_true, y_pred, digits = 4, output_dict = True)).transpose()
    #report.to_csv("./TXT/report_C{0}_{1}.csv".format(client,flag))
    # 開始繪製混淆矩陣並存檔
    cf_matrix = confusion_matrix(y_true, y_pred) 
    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                    # https://stackoverflow.com/a/53824126/13369757
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)     # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Image-Classification-using-PyTorch.html
    #print(cf_matrix)
    plt.figure(figsize = (18,12))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label")
    plt.savefig("./TXT/matrix__C{0}_{1}.png".format(client,flag))
    plt.close()
    
    return loss / len(testloader.dataset), correct / total

    #####################################################################################################提前停止程式#############################################################################################
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
"""
def load_data(x_train,y_train,x_test,y_test):
    Training_data = [(x, y) for x, y in zip(x_train,y_train)]
    Testing_data = [(x, y) for x, y in zip(x_test,y_test)]
    import random
    random.shuffle(Testing_data)

    Training_DataSet=Data_Loader(Training_data)
    Testing_DataSet=Data_Loader(Testing_data)

    # trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = DataLoader(dataset=Training_DataSet, batch_size=512, shuffle=True)
    testset = DataLoader(dataset=Testing_DataSet, batch_size=len(Testing_DataSet), shuffle=False)
    return trainset, testset
"""
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
"""
def optim():
    temp_net_weight = [val.cpu().numpy() for _, val in net.state_dict().items()]
    local_net_weight = [val.cpu().numpy() for _, val in local_net.state_dict().items()]
    result = []
    for i in range( len(temp_net_weight)):
        print(temp_net_weight[i])
        t = (temp_net_weight[i] + local_net_weight[i])/2
        result.append( t )
        #weights_prime = [ reduce(np.add, layer_updates) / 2 for layer_updates in zip(*result) ]
    #weighted_weights = [ [layer * num_examples for layer in weights] for weights, num_examples in results ]
    return result
"""
def optim():
    temp_net_weight = [ [ layer for layer in val.cpu().numpy()] for _, val in net.state_dict().items()]
    local_net_weight = [ [ layer for layer in val.cpu().numpy()] for _, val in local_net.state_dict().items()]
    return [ reduce(np.add, layer_updates) / 2 for layer_updates in zip(temp_net_weight,local_net_weight) ]
    
# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)

# net = MLP().to(DEVICE)
# trainloader, testloader = load_data()
maxEpochforIDS=100
try:
    maxEpochforIDS=sys.argv[1]
    maxEpochforIDS = int(maxEpochforIDS)
except:
    maxEpochforIDS
    
net = DNN().to(DEVICE)
local_net = DNN().to(DEVICE)

#target_state = nn.Linear(500,6).to(DEVICE)
#original_state = net.layer5
#net.layer5 = target_state
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

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
local_net.eval()
save_state = {}
print("Model's state_dict:")
for param_tensor in net.state_dict():
    freeze = False
    for keyword in strategy:
        if keyword in param_tensor:
            freeze = True
            continue
    if not freeze:        
        save_state.update({param_tensor:net.state_dict()[param_tensor]})
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())

from imblearn.over_sampling import SMOTE
def smote_train_data(x_train,y_train,label):
    #smo = SMOTE( sampling_strategy ={label: generate_num },random_state=42)
    smo = SMOTE( sampling_strategy ='auto',random_state=42)
    #print(Counter(y_train))
    X_smo, y_smo = smo.fit_resample(x_train, y_train)
    #print(Counter(y_smo))
    return X_smo, y_smo

#for name, para in net.named_parameters():
#   print('{}: {}'.format(name, para.shape))

start = time.time()
client = 1
try:
    client=sys.argv[3]
    client = int(client)
except:
    client

x_train = np.loadtxt('./electra_modbus/xtrain{0}t.txt'.format(client))
x_test = np.loadtxt('./electra_modbus/xtest{0}t.txt'.format(client))
y_train = np.loadtxt('./electra_modbus/ytrain{0}t.txt'.format(client))
y_test = np.loadtxt('./electra_modbus/ytest{0}t.txt'.format(client))

#x_smo,y_smo = smote_train_data(x_train,y_train,5)
#trainloader = load_data(x_smo,y_smo,512)
trainloader = load_data(x_train,y_train,512)
testloader = load_data(x_test,y_test,len(y_test),shuffle_flag = False)

import os
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # net.layer8 = nn.Linear(50,3)
        for param_tensor in net.state_dict():
            freeze = False
            for keyword in strategy:
                if keyword in param_tensor:
                    freeze = True
                    continue
            if not freeze:        
                save_state.update({param_tensor:net.state_dict()[param_tensor]})

        return [val.cpu().numpy() for _, val in save_state.items()]

    def set_parameters(self, parameters):
        # net.layer1 = nn.Linear(67,50)
        #net.layer5 = target_state
      
        # for name, param in net.named_parameters():
        #     print(name,param.requires_grad)
        params_dict = zip(save_state.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #print(state_dict)
        net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train(net, local_net, trainloader, epochs=maxEpochforIDS , device=DEVICE)

        test(net, testloader, device=DEVICE, flag = "-before-net")
        #test1(net, testloader, device=DEVICE,flag = "-before-net") 

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader, device=DEVICE, flag = "-after-net")
       
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="192.168.1.132:53388",
    client=FlowerClient(),
)
end = time.time()
print('花費的時間:',round((end - start)/60, 2),'分')