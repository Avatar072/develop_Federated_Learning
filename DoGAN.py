import warnings
import os
import time
import argparse
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
from torch.autograd import Variable
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
# 引用 datasetsPreprocess.py 中的函數
from mytoolfunction import SaveDataToCsvfile, generatefolder

filepath = "D:\\Labtest20230911\\"
start_IDS = time.time()
# #############################################################################
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

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

def ones_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    
    return data

def true_target(y):
    '''
    Tensor containing zeros, with shape = size
    '''
    data= Variable(torch.from_numpy(y).type(torch.FloatTensor))
    return data


#### model generation for discriminator
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 78
        n_out = 1
        
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 64),
            nn.Tanh()
            #nn.Dropout(0.3)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh()
            #nn.Dropout(0.3)
        )
        
        self.out = nn.Sequential(
            torch.nn.Linear(32, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        #print("Discriminator", x)
        x = self.hidden0(x)
        #print("Discriminator", x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.out(x)
        return x

#### model generation for generator
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_noise = 100
        n_out = 78  # 与特征的数量匹配
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_noise, 64),
            nn.Tanh()
        )
        
        self.hidden1 = nn.Sequential(            
            nn.Linear(64, 32),
            nn.Tanh()
        )
        
        self.out = nn.Sequential(
            nn.Linear(32, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        
        #x = self.hidden2(x)
        #print("Generator", x)
        
        x = self.out(x)
        return x
    
# training discriminator 
def train_discriminator(optimizer, real_data, fake_data, y_real):
    
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    y_real = y_real.float()  # Convert y_real to Float data type
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    # error_real = loss(prediction_real, true_target(y_real))
    print("y_real",y_real)
    error_real = loss(prediction_real, y_real.view(-1, 1))
    error_real.backward()
    
    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # 目标的形状是在 y_real 变量中定义的
    y_real = torch.ones((N, 1))
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

# training generator
def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

#################  Partial Data  ##################
#total_encoded_updated是只取10000筆的dataframe
partial_dataframe = pd.read_csv('D:\\Labtest20230911\\data\\total_encoded_updated.csv')
train_dataframe = pd.read_csv('D:\\Labtest20230911\\data\\test_dataframes.csv')
test_dataframe = pd.read_csv('D:\\Labtest20230911\\data\\train_dataframes.csv')


y_partial=np.array(partial_dataframe.iloc[:,-1])
x_partial=np.array(partial_dataframe.iloc[:,:-1])
#*************************************************
x_train=np.array(train_dataframe.iloc[:,:-1])
x_test=np.array(test_dataframe.iloc[:,:-1])
y_train=np.array(train_dataframe.iloc[:,-1])
y_test=np.array(test_dataframe.iloc[:,-1])


########################   Finding Weak labels ##############################
numOfSamples=50
# 读取CSV文件
df_client1 = pd.read_csv("D:\\Labtest20230911\\single_AnalyseReportFolder\\baseline_report_client1.csv")
# 去除末3行
df_client1 = df_client1.iloc[:-3]
# 筛选"recall"小于0.95的行
recall_threshold = 0.94

# 創建一個存儲弱標籤的列表
weakpoints = []

# 使用迴圈遍歷 df_client1 中的每一行數據
for index, row in df_client1.iterrows():
    # 在這裡，index 是行索引，row 是一行的數據
    if row["recall"] < recall_threshold:
        weakpoint = int(row["Unnamed: 0"])
        weakpoints.append(weakpoint)  # 添加找到的弱標籤到列表中

        # 打印找到的所有弱標籤
        print("找到的弱標籤:", weakpoints)
    #####################********** GAN Parameters  **************##############
            
        x_g=np.copy(x_partial)
        y_g=np.zeros(y_partial.shape)
        y_g[y_partial==weakpoint]=1

        data = [(x, y) for x, y in zip(x_g,y_g)]
        dataSet=Data_Loader(data)
        data_loader = DataLoader(dataset=dataSet, batch_size=78, shuffle=True)
        discriminator = DiscriminatorNet()
        generator = GeneratorNet()

        ###########################################################
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

        loss = nn.BCELoss()
        ###########################################################
        num_test_samples = 20
        test_noise = noise(num_test_samples)

        #*********************************************  Running GAN   ***************************
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        num_epochs = 10
        # d_errs=[]
        # g_errs=[]
        d_errs = torch.tensor([])  # 初始化为空张量
        g_errs = torch.tensor([])  # 初始化为空张量

        print("data_loader:",data_loader)
        for epoch in range(num_epochs):
            g_error_sum=0
            d_error_sum=0
            for n_batch, (real_batch,y_real) in enumerate(data_loader):
                N = real_batch.size(0)

                # 1. Train Discriminator
                real_data = Variable(real_batch)
                #real_target=Variable(real_target)
                # Generate fake data and detach 

                #print(real_data.shape)

                # (so gradients are not calculated for generator)
                fake_data = generator(noise(N)).detach()

                # Train D
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,y_real)

                # 2. Train Generator

                # Generate fake data
                fake_data = generator(noise(N))
                # Train G
                g_error = train_generator(g_optimizer, fake_data)

                g_error_sum+=g_error

                d_error_sum+=d_error
                
                d_errs = torch.cat((d_errs, torch.tensor([d_error_sum])))
                g_errs = torch.cat((g_errs, torch.tensor([g_error_sum])))


            d_errs.append(d_error_sum)
            g_errs.append(g_error_sum)  

            if (epoch) % 10 == 0:
                print("epoch: ",epoch)
                test_noise = noise(num_test_samples)
                test_images =(generator(test_noise))
                test_images = test_images.data
                real_data=real_data.data


                plt.plot()
                plt.scatter(real_data[:,0][y_real==1], real_data[:,1][y_real==1], s=40, marker='x',c='red')
                plt.scatter(real_data[:,0][y_real==0], real_data[:,1][y_real==0], s=40, marker='o',c='blue')
                plt.scatter(test_images[:,0], test_images[:,1], s=40, marker='p',c='green')
                plt.axis('equal')
                plt.show()
                plt.plot()
                plt.plot(d_errs.detach().numpy())
                plt.plot(g_errs.detach().numpy())
                plt.show()

        ############  Generating and adding new samples  ###########
        x_syn=(generator(noise(numOfSamples))).detach().numpy() 
        y_syn=np.ones(numOfSamples)*weakpoint
        #************************************************
        x_train=np.concatenate((x_train,x_syn),axis=0)
        y_train=np.concatenate((y_train,y_syn),axis=0)
        print("Shapes: ")
        print(x_train.shape,y_train.shape)













