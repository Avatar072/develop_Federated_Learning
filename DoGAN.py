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
from mytoolfunction import SaveDataToCsvfile, generatefolder, mergeDataFrameToCsv

filepath = "D:\\Labtest20230911\\"
start_IDS = time.time()
# 檢查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #############################################################################
class Data_Loader(): 
    def __init__(self, data_list):       
        self.data = data_list

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
    n = Variable(torch.randn(size, 100)).to(device)
    return n

def ones_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.ones(size, 1)).to(device)
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1)).to(device)
    return data

def true_target(y):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.from_numpy(y).type(torch.FloatTensor)).to(device)
    return data

#### model generation for discriminator
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 78 #78並非單單是因為特徵是有78個，而是因為GeneratorNet會生成的78個特徵假資料會餵給DiscriminatorNet做判斷，兩者做矩陣相處要match，
        n_out = 1 #這邊的1是因為forward return實際的值是切0~1之間
        
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
        # print("DiscriminatorNet", x)
        return x

#### model generation for generator
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_noise = 100
        n_out = 78  # 要生產資料集特征的数量匹配的假資料
        
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
        x = self.hidden0(x)#實際層數要看這一層
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
    y_real = y_real.float().to(real_data.device)  # Convert y_real to Float data type
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data) #real_data為1
    # Calculate error and backpropagate
    # error_real = loss(prediction_real, true_target(y_real))
    
    # e.g:prediction_real實際答案有0有1，  y_real.view(-1, 1)解答 #辨識為1
    # 透過error_real藉由loss值去看出prediction_real實際答案會有0的
    # 然後由error_real去覆盤，告訴優惠器哪邊可以去做優化
    error_real = loss(prediction_real, y_real.view(-1, 1)) #辨識為1
    error_real.backward()
    
    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data) #fake_data為0
    # 目标的形状是在 y_real 变量中定义的
    y_real = torch.ones((N, 1))
    # Calculate error and backpropagate
    # e.g:prediction_fake實際答案有0有1，  zeros_target(N)為解答辨識為0
    # 透過error_real藉由loss值去看出prediction_fake實際答案會有1
    # 然後由error_real去覆盤，告訴優惠器哪邊可以去做優化

    error_fake = loss(prediction_fake, zeros_target(N)) #辨識為0
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
partial_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'total_encoded_updated.csv'))
test_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'test_dataframes.csv'))
# train_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'train_dataframes.csv'))
train_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'train_df_half1.csv'))

x_partial = np.array(partial_dataframe.iloc[:, :-1])
y_partial = np.array(partial_dataframe.iloc[:, -1])

#*************************************************
x_train = np.array(train_dataframe.iloc[:, :-1])
y_train = np.array(train_dataframe.iloc[:, -1])

x_test = np.array(test_dataframe.iloc[:, :-1])
y_test = np.array(test_dataframe.iloc[:, -1])

########################   Finding Weak labels ##############################
# numOfSamples = 50
numOfSamples = 50
recall_threshold = 0.94
weakpoint =8 

#####################********** GAN Parameters  **************##############
            
# x_g=np.copy(x_partial)
# y_g=np.zeros(y_partial.shape)

x_g=np.copy(x_train)
y_g=np.zeros(y_train.shape)

print("x_g",len(x_g))
print("y_g",len(y_g))
# 找到值為8的標籤的索引
indices_8 = [index for index, value in enumerate(y_train) if value == 8]
print(f"值为8的标签的索引: {indices_8}")
# 將值為8的標籤標記為“真實”或“正類別”
y_g[indices_8==weakpoint] = 1 #y_g[y_partial==weakpoint]=1 #等於1表示標記為“真實”或“正類別”特定類別（在這裡是 類別）標記為1，而其他類別標記為0


# 將其他標籤標記為0
y_g[~np.isin(np.arange(len(y_g)), indices_8)] = 0


data = [(x, y) for x, y in zip(x_g,y_g)]

print("data",len(data))
dataSet=Data_Loader(data)
data_loader = DataLoader(dataset=dataSet, batch_size=256, shuffle=True)
discriminator = DiscriminatorNet().to(device)
generator = GeneratorNet().to(device)

###########################################################
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()
###########################################################
num_test_samples = 20
test_noise = noise(num_test_samples)

#*********************************************  Running GAN   ***************************
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
num_epochs = 100
# d_errs=[]
# g_errs=[]
d_errs = torch.Tensor([])  # 初始化为空张量
g_errs = torch.Tensor([])  # 初始化为空张量

print("data_loader",len(data_loader))
for epoch in range(num_epochs):
    print("epoch: ",epoch)
    g_error_sum=0
    d_error_sum=0
    #第二個 for 循環主要是執行 GAN（生成對抗網絡）的訓練過程，包括訓練鑑別器（Discriminator）和生成器（Generator）以及生成新的數據
    for n_batch, (real_batch,y_real) in enumerate(data_loader):# 一次復盤完，就生一次資料去考試
        
        print("n_batch",n_batch)
        print("real_batch",len(real_batch))
        print("y_real",len(y_real))
        print("data_loader",len(data_loader))
        # 使用 n_batch 來知道正在處理的是第幾個批次
        # n_batch 是當前批次的索引 它從 0 開始遞增，直到 data_loader 中的所有批次都被處理
        # print(n_batch)
        #real_batch.size(0) 給出了當前批次中包含的樣本數量。
        #例如，如果 real_batch 是一個形狀為 (256, 100) 的張量，那麼 real_batch.size(0) 將返回 256，表示這個批次包含了 256 個樣本
        # 表示當前批次包含了多少筆數據
        N = real_batch.size(0)

        # 訓練鑑別器（Discriminator）：
        # 首先，將實際數據（真實圖像）設置為 real_data。
        # 接著，生成假數據（fake data）並使用 generator 生成器生成它們，同時使用 .detach() 函數將梯度從生成器的結果中分離出來，這是為了確保只訓練鑑別器，不訓練生成器。
        # 調用 train_discriminator 函數，該函數執行鑑別器的訓練，並計算鑑別器的損失（d_error）和相關預測（d_pred_real 和 d_pred_fake）。

        # 1. Train Discriminator
        real_data = Variable(real_batch).to(device)
        #real_target=Variable(real_target)
        # Generate fake data and detach 

        #print(real_data.shape)
        # generator透過 noise 產生假資料    
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach().to(device)

        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,y_real)

        # 2. Train Generator

        # Generate fake data
        fake_data = generator(noise(N)).to(device)
        # Train G
        # train_generator裡面會做覆盤，每次訓練持續告訴優化器哪邊還可以優化
        g_error = train_generator(g_optimizer, fake_data)

        g_error_sum+=g_error

        d_error_sum+=d_error
                
        d_errs = torch.cat((d_errs, torch.tensor([d_error_sum])))
        g_errs = torch.cat((g_errs, torch.tensor([g_error_sum])))



            # d_errs.append(d_error_sum)
            # g_errs.append(g_error_sum)  

            # if (epoch) % 10 == 0:
            #     print("epoch: ",epoch)
            #     test_noise = noise(num_test_samples)
            #     test_images =(generator(test_noise))
            #     test_images = test_images.data
            #     real_data=real_data.data


            #     plt.plot()
            #     plt.scatter(real_data[:,0][y_real==1], real_data[:,1][y_real==1], s=40, marker='x',c='red')
            #     plt.scatter(real_data[:,0][y_real==0], real_data[:,1][y_real==0], s=40, marker='o',c='blue')
            #     plt.scatter(test_images[:,0], test_images[:,1], s=40, marker='p',c='green')
            #     plt.axis('equal')
            #     # plt.show()
            #     plt.plot()
            #     plt.plot(d_errs.detach().numpy())
            #     plt.plot(g_errs.detach().numpy())
                # plt.show()
############  Generating and adding new samples  ###########
x_syn=(generator(noise(numOfSamples))).detach().to(device).cpu().numpy() # (generator) 生成的假特徵數據
y_syn=np.ones(numOfSamples)*weakpoint #這是與 x_syn 相關的標籤（labels）。在這裡，所有這些假數據的標籤都被設置為 weakpoint

print("Number of samples generated in current batch:", numOfSamples)
#************************************************
x_train=np.concatenate((x_train,x_syn),axis=0)
y_train=np.concatenate((y_train,y_syn),axis=0)
# mergeDataFrameToCsv(x_train,y_train)
df_x_train = pd.DataFrame(x_train)
df_y_train = pd.DataFrame(y_train)

# 使用concat函数将它们合并
combined_data = pd.concat([df_x_train, df_y_train], axis=1)
# 保存合并后的DataFrame为CSV文件
combined_data.to_csv(f'combined_data_08.csv', index=False)

# mergeDataFrameToCsv(df_x_train,df_y_train,weakpoint)
print("Shapes: ")#顯示 (資料筆數, 特徵數)。
print(x_train.shape,y_train.shape)#顯示 (資料筆數,)，因為這是一維的標籤數組