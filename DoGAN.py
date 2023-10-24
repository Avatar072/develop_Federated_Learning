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
from sklearn.decomposition import PCA
# 引用 datasetsPreprocess.py 中的函數
from mytoolfunction import SaveDataToCsvfile, generatefolder, mergeDataFrameAndSaveToCsv, ChooseTrainDatastes, ParseCommandLineArgs
from mytoolfunction import getStartorEndtime, CalculateTime

filepath = "D:\\Labtest20230911\\"
# 檢查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################## Choose Dataset ##############################
# 根据命令行参数选择数据集
# python DoGAN.py --dataset train_half1
# python DoGAN.py --dataset train_half1 --epochs 10000 --weaklabel 13
file, num_epochs, weakLabel= ParseCommandLineArgs(["dataset", "epochs","weaklabel"])
print(f"Dataset: {file}")
# num_epochs = 1
print(f"Number of epochs: {num_epochs}")
print(f"weakLabel code: {weakLabel}")
# weakpoint = weakLabel


# call ChooseTrainDatastes function，返回 x_train、y_train 和 client_str
x_train, y_train, client_str = ChooseTrainDatastes(filepath, file)
print(client_str)

########################  Weak labels and numOfSamples ##############################
numOfSamples = 1500
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
    #這個函數的作用是生成一個形狀為 （size， 100） 的張量（Tensor），其中size是指定生成隨機雜訊的批次大小（batch size），
    #而100是雜訊向量的維度。這個函數使用torch.randn（size， 100） 
    #生成服從標準正態分佈（均值為0，標準差為1）的隨機值
    n = Variable(torch.randn(size, 100)).to(device)
    return n

def ones_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    # ones_target 函數是用於創建一個張量（Tensor），其中所有元素的值都為1，並且具有指定的形狀 size。 
    # 這個函數的作用是生成目標Label，通常在訓練生成對抗網路（GAN）中用於表示DiscriminatorNet的目標。 
    # 在這裡，目標是讓DiscriminatorNet相信輸入的數據是真實數據
    data = Variable(torch.ones(size, 1)).to(device)
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    #zeros_target（size） 用於創建一個指定大小的零張量（Tensor），形狀是 （size， 1）。
    #這個函數將創建一個大小為size的一維張量，每個元素的值都是0，形狀為 （size， 1）。
    #目標是將真實數據標記為1（真實數據）和將Generator生成的假數據標記為0（假數據）。 
    #因此，在訓練DiscriminatorNet時，我們將使用zeros_target函數生成與批次中的假數據相對應的目標Label，其中所有標籤的值都是0，表示這些數據是假的。
    #zeros_target 函數生成的張量，每個元素都是標量值0。 這意味著生成的張量是一維的，包含多個元素，每個元素都設置為0。 
    #如果生成的張量大小為 （n，），那麼其中的每個元素都是0，共有 n 個0。
    #例如，如果size參數傳遞給 zeros_target 函數為256，那麼生成的張量將包含256個0，其中每個元素都為0。
    data = Variable(torch.zeros(size, 1)).to(device)
    return data

def true_target(y):
    '''
    Tensor containing zeros, with shape = size
    '''
    #目的是將一個 NumPy 數位列（y）轉換為 PyTorch Tensor（data），並確保該Tensor的數據類型是浮點數（FloatTensor）。 
    #這通常用於在訓練神經網路時，將標籤數據從 NumPy 陣列轉換為 PyTorch Tensor，以便進行後續計算。
    #.to（device）將PyTorch Tensor移動到指定的設備（device）。 這通常用於將PyTorch Tensor放在 GPU 上進行計算
    data = Variable(torch.from_numpy(y).type(torch.FloatTensor)).to(device)
    # retrun 型態為PyTorch Tensor的data
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
            nn.Linear(n_features, 512),
            nn.Tanh()
            #nn.Dropout(0.3)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh()
            #nn.Dropout(0.3)
        )
        
        self.out = nn.Sequential(
            torch.nn.Linear(512, n_out),
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
    ### 增加一層
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_noise = 100
        n_out = 78  # 要生產資料集特征的数量匹配的假資料
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_noise, 512),
            # nn.Tanh()
            nn.ReLU() #改用ReLU激活函數試
        )
        
        self.hidden1 = nn.Sequential(            
            nn.Linear(512, 512),
            # nn.Tanh()
            nn.ReLU() #改用ReLU激活函數試
        )
        
        self.hidden2 = nn.Sequential(            
            nn.Linear(512, 512),
            # nn.Tanh()
            nn.ReLU() #改用ReLU激活函數試
        )
        self.out = nn.Sequential(
            nn.Linear(512, n_out),
            # nn.Tanh()
            nn.ReLU() #改用ReLU激活函數試
        )

    def forward(self, x):
        x = self.hidden0(x)#實際層數要看這一層
        x = self.hidden1(x)
        
        x = self.hidden2(x)
        #print("Generator", x)
        
        x = self.out(x)
        return x

# training discriminator 
def train_discriminator(optimizer, real_data, fake_data, y_real):
    # discriminator任務盡可能準確地將真實數據與生成數據區分開來

    N = real_data.size(0) # 返回張量 real_data 第一個維度的大小，通常是批次大小（batch size），表示在這一批次中有多少樣本
    # Reset gradients
    optimizer.zero_grad() #optimizer是用於更新神經網路權重的優化器，zero_grad方法用於將所有模型參數（權重和偏置）的梯度歸零，以確保在每個批次的反向傳播之前沒有任何殘留的梯度資訊。
    y_real = y_real.float().to(real_data.device)  # Convert y_real to Float data type
    
    # 1.1 Train on Real Data
    # prediction_real 包含了 discriminator對真實數據的預測結果，這些預測結果是介於 0 到 1 之間的值，表示輸入數據是真實數據的概率。 
    # 例如，如果 prediction_real 的某個元素接近 1，那麼discriminator認為對應的輸入數據非常可能是真實數據(real)，而如果接近 0，則認為是生成數據(fake)。
    # 這個操作是鑒別器在訓練過程中的一部分，它幫助鑒別器學習如何對輸入的數據進行分類，並衡量輸入數據是真實數據的概率。 
    # 隨後，這些預測結果將用於計算損失（通常是二元交叉熵損失），以便優化鑒別器的參數，使其能夠更好地區分真實數據和生成數據。
    prediction_real = discriminator(real_data) #real_data為1

    ####################################################################    remind  prediction_real and y_real size ####################################################################
    # print("prediction_real shape",prediction_real.size())#prediction_real shape torch.Size([256, 1])
    # print("y_real shape",y_real.size())#y_real shape torch.Size([256])
    #target y_real 的size要跟input prediction_real的size match不然會出現下列報錯
    #Please ensure they have the same size.".format(target.size(), input.size())
    #ValueError: Using a target size (torch.Size([256])) that is different to the input size (torch.Size([256, 1])) is deprecated. Please ensure they have the same size.
    # prediction_real 的形狀是 （256， 1），而 y_real 的形狀也是 （256），它們的大小確實不匹配。 這正是出現錯誤的原因。
    # 將 y_real 從 （256） 調整為 （256， 1），確保了它的形狀與 prediction_real 的形狀匹配
    #例如，如果原始張量 y_real 的形狀是 （256），那麼在調用 .view（-1， 1） 后，它的形狀將變為 （256， 1），這是一個包含 256 行和 1 列的二維張量
    y_real = y_real.view(-1, 1)
    ####################################################################    remind  prediction_real and y_real size ####################################################################
   
    ####################################################################    轉換成 NumPy array ####################################################################
    # 將 y_real 從Tensor 轉換成 NumPy array 因y_real要帶入true_target()要是np.ndarray不能是Tensor
    # data = Variable(torch.from_numpy(y).type(torch.FloatTensor)).to(device)
    # TypeError: expected np.ndarray (got Tensor)
    y_real =  y_real.detach().cpu().numpy()
    ####################################################################    轉換成 NumPy array ####################################################################

    # Calculate error and backpropagate
    # e.g:prediction_real實際答案有0有1
    # 透過error_real藉由loss值去看出prediction_real實際答案會有0的
    # true_target。 這個函數主要作用是將 y_real 轉換為張量，確保其數據類型與 prediction_real 匹配
    error_real = loss(prediction_real, true_target(y_real)) #error_real是 預測結果結果（prediction_real）與真實標籤（y_real）之間的差距。 它返回一個表示loss的Tensor
    error_real.backward() # 然後由error_real去覆盤告訴優惠器哪邊可以去做優化，最小化這個損失
    
    # 1.2 Train on Fake Data
    ####################################################################    轉換成 NumPy array ####################################################################
    #prediction_fake 是discriminator對Generator生成的fake_data的預測結果
    #如果 prediction_fake 接近1，那麼discriminator認為fake_data與真實數據非常相似，#因此它們的標籤被預測為1（真實數據）。 
    #如果 prediction_fake 接近0，那麼discriminator為這些fake_data與真實數據差異很大，因此它們的標籤被預測為0（假數據）。
    #在訓練 GAN 時，Generator的目標是生成能夠騙過discriminator的fake_data，
    #以使 prediction_fake 的值接近0.5。 這表示生成的數據足夠逼真，以至於鑒別器無法明確判斷這些數據是真實的還是偽造的。
    #在GAN的訓練中，通常需要比較prediction_fake的值與一個閾值，以確定生成的假數據是否能夠欺騙discriminator。 
    #這個閾值通常是0.5，因為在二元分類問題中，0.5 是一個常用的決策閾值。
    #如果prediction_fake大於0.5，通常可以認為生成的fake_data被判定為真實數據。 如果prediction_fake 小於0.5，通常可以認為生成的fake_data被判定為假數據。
    #這個比較過程在訓練 GAN 時很重要。Generator的目標是生成足夠逼真的假數據，以至於它們的 prediction_fake 值接近0.5，因為這表示它們難以被鑒別器區分。 
    #discriminator的目標是盡量正確地將真實數據與生成的假數據區分開，因此它會努力使prediction_fake接近0或1。
    ####################################################################    轉換成 NumPy array ####################################################################
   
    # Calculate error and backpropagate
    prediction_fake = discriminator(fake_data) #fake_data為0
    # e.g:prediction_fake實際答案有0有1，  zeros_target(N)為解答辨識為0
    # 透過error_real藉由loss值去看出prediction_fake實際答案會有1
    # 然後由error_real去覆盤，告訴優惠器哪邊可以去做優化
    # prediction_fake 是discriminator對生成的fake_data樣本的預測，
    # zeros_target（N） 創建了一個與生成的假數據樣本數量相同的目標張量，該目標張量中的所有元素都設置為0。 然後，通過計算損失（loss）來比較鑒別器的預測和這些目標

    #在訓練 DiscriminatorNet 時，我們需要為模型提供一個目標標籤，以便計算模型的損失（loss）。 
    #對於 DiscriminatorNet，我們希望它能夠將真實數據識別為真實（1），將生成的假數據識別為假（0）。這就是在 GAN 中DiscriminatorNet的任務。
    #zeros_target 函數的作用是生成一個與輸入大小相匹配的標籤張量，其中每個元素都被設置為0。
    #這是因為在訓練鑒別器時，我們希望它將生成的假數據識別為假，所以我們將目標標籤設置為 0。 通過將這個標籤傳遞給loss函數，
    #我們可以計算鑒別器的預測與真實標籤之間的誤差，然後通過反向傳播來更新鑒別器的參數，以便更好地區分真實和假數據。

    #所以，在 train_discriminator 函數中，以下步驟發生：

    #1.prediction_fake 包含了DiscriminatorNet對生成的假數據樣本的預測。

    #zeros_target（N） 生成一個大小為 N（批次中樣本的數量）的目標標籤張量，其中所有元素都為 0。

    #error_fake 計算了鑒別器的預測與這些目標標籤之間的誤差。
    #我們為真實數據設置目標標籤為 1，而為生成的假數據設置目標標籤為 0。 所以 「這些目標標籤」 指的是生成的假數據的標籤，即全為 0。

    #error_fake.backward（） 用於計算 error_
    
    error_fake = loss(prediction_fake, zeros_target(N)) #辨識為0 #通過計算損失（loss）來比較discriminator的預測和這些目標
    # 使Generator能夠生成假數據樣本，從而生成的fake_data的預測接近1，使其能夠騙過discriminator，這是通過最小化損失來實現的
    error_fake.backward()# 然後由error_fake去覆盤告訴優化器哪邊可以去做優化，最小化這個損失
    
    # 1.3 Update weights with gradients
    optimizer.step()#將梯度下降的結果應用到模型的權重上，從而使模型逐漸收斂到更好的狀態，以最小化損失函數
    
    # Return error and predictions for real and fake inputs
    # print("error_real", error_real)
    # print("error_fake", error_fake)
    #error_real：這個變數包含了鑒別器在真實數據上的預測與真實標籤之間的誤差。 它是通過計算鑒別器在真實數據上的預測結果與真實標籤（即真實數據的標籤）之間的損失來得到的。
    #error_fake：這個變數包含了鑒別器在生成數據上的預測與生成標籤之間的誤差。 它是通過計算鑒別器在生成的數據上的預測結果與生成標籤之間的損失來得到的。
    #error_real + error_fake：這是discriminator在真實數據和生成數據上的總loss，它衡量了鑒別器的性能。
    return error_real + error_fake, prediction_real, prediction_fake

# training generator
def train_generator(optimizer, fake_data):
    # generato的目標是生成看起來像真實數據的假數據，以騙過discriminator。
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    # fake_data 是generator生成的假數據。 這些數據是通過將noise作為輸入傳遞給generator來生成的。
    #prediction，即discriminator對這些假數據的預測結果
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    #loss 函數計算 prediction 與目標標籤之間的誤差。 
    #在這裡，目標標籤是 ones_target（N），表示所有這些假數據都希望被discriminator認為是真實數據
    error = loss(prediction, ones_target(N))
    error.backward()# 然後由error去覆盤告訴優化器哪邊可以去做優化，最小化這個損失 ，復盤對象
    # Update weights with gradients
    optimizer.step()#來根據計算得到的梯度來更新生成器的權重
    # Return error
    return error #回傳loss



#####################********** GAN Parameters  **************##############
            
x_g=np.copy(x_train)
y_g=np.zeros(y_train.shape)

print("x_g",len(x_g))
print("y_g",len(y_g))
# 找到值為8的標籤的索引
indices = [index for index, value in enumerate(y_train) if value == weakLabel]
print(f"值为{weakLabel}的标签的索引: {indices}")
# 將值為8的標籤標記為“真實”或“正類別”
y_g[indices==weakLabel] = 1 #y_g[y_partial==weakpoint]=1 #等於1表示標記為“真實”或“正類別”特定類別（在這裡是 類別）標記為1，而其他類別標記為0


# 將其他標籤標記為0
y_g[~np.isin(np.arange(len(y_g)), indices)] = 0


data = [(x, y) for x, y in zip(x_g,y_g)]

print("data",len(data))
dataSet=Data_Loader(data)
data_loader = DataLoader(dataset=dataSet, batch_size=256, shuffle=True)
discriminator = DiscriminatorNet().to(device)
generator = GeneratorNet().to(device)

###########################################################
# d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
d_optimizer = optim.SGD(discriminator.parameters(), lr=0.0002) #改用SGD
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# nn.BCELoss() 是二元交叉熵损失函数（Binary Cross-Entropy Loss）
# 通常用於二元分類問題
# 使用 nn. BCELoss（）時，通常需要將模型的輸出視為概率值，並將真實標籤表示為0或1。
# 然後，損失函數將計算模型的輸出與真實標籤之間的二進位交叉熵損失
loss = nn.BCELoss().cuda()  # loss function使用cuda
###########################################################
num_test_samples = 20
#*********************************************  Running GAN   ***************************
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
# num_epochs = 1
d_errs=[]
g_errs=[]

startimeStamp, starttime= getStartorEndtime("start")
print("data_loader",len(data_loader))
for epoch in range(num_epochs):
    print("epoch: ",epoch)
    g_error_sum=0
    d_error_sum=0
    counter = 0
    #第二個 for 循環主要是執行 GAN（生成對抗網絡）的訓練過程，包括訓練鑑別器（Discriminator）和生成器（Generator）以及生成新的數據
    for n_batch, (real_batch,y_real) in enumerate(data_loader):# 一次復盤完，就生一次資料去考試
        counter+=1
        # print("n_batch",n_batch)
        # print("real_batch",len(real_batch))
        # print("y_real",len(y_real))
        # print("data_loader",len(data_loader))
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
        #y_real=np.array(y_real)
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,y_real)

        # 2. Train Generator

        # Generate fake data
        fake_data = generator(noise(N)).to(device)
        # Train G
        # train_generator裡面會做覆盤，每次訓練持續告訴優化器哪邊還可以優化
        g_error = train_generator(g_optimizer, fake_data)
        
        g_error_sum+=g_error #把每次bacth_size的g_loss加總起來
        d_error_sum+=d_error #把每次bacth_size的d_loss加總起來  
        #print(counter) 
        # g_error_sum = g_error_sum/counter #除上counter一次epoch的平均g_loss 
        # d_error_sum = d_error_sum/counter #一次epoch的平均d_loss

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
        #     plt.show()
        #     plt.plot()
        #     plt.plot(d_errs.detach().numpy())
        #     plt.plot(g_errs.detach().numpy())
        #     plt.show()
    #因tensor 在 GPU上會報下列
    #can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    #將 tensor 放在 CPU，再轉numpy
    g_error_sum = g_error_sum/counter #除上counter一次epoch的平均g_loss 
    d_error_sum = d_error_sum/counter #一次epoch的平均d_loss
    d_error_sum = d_error_sum.cpu()
    g_error_sum = g_error_sum.cpu()
    d_error_sum = d_error_sum.detach().numpy()
    g_error_sum = g_error_sum.detach().numpy()
    d_errs.append(d_error_sum) #一次epoch append一次
    g_errs.append(g_error_sum) 


############  Generating and adding new samples  ###########
# numOfSamples 表示生成的假資料的數量，用(generator) 生成的假特徵數據
x_syn=(generator(noise(numOfSamples))).detach().to(device).cpu().numpy() 
# 將y_syn設定為一個具有相同形狀的numOfSamples的數組，並將每個元素設為weakpoint。這裡，weakpoint是生成假數據的標籤
# numOfSamples是50，並將它們的標籤都設置為weakpoint，即8。因此，生成標籤都是8的50個假樣本
y_syn=np.ones(numOfSamples)*weakLabel #這是與 x_syn 相關的標籤（labels）。在這裡，所有這些假數據的標籤都被設置為 weakpoint

#保存透過GAN生出來的資料
mergeDataFrameAndSaveToCsv("GAN", x_syn, y_syn, file, weakLabel, num_epochs)
# print("透過GAN生成出來的資料Shapes: ")#顯示 (資料筆數, 特徵數)。
# print(x_syn.shape,y_syn.shape)#顯示 (資料筆數,)，因為這是一維的標籤數組
# print("Number of samples generated in current batch:", numOfSamples)
#************************************************
newdata = pd.read_csv(filepath + f"GAN_data_train_half1\\GAN_data_generate_weaklabel_{weakLabel}_epochs_{num_epochs}.csv")
# newdata = newdata.iloc[1:]
train_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'train_df_half1.csv'))
#第一列名稱要一樣append時才部會往外跑
train_dataframe = train_dataframe.append(newdata)
train_dataframe.to_csv(filepath+ f"GAN_data_train_half1\\GAN_data_{file}_ADD_weakLabel_{weakLabel}.csv", index=False)

endtimeStamp,endtime = getStartorEndtime("end")

# 绘制鉴别器和生成器损失曲线
# np.arange參數解釋：
# start（可選）：表示數列的起始值，預設為0。
# stop：表示數列的結束值，生成的陣列不包括這個值。
# step（可選）：表示數列的步進值，即相鄰兩個數之間的差。 預設為1。
# dtype（可選）：生成數位的數據類型。
# np.arange 創建一個從 start 到 stop 的數列，步進值為 step。 這個函數通常用於生成一組整數或浮點數，供迴圈或數據處理使用。
# 例如，np.arange（0， 10， 2） 將生成一個包含 [0， 2， 4， 6， 8] 的NumPy陣列。 在你的代碼中，test_range 是一個包含從0到 len（d_errs） - 1 的整數的NumPy陣列，用於表示Epochs。
test_range = np.arange(len(d_errs))
print(test_range)
# 繪製兩個損失曲線
plt.figure(figsize=(10, 5))
plt.plot(test_range, d_errs, label="Discriminator Loss", color='blue')
plt.plot(test_range, g_errs, label="Generator Loss", color='green')
# 設置標籤、標題、圖例、網格等
plt.xlabel("Epochs")
plt.ylabel("Loss")
# 添加標題，並在標題和文本之間添加空行
title_text = f"Label_{weakLabel}_Discriminator and Generator Loss Over Epochs\nStartTime:{starttime} EndTime:{endtime}"
plt.title(title_text)
# plt.title(f"Label_{weakLabel}_Discriminator and Generator Loss Over Time")
plt.legend() #作用是將圖例添加到當前的繪圖中
plt.grid(True)
# 儲存圖片並顯示
plt.savefig(f"./GAN_data_train_half1/epochs_{num_epochs}_weaklabel_{weakLabel}_Loss.png")
plt.show()

CalculateTime(endtimeStamp, startimeStamp)
print("start time",starttime)
print("end time",endtime)
# x_train=np.concatenate((x_train,x_syn),axis=0)
# y_train=np.concatenate((y_train,y_syn),axis=0)

# print("Shapes: ")#顯示 (資料筆數, 特徵數)。
# print(x_train.shape,y_train.shape)#顯示 (資料筆數,)，因為這是一維的標籤數組

# print("combined_data\n",generateNewdata['Label'].value_counts())
