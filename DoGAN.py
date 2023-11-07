import warnings
import os
import time
import datetime
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
from sklearn.metrics import accuracy_score
from scipy.signal import savgol_filter
# 引用 datasetsPreprocess.py 中的函數
from mytoolfunction import SaveDataToCsvfile, generatefolder, mergeDataFrameAndSaveToCsv, ChooseTrainDatastes, ParseCommandLineArgs
from mytoolfunction import getStartorEndtime, CalculateTime
from collections import Counter

filepath = "D:\\Labtest20230911\\"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\After_PCA產生天日期的資料夾
generatefolder(filepath + "data", "\\After_GAN")
generatefolder(filepath + "data\\After_GAN\\", today)
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

counter = Counter(y_train)
print(counter)

########################  Weak labels and numOfSamples ##############################
numOfSamples = 1500
# #############################################################################
again_epoch = 100                                                   ######################################################## 0628修改 ###############################################
for again in range(again_epoch): 
    print('第',again,'Round')
    start = time.time()
    cuda = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor   # GPU
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

        # 繪製經過smooth GD_loss圖
    def Print_smooth_loss(d_errs, g_errs, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        #plt.plot()
        #plt.plot(d_errs)
        d_errs_prime = savgol_filter(d_errs,53,3)
        #plt.plot(g_errs)
        g_errs_prime = savgol_filter(g_errs,53,3)
        title_name = "Label " + str(weakpoint)
        plt.title(title_name)
        plt.plot(d_errs_prime, color="blue")
        plt.plot(g_errs_prime, color="black")
        plt.legend(["D_loss", "G_loss"], loc="upper left")
        smooth_photo = "./data/After_GAN/" +{today}+ "Label_" + str(weakpoint) + "_" + str(epoch+1) + "_smooth.jpg"
        plt.savefig(smooth_photo)
        plt.close("all")

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

    def soft_ones_target(size):   #全部填1
        '''
        Tensor containing zeros, with shape = size
        '''
        #data = Variable(torch.ones(size, 1)).cuda()
        ##### 使用軟標籤 ######
        radom_value = np.random.uniform(0.9, 1, [size,1])
        radom_value = torch.from_numpy(radom_value).type(torch.FloatTensor)
        data = Variable(radom_value).cuda()
        ### 使用軟標籤  ######
        return data

    def zeros_target(size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1)).to(device)
        return data

    def soft_zeros_target(size):    #全部填0
        '''
        Tensor containing zeros, with shape = size
        '''
        #data = Variable(torch.zeros(size, 1)).cuda()
        ##### 使用軟標籤 ######
        radom_value = np.random.uniform(0, 0.1, [size,1])
        radom_value = torch.from_numpy(radom_value).type(torch.FloatTensor)
        data = Variable(radom_value).cuda()
        ##### 使用軟標籤 ######
        return data

    def true_target(y):
        '''
        Tensor containing zeros, with shape = size
        '''
        random_value = np.random.uniform(0.9, 1, [len(y_real),1])
        data = Variable(torch.from_numpy(random_value).type(torch.FloatTensor)).to(device)
        # retrun 型態為PyTorch Tensor的data
        # data = data[:,None]
        return data

    #### model generation for discriminator
    class DiscriminatorNet(torch.nn.Module):
        """
        A three hidden-layer discriminative neural network
        """
        def __init__(self):
            super(DiscriminatorNet, self).__init__()
            # n_features = 78 #78並非單單是因為特徵是有78個，而是因為GeneratorNet會生成的78個特徵假資料會餵給DiscriminatorNet做判斷，兩者做矩陣相處要match，
            n_features = 58
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
            n_out = 58  # 要生產資料集特征的数量匹配的假資料
            self.fc1 = nn.Linear(n_noise, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, n_out)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
            

    # training discriminator 
    def train_discriminator(optimizer, real_data, fake_data, y_real):
        # discriminator任務盡可能準確地將真實數據與生成數據區分開來
        N = real_data.size(0) # 返回張量 real_data 第一個維度的大小，通常是批次大小（batch size），表示在這一批次中有多少樣本
        # Reset gradients
        optimizer.zero_grad() #optimizer是用於更新神經網路權重的優化器，zero_grad方法用於將所有模型參數（權重和偏置）的梯度歸零，以確保在每個批次的反向傳播之前沒有任何殘留的梯度資訊。
        
        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data) #real_data為1
        ####################################################################    轉換成 NumPy array ####################################################################
        error_real = loss(prediction_real, true_target(y_real)) 
        error_real.backward() 
        #print("判斷為真實資料機率: ", prediction_real)
        threshold = torch.tensor([0.5]).cuda()  #閥值設定
        real_final = (prediction_real > threshold).float()*1  # 機率分布大於0.5，即視為真實資料 1
        real_result = real_final.data.cpu().numpy()
        #print("真實資料判斷結果: ", real_result)

        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data) #fake_data為0
        error_fake = loss(prediction_fake, soft_zeros_target(N))
        error_fake.backward()# 假資料Loss的反向傳播
        threshold = torch.tensor([0.5]).cuda()  #閥值設定
        fake_final = (prediction_fake > threshold).float()*1 # 機率分布大於0.5，即視為生成資料 0
        fake_result = fake_final.data.cpu().numpy()

        # error_real與error_fake的平均
        # error_real：這個變數包含了鑒別器在真實數據上的預測與真實標籤之間的誤差。 它是通過計算鑒別器在真實數據上的預測結果與真實標籤（即真實數據的標籤）之間的損失來得到的。
        # error_fake：這個變數包含了鑒別器在生成數據上的預測與生成標籤之間的誤差。 它是通過計算鑒別器在生成的數據上的預測結果與生成標籤之間的損失來得到的。
        # error：這是discriminator在真實數據和生成數據上的總loss，它衡量了鑒別器的性能。
        error = (error_real + error_fake) / 2

        # 1.3 Update weights with gradients
        optimizer.step()
    
        
        return error, real_result, fake_result

    # training generator
    def train_generator(optimizer, fake_data):
        # generato的目標是生成看起來像真實數據的假數據，以騙過discriminator。
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        prediction = discriminator(fake_data)
        error = loss(prediction, soft_ones_target(N))  #驗證，將ones_target改成zeros_target
        error.backward()
        optimizer.step()
        ##### test 並且設定 threshold ######
        #print("生成資料的機率分布: ", prediction)
        threshold = torch.tensor([0.5]).cuda()  #閥值設定
        final = (prediction > threshold).float()*1  # 希望被判斷出來是1
        result = final.data.cpu().numpy()
        
        return error, result #回傳loss


    i = 1 ### count用
    j = 0
    list_count = [] 
    list_GD = [[]]
    test = [] # 重組生成資料用
    auc_class = 0 #計算class編號
    patient = 0
    patient2 = 0
    loss_patient = 0


    #####################********** GAN Parameters  **************##############

    #################這邊使用硬標籤##########################################           
    # x_g=np.copy(x_train)
    x_g=np.copy(x_train[y_train == weakLabel]) #使用 NumPy 複製了 x_train 中等於 weakLabel 的行
    y_g=np.zeros(y_train.shape)

    print(x_g.shape)
    # 找到值為8的標籤的索引
    indices = [index for index, value in enumerate(y_train) if value == weakLabel]
    # print(f"值为{weakLabel}的标签的索引: {indices}")
    # 將值為8的標籤標記為“真實”或“正類別”
    y_g[indices==weakLabel] = 1 #y_g[y_partial==weakpoint]=1 #等於1表示標記為“真實”或“正類別”特定類別（在這裡是 類別）標記為1，而其他類別標記為0


    # 將其他標籤標記為0
    y_g[~np.isin(np.arange(len(y_g)), indices)] = 0
    #################這邊使用硬標籤########################################## 
    x_g = torch.from_numpy(x_g).type(torch.cuda.FloatTensor) 
    y_g = torch.from_numpy(y_g).type(torch.cuda.FloatTensor)            
    print("x_g",len(x_g))
    print("y_g",len(y_g))

    data = [(x, y) for x, y in zip(x_g,y_g)]

    print("data",len(data))
    dataSet=Data_Loader(data)
    data_loader = DataLoader(dataset=dataSet, batch_size=256, shuffle=True)
    discriminator = DiscriminatorNet().to(device)
    generator = GeneratorNet().to(device)

    ###########################################################
    # d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    d_optimizer = optim.SGD(discriminator.parameters(), lr=0.0001) #改用SGD
    g_optimizer = optim.SGD(generator.parameters(), lr=0.0001)
    loss = nn.BCELoss().cuda()  # loss function使用cuda
    ###########################################################
    #*********************************************  Running GAN   ***************************
            
    d_errs=[]
    g_errs=[]
    accuracy=[]  #計算鑑別器準確率(訓練生成器過程)
    D_real_accuracy = [] #計算鑑別器真實資料之準確率(訓練鑑別器過程)
    D_fake_accuracy = [] #計算鑑別器生成資料之誤判率(訓練鑑別器過程)
    D_combine_accuracy = [] #計算鑑別器總準確率(訓練鑑別器過程)
    startimeStamp, starttime= getStartorEndtime("start")
    # print("data_loader",len(data_loader))
    for epoch in range(num_epochs):
        t = 0 #
        g_error_sum=0
        d_error_sum=0
        acc = 0
        D_training_real_acc = 0
        D_training_fake_acc = 0
        D_combine_acc = 0
        counter = 0
        #第二個 for 循環主要是執行 GAN（生成對抗網絡）的訓練過程，包括訓練鑑別器（Discriminator）和生成器（Generator）以及生成新的數據
        for n_batch, (real_batch,y_real) in enumerate(data_loader):
            counter+=1
            N = real_batch.size(0) #回傳real_batch的行數(0表示axis)
        # ----------1. Train Discriminator----------------------------
            real_data = Variable(real_batch).to(device)
            # Generate fake data and detach 
            fake_data = generator(noise(N)).detach().to(device)

            #print("增加維度",y_real.shape)
            y_real = y_real.data.cpu().numpy()
            y_real = np.expand_dims(y_real,axis=1) #增加維度
            y_real = np.expand_dims(y_real,axis=1)
            #y_real=np.array(y_real)
            #print("增加維度",y_real.shape)
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,y_real)
            d_pred_real = np.squeeze(d_pred_real) #降維
            #print("資料: ", d_pred_real)
            #print("預測資料維度: ", d_pred_real.shape)
            D_real = ones_target(N)
            D_real = D_real.data.cpu().numpy()
            #print("實際: ", D_real)
            #print("實際資料維度: ", D_real.shape)
            D_real_acc = accuracy_score(D_real, d_pred_real)

            D_fake = ones_target(N)  ## 目的為查看鑑別器對於生成資料之誤判率，因此使用1(真實資料標籤)
            D_fake = D_fake.data.cpu().numpy()
            D_fake_acc = accuracy_score(D_fake, d_pred_fake)

            #D_mean_acc = (D_real_acc + D_fake_acc) / 2
            D_mean_acc = D_real_acc / ((D_real_acc + 0.01) + D_fake_acc)  # alpha = 0.01，避免D_mean_acc為nan
            
            D_training_real_acc += D_real_acc
            sum_D_training_real_acc = D_training_real_acc / counter
            D_training_fake_acc += D_fake_acc
            sum_D_training_fake_acc = D_training_fake_acc / counter
            D_combine_acc += D_mean_acc
            sum_D_combine_acc = D_combine_acc / counter

        # ------------2. Train Generator-----------------------------

            # Generate fake data
            fake_data = generator(noise(N)).to(device)#前面訓練 D時已經生成1組fake_data
            # Train G
            g_error, result = train_generator(g_optimizer, fake_data)
            real_result = zeros_target(N)
            real_result = real_result.data.cpu().numpy()
            sub_acc = accuracy_score(real_result, result)

            g_error_sum+=g_error #把每次bacth_size的g_loss加總起來
            d_error_sum+=d_error #把每次bacth_size的d_loss加總起來  
            g_error_sum = g_error_sum/counter #除上counter一次epoch的平均g_loss 
            d_error_sum = d_error_sum/counter #一次epoch的平均d_loss
        #--------------------Evan----------------------#
            acc+=sub_acc
            sum_acc = acc/counter
                
            g = g_error
            d = d_error 
            g = g.item()
            d = d.item()
            #print("G_loss: ",g)
            #print("D_loss: ",d)
            list_count.append(int(i))
            list_GD.append([g,d])
            i = i + 1
            t = t + 1
            #print("G_error: ",type(g_error))

            g_error = g_error.item()
            g_error_sum+=g_error #型態tensor
            g_sum = g_error_sum/counter
            

            d_error = d_error.item()
            d_error_sum+=d_error
            d_sum = d_error_sum/counter
            
        
        print('Epoch: {}, 生成訓練之鑑別器準確率: {:.4f}, 真實資料準確率: {:.4f}, 生成資料誤判率: {:.4f}, 鑑別器總準確率: {:.4f}' .format(epoch+1, sum_acc, sum_D_training_real_acc, sum_D_training_fake_acc, sum_D_combine_acc))
        
        accuracy.append(sum_acc)
        D_real_accuracy.append(sum_D_training_real_acc)
        D_fake_accuracy.append(sum_D_training_fake_acc)
        D_combine_accuracy.append(sum_D_combine_acc)
        d_errs.append(d_sum) #一次epoch append一次
        g_errs.append(g_sum) 


    ############  Generating and adding new samples  ###########
    # numOfSamples 表示生成的假資料的數量，用(generator) 生成的假特徵數據
    x_syn=(generator(noise(numOfSamples))).detach().to(device).cpu().numpy() 
    # 將y_syn設定為一個具有相同形狀的numOfSamples的數組，並將每個元素設為weakpoint。這裡，weakpoint是生成假數據的標籤
    # numOfSamples是50，並將它們的標籤都設置為weakpoint，即8。因此，生成標籤都是8的50個假樣本
    y_syn=np.ones(numOfSamples)*weakLabel #這是與 x_syn 相關的標籤（labels）。在這裡，所有這些假數據的標籤都被設置為 weakpoint

    # #保存透過GAN生出來的資料
    # # mergeDataFrameAndSaveToCsv("GAN", x_syn, y_syn, file, weakLabel, num_epochs)
    df_x_train = pd.DataFrame(x_syn)
    df_y_train = pd.DataFrame(y_syn)

    # 使用concat函数将它们合并
    generateNewdata = pd.concat([df_x_train, df_y_train], axis=1)
    generateNewdata.to_csv(filepath + "data\\After_GAN\\"+ today + f"\\GAN_Label_{weakLabel}.csv", index=False)
    print("透過GAN生成出來的資料Shapes: ")#顯示 (資料筆數, 特徵數)。
    print(x_syn.shape,y_syn.shape)#顯示 (資料筆數,)，因為這是一維的標籤數組

    endtimeStamp,endtime = getStartorEndtime("end")
    test_range = np.arange(len(d_errs))
    print(test_range)
    print(d_errs)
    print(g_errs)
    # d_errs = d_errs.cpu()
    # d_errs = d_errs.detach().numpy()
    d_errs_cpu = [item.cpu().detach().numpy() for item in d_errs]
    g_errs_cpu = [item.cpu().detach().numpy() for item in g_errs]
    print(d_errs_cpu)
    # Print_smooth_loss(d_errs_cpu, g_errs_cpu, weakLabel, epoch)
# 繪製兩個損失曲線
plt.figure(figsize=(10, 5))
plt.plot(test_range, d_errs_cpu, label="Discriminator Loss", color='blue')
plt.plot(test_range, g_errs_cpu, label="Generator Loss", color='green')
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
plt.savefig(f"./data/After_GAN/{today}/epochs_{num_epochs}_weaklabel_{weakLabel}_Loss.png")
plt.show()

CalculateTime(endtimeStamp, startimeStamp)
print("start time",starttime)
print("end time",endtime)
