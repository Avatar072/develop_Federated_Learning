import matplotlib.pyplot as plt
from sklearn import preprocessing 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder 
import random
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

import torch 
from torch import Tensor
import numpy as np 
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seaborn as sn
import torch.optim as optim
from scipy.signal import savgol_filter
import time
import os
fileName = 'save_file/end_loop.txt'
again_epoch = 100                                                   ######################################################## 0628修改 ###############################################
for again in range(again_epoch): 
    print('第',again,'Round')
    start = time.time()
    cuda = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)

    class_nums = [0,1,2,3,4,5]
    max_epoch = 500
    filepath = "D:\\Labtest20230911\\"
    # x_train = np.loadtxt("dataset/pca_70_x_train.txt")
    # y_train = np.loadtxt("dataset/pca_70_y_train.txt")
    # x_test = np.loadtxt("dataset/pca_70_x_test.txt")
    # y_test = np.loadtxt("dataset/pca_70_y_test.txt")
    # x_train = np.loadtxt("Thursday_x_train.txt")
    # y_train = np.loadtxt("Thursday_y_train.txt")
    # x_test = np.loadtxt("Thursday_x_test.txt")
    # y_test = np.loadtxt("Thursday_y_test.txt")
    x_train = np.load(filepath + "x_train_1.npy", allow_pickle=True)
    y_train = np.load(filepath + "y_train_1.npy", allow_pickle=True)
    x_test = np.load(filepath + "x_test_1.npy", allow_pickle=True)
    y_test = np.load(filepath + "y_test_1.npy", allow_pickle=True)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor   # GPU

    X_train = torch.from_numpy(x_train).type(torch.cuda.FloatTensor)
    X_test = torch.from_numpy(x_test).type(torch.cuda.FloatTensor)

    Y_train = torch.from_numpy(y_train).type(torch.cuda.LongTensor)
    Y_test = torch.from_numpy(y_test).type(torch.cuda.LongTensor)

    Training_data = [(x, y) for x, y in zip(X_train,Y_train)]
    Testing_data = [(x, y) for x, y in zip(X_test,Y_test)]

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


    # 繪製原始GD_loss圖
    def Print_loss(d_errs, g_errs, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        #plt.plot()
        plt.plot(d_errs)
        plt.plot(g_errs)
        title_name = "Label " + str(weakpoint)
        plt.title(title_name)
        plt.legend(["D_loss","G_loss"], loc="upper right")
        photo_name = "save_file/" + "Label_" + str(weakpoint) + "_" + str(epoch+1) + ".jpg"
        plt.savefig(photo_name)
        #plt.show()
        plt.close("all")

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
        smooth_photo = "save_file/" + "Label_" + str(weakpoint) + "_" + str(epoch+1) + "_smooth.jpg"
        plt.savefig(smooth_photo)
        plt.close("all")
        #plt.show()

    # 繪製G_Training stage 鑑別器準確率
    def Print_G_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        plt.plot(accuracy)
        plt.title("Discriminator's Accuracy when training generator stage")
        accuracy_photo = "save_file/" + "G_" + str(epoch+1) + "_accuracy.jpg"
        #plt.show()
        plt.savefig(accuracy_photo)
        plt.close("all")

    # 繪製D_Training stage 鑑別器真實資料準確率
    def Print_D_real_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        plt.plot(accuracy)
        plt.title("Discriminator's Accuracy for real data")
        accuracy_photo = "save_file/"+ "D_real_" + str(epoch+1) + "_accuracy.jpg"
        #plt.show()
        plt.savefig(accuracy_photo)
        plt.close("all")

    def Print_smooth_D_real_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        accuracy_prime = savgol_filter(accuracy,21,3)
        plt.plot(accuracy_prime)
        plt.title("Discriminator's Accuracy for real data")
        accuracy_photo = "save_file/"+ "D_real_smooth_" + str(epoch+1) + "_accuracy.jpg"
        plt.savefig(accuracy_photo)
        plt.close("all")

    # 繪製D_Training stage 鑑別器生成資料準確率
    def Print_D_fake_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        plt.plot(accuracy)
        plt.title("Discriminator's false Accuracy for fake data")
        accuracy_photo = "save_file/"+ "D_fake_" + str(epoch+1) + "_accuracy.jpg"
        #plt.show()
        plt.savefig(accuracy_photo)
        plt.close("all")

    def Print_smooth_D_fake_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        accuracy_prime = savgol_filter(accuracy,21,3)
        plt.plot(accuracy_prime)
        plt.title("Discriminator's false Accuracy for fake data")
        accuracy_photo = "save_file/"+ "D_fake_smooth_" + str(epoch+1) + "_accuracy.jpg"
        plt.savefig(accuracy_photo)
        plt.close("all")

    # 繪製D_Training stage 鑑別器整體準確率
    def Print_D_total_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        plt.plot(accuracy)
        plt.title("Discriminator total Precision")
        accuracy_photo = "save_file/"+ "D_total_" + str(epoch+1) + "_accuracy.jpg"
        #plt.show()
        plt.savefig(accuracy_photo)
        plt.close("all")

    def Print_smooth_D_total_accuracy(accuracy, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        accuracy_prime = savgol_filter(accuracy,21,3)
        plt.plot(accuracy_prime)
        plt.title("Discriminator total Precision")
        accuracy_photo = "save_file/"+ "D_total_smooth_" + str(epoch+1) + "_accuracy.jpg"
        plt.savefig(accuracy_photo)
        plt.close("all")

    def Print_smooth_real_fake(real_acc, fake_acc, weakpoint, epoch):
        fig = plt.figure(figsize=(18,4.8))
        real_prime = savgol_filter(real_acc, 21, 3)
        fake_prime = savgol_filter(fake_acc, 21, 3)
        plt.plot(real_prime, color="blue")
        plt.plot(fake_prime, color="black")
        plt.title("Real data acc and Fake data false acc")
        accuracy_photo = "save_file/"+ "R&F_smooth_" + str(epoch+1) + "_accuracy.jpg"
        plt.savefig(accuracy_photo)
        plt.close("all")

    def remove_string_type(x):# x 是
        x_removed = np.delete(x,[0,1,2,3,4,5],1)
        #-------共9個 features -------------#
        protocol_x = x[:,0:1] # feature 1
        service_x = x[:,1:2] # feature 2
        flag_x = x[:,2:3] # feature 3
        land_x = x[:,3:4] # feature 6
        logged_in_x = x[:,4:5] #feature 11
        root_shell_x = x[:,5:6] # feature 13
        # su_attempted_x = x[:,14:15] # feature 14 
        # is_hot_login_x = x[:,20:21] # feature 20
        # is_guest_login_x = x[:,21:22] #feature 21  
        #-------------共9個 features -------------#
        list_protocol = protocol_x.tolist()
        list_service = service_x.tolist()
        list_flag = flag_x.tolist()
        list_land = land_x.tolist()
        list_logged_in = logged_in_x.tolist()
        list_root_shell = root_shell_x.tolist()
        # list_su_attempted = su_attempted_x.tolist()   
        # list_is_hot_login = is_hot_login_x.tolist()
        # list_is_guest_login = is_guest_login_x.tolist()
        
        return x_removed, list_protocol, list_service, list_flag, list_land, list_logged_in, list_root_shell

    def noise(size):
        '''
        Generates a 1-d vector of gaussian sampled random values
        '''
        n = Variable(torch.randn(size, 20)).cuda()  # 修改生成器的input
        #print("Shape of n: ", n.shape)
        n = n[:,None]
        return n

    def ones_target(size):   #全部填1
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.ones(size, 1)).cuda()
        ##### 使用軟標籤 ######
        #a = np.random.uniform(0.9, 1, [size,1])
        #aa = torch.from_numpy(a).type(torch.FloatTensor)
        #data = Variable(aa).cuda()
        ### 使用軟標籤  ######
        return data

    def soft_ones_target(size):   #全部填1
        '''
        Tensor containing zeros, with shape = size
        '''
        #data = Variable(torch.ones(size, 1)).cuda()
        ##### 使用軟標籤 ######
        a = np.random.uniform(0.9, 1, [size,1])
        aa = torch.from_numpy(a).type(torch.FloatTensor)
        data = Variable(aa).cuda()
        ### 使用軟標籤  ######
        return data

    def zeros_target(size):    #全部填0
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1)).cuda()
        ##### 使用軟標籤 ######
        #a = np.random.uniform(0, 0.1, [size,1])
        #aa = torch.from_numpy(a).type(torch.FloatTensor)
        #data = Variable(aa).cuda()
        ##### 使用軟標籤 ######
        return data

    def soft_zeros_target(size):    #全部填0
        '''
        Tensor containing zeros, with shape = size
        '''
        #data = Variable(torch.zeros(size, 1)).cuda()
        ##### 使用軟標籤 ######
        a = np.random.uniform(0, 0.1, [size,1])
        aa = torch.from_numpy(a).type(torch.FloatTensor)
        data = Variable(aa).cuda()
        ##### 使用軟標籤 ######
        return data
    '''  原始版本 true_target
    def true_target(y):
        data= Variable(torch.from_numpy(y).type(torch.cuda.FloatTensor))
        return data
    '''
    def true_target(y):
        '''
        Tensor containing zeros, with shape = size
        '''
        a = np.random.uniform(0.9, 1, [len(y_real),1])
        data= Variable(torch.from_numpy(a).type(torch.cuda.FloatTensor))
        data = data[:,None]
        return data

    class DiscriminatorNet(torch.nn.Module):
        """
        A three hidden-layer discriminative neural network
        """
        def __init__(self):
            super(DiscriminatorNet, self).__init__()
            n_features = 58 # 41-9
            n_out = 1
            
            
            self.hidden0 = nn.Sequential( 
                nn.Linear(n_features, 64), #45
                nn.Tanh(),
                nn.Dropout(0.2)
            )
            
            self.hidden1 = nn.Sequential(
                nn.Linear(64, 64), # 45, 50
                #nn.LeakyReLU(inplace=True),
                nn.Tanh(),
                nn.Dropout(0.2)
            )
            
            self.hidden2 = nn.Sequential(
                nn.Linear(64, 64), #50 25
                #nn.LeakyReLU(inplace=True),
                nn.Tanh(),
                nn.Dropout(0.2)
            )
            
            #self.hidden3 = nn.Sequential(
                #nn.Linear(64, 64),
                #nn.LeakyReLU(inplace=True),
                #nn.Tanh(),
                #nn.Dropout(0.5)
            #)
            
            self.out = nn.Sequential(
                torch.nn.Linear(64, n_out), # 25
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            x = self.hidden0(x)
            #print("Discriminator", x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            #x = self.hidden3(x)
            x = self.out(x)
            return x

    ##################  Based on CNN Generator  #######################
    class GeneratorNet(torch.nn.Module):
        def __init__(self):
            super(GeneratorNet, self).__init__()
            # image shape is 1 * 20 * 20, where 1 is one color channel
            # 20 * 20 is the image size
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5)    # output shape = 3 * 36 * 36
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)                       # output shape = 3 * 18 * 18
            # intput shape is 3 * 18 * 18
            self.conv2 = nn.Conv1d(in_channels=3, out_channels=9, kernel_size=5)    # output shape = 9 * 14 * 14
            # add another max pooling, output shape = 9 * 4 * 4
            self.fc1 = nn.Linear(3*8, 24)
            #self.bn1 = nn.BatchNorm1d(24)  # BN 1，原本24
            # self.d1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(24, 24)
            #self.bn2 = nn.BatchNorm1d(24)  # BN 2
            # self.d2 = nn.Dropout(0.2)
            # last fully connected layer output should be same as classes
            self.fc3 = nn.Linear(24, 24)
            #self.bn3 = nn.BatchNorm1d(24)  # BN 3
            # self.d3 = nn.Dropout(0.2)
            self.fc4 = nn.Linear(24, 58)
            #elf.d4 = nn.Dropout(0.2)
            #self.fc5 = nn.Linear(24, 24)
            #self.d5 = nn.Dropout(0.2)
            #self.fc6 = nn.Linear(24, 24)
            #self.d6 = nn.Dropout(0.2)
            #self.fc7 = nn.Linear(24, 28)

            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            #x = self.pool(F.relu(self.conv2(x)))
            # flatten all dimensions except batch
            x = torch.flatten(x, 1)
            # fully connected layers
            x = F.relu(self.fc1(x))
            # x = self.d1(x)
            x = F.relu(self.fc2(x))
            # x = self.d2(x)
            x = self.fc3(x)
            # x = self.d3(x)
            x = self.fc4(x)

            return x
        
    def train_discriminator(optimizer, real_data, fake_data, y_real):
        
        N = real_data.size(0)
        # Reset gradients
        optimizer.zero_grad() #鑑別器梯度歸零

        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, true_target(y_real))
        #error_real = loss(prediction_real, soft_ones_target(y_real))
        #print("Error Real: ",error_real) # loss資訊_real_data
        error_real.backward()  # 真實資料Loss的反向傳播
        #print("判斷為真實資料機率: ", prediction_real)
        threshold = torch.tensor([0.5]).cuda()  #閥值設定
        real_final = (prediction_real > threshold).float()*1  # 機率分布大於0.5，即視為真實資料 1
        real_result = real_final.data.cpu().numpy()
        #print("真實資料判斷結果: ", real_result)

        
        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, soft_zeros_target(N))
        #print("Error Fake: ",error_fake) # loss資訊_Fake_data
        error_fake.backward() # 假資料Loss的反向傳播
        #print("判斷為生成資料機率: ", prediction_fake)
        threshold = torch.tensor([0.5]).cuda()  #閥值設定
        fake_final = (prediction_fake > threshold).float()*1 # 機率分布大於0.5，即視為生成資料 0
        fake_result = fake_final.data.cpu().numpy()
        #print("生成資料判斷結果: ", fake_result)
        
        # error_real與error_fake的平均
        error = (error_real + error_fake) / 2
        #error.backward()
        
        # 1.3 Update weights with gradients
        optimizer.step()
        
        # Return error and predictions for real and fake inputs
        #return error_real + error_fake, prediction_real, prediction_fake  #d_error = error_real + error_fake
        #return error, prediction_real, prediction_fake  #d_error = error_real + error_fake
        return error, real_result, fake_result

    # training generator
    def train_generator(optimizer, fake_data):
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = loss(prediction, soft_ones_target(N))  #驗證，將ones_target改成zeros_target
        #print("Generatot Error loss: ", error)
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        ##### test 並且設定 threshold ######
        #print("生成資料的機率分布: ", prediction)
        threshold = torch.tensor([0.5]).cuda()  #閥值設定
        #pred_result = prediction.data.cpu().numpy()
        final = (prediction > threshold).float()*1  # 希望被判斷出來是1
        #print("Final result: ", final)
        result = final.data.cpu().numpy()
        #print("判斷生成資料結果: ", result)
        #print(result)
        
        return error, result

    i = 1 ### count用
    j = 0
    list_count = [] 
    list_GD = [[]]
    test = [] # 重組生成資料用
    auc_class = 0 #計算class編號
    patient = 0
    patient2 = 0
    loss_patient = 0
    #list_D = np.empty((24,200))


    weakpoint = 14 # 類別2
    print("The weak label is: " + str(weakpoint))
        

    #####################********** GAN Parameters  **************##############
                
    x_g=np.copy(x_train[y_train == weakpoint]) #

    # x_g, protocol_x_g, service_x_g, flag_x_g, land_x_g, logged_in_x_g, root_shell_x_g = remove_string_type(x_g)  # 拿掉string type 

    # print(x_g.shape)
    y_g=np.zeros(y_train.shape)  #填0 # y_partial -> y_train
    y_g = y_g[y_train==weakpoint]
    y_g = np.random.uniform(0.9, 1, y_g.shape) # 軟標籤 0.9~1
    #y_g = np.ones(y_g.shape) #硬標籤

    print(y_g)
    print(y_g.shape)
    print(type(y_g))

    #################### x_g 轉換維度 ###################
    x_g = x_g[:,None]
    print(x_g.shape)
    X_g = torch.from_numpy(x_g).type(torch.cuda.FloatTensor) 
    Y_g = torch.from_numpy(y_g).type(torch.cuda.FloatTensor)    
            

    data = [(x, y) for x, y in zip(X_g,Y_g)]
    dataSet=Data_Loader(data)
    #print("dataSet: ", dataSet)  #dataSet是甚麼東西
    data_loader = DataLoader(dataset=dataSet, batch_size=128, shuffle=True) # batch_size=64
    #print("data_loader: ", data_loader)  #data_loader是甚麼

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
            
    discriminator.cuda()
    generator.cuda()
    ###########################################################
    d_optimizer = optim.SGD(discriminator.parameters(), lr=0.0001) # SGD is good for Descriminator
    g_optimizer = optim.SGD(generator.parameters(), lr=0.0001) # Adam is good for Generator

    loss = nn.BCELoss().cuda()  # loss function使用cuda
    ###########################################################

    #*********************************************  Running GAN   ***************************    

    num_epochs = 50000

    start1 = time.time()

        #print(data_loader)
    d_errs=[]
    g_errs=[]
    accuracy=[]  #計算鑑別器準確率(訓練生成器過程)
    D_real_accuracy = [] #計算鑑別器真實資料之準確率(訓練鑑別器過程)
    D_fake_accuracy = [] #計算鑑別器生成資料之誤判率(訓練鑑別器過程)
    D_combine_accuracy = [] #計算鑑別器總準確率(訓練鑑別器過程)
    for epoch in range(num_epochs):
        t = 0 #
        g_error_sum=0
        d_error_sum=0
        acc = 0
        D_training_real_acc = 0
        D_training_fake_acc = 0
        D_combine_acc = 0
        counter = 0
        for n_batch, (real_batch,y_real) in enumerate(data_loader):
            counter += 1
            N = real_batch.size(0)  #回傳real_batch的行數(0表示axis)
            #print("Batch_size: ", N) # N是batch_size 1000
            #print("Length of real_batch: ", len(real_batch))
            #print("Length of real_batch[0]: ", len(real_batch[0]))
            #print("Length of y_real: ", len(y_real))

        # ----------1. Train Discriminator------------------
            real_data = Variable(real_batch)
            #real_target=Variable(real_target)
            # Generate fake data and detach 

            #print(real_data.shape)

            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N)).detach()  # detach()的意義
            #g_fake_data = generator(noise(N))
            #fake_data = g_fake_data.detach()  # detach()的意義

            # Train D
            #y_real=np.array(y_real)
            y_real = y_real.data.cpu().numpy()
            #y_real = torch.from_numpy(y_real).type(torch.cuda.FloatTensor) #GPU
                    
            # 除錯用
            #print(y_real.shape)
            a = np.expand_dims(y_real,axis=1) #增加維度
            aa = np.expand_dims(a,axis=1)
            #print(a.shape)
                    
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data, aa) # y_real -> a
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
            fake_data = generator(noise(N)) #前面訓練 D時已經生成1組fake_data
            # Train G
            g_error, result = train_generator(g_optimizer, fake_data)
            real_result = zeros_target(N)
            real_result = real_result.data.cpu().numpy()
            sub_acc = accuracy_score(real_result, result)
            #g_error = train_generator(g_optimizer, g_fake_data)
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

        d_errs.append(d_sum) # error合併
        #d_errs.append(d) # error分離
        g_errs.append(g_sum) # error合併
        #g_errs.append(g) # error分離
        #a = np.array(d_errs)
        #b = np.array(g_errs)
        #print("Current d_errs", d_errs)
        #print("Current g_errs", g_errs)
            
                        
        # early stopping
        #g_current = g
        #d_current = d
        #print("Epoch:", epoch)
        #print("---------individual---------")
        #print(g_current)
        #print(d_current)
        #print("---------sum-----------")
        #print(g_sum)
        #print(d_sum)
        if (sum_acc < sum_D_training_fake_acc) and epoch<100:                                          ######################################################## 0628修改 ###############################################
            break
        if sum_D_combine_acc < 0.6 and sum_D_combine_acc > 0.4:
            patient += 1
        else:
            patient = 0
        if patient == 1000:
            print("這是收斂點: ", epoch+1)
            print("-----儲存模型-----")
            Gfile_name =  "save_file/" + "thousand_Cpoint_Generator_" + str(epoch+1) + ".pth"
            torch.save(generator, Gfile_name)
            patient = 0
        
        if sum_D_combine_acc < 0.6 and sum_D_combine_acc > 0.4:
            patient2 += 1
        else:
            patient2 = 0
        if patient2 == 100 and epoch >= 2000:
            end1 = time.time()
            a = a =(end1 - start1)
            with open("save_file/GAN_train_sec.txt","a+") as f:
                f.writelines(str(a))
                f.writelines("\n")
            print("這是收斂點: ", epoch+1)
            print("-----儲存模型-----")
            Gfile_name =  "save_file/" + "hundred_Cpoint_Generator_" + str(epoch+1) + ".pth"
            torch.save(generator, Gfile_name)
            patient2 = 0
            end_loop = 0
            with open("save_file/end_loop.txt","a") as f:
                f.writelines(str(end_loop))
        


    ######  以G_loss, D_loss收斂  ########
        if epoch == 0:
            tmp_loss = abs(g_sum - d_sum) #取絕對值
            last_loss = round(tmp_loss,4)
            tmp_g = g_sum
            tmp_d = d_sum
        else:
            tmp_loss = abs(g_sum - d_sum)
            tmp_loss = round(tmp_loss,4)
            
            if abs(tmp_g - g_sum) < 0.05 and abs(tmp_d - d_sum) < 0.05:
                #if tmp_loss == last_loss:
                if abs(tmp_loss - last_loss) <= 0.005: #相鄰兩次loss值相差正負0.0005
                    loss_patient += 1
                    last_loss = tmp_loss
                else:
                    loss_patient = 0
                    last_loss = tmp_loss
                #print("Epoch " + str(epoch) + " : " + str(last_loss))
                if loss_patient == 200:
                    Gfile_name = "save_file/" + "Loss_Generator_" + str(epoch) + ".pth"
                    print("Epoch: " + str(epoch))
                    torch.save(generator, Gfile_name)
                    loss_patient = 0
            else:
                loss_patient = 0

    ###########       儲存圖檔       ###########
        if (epoch+1) % 10000 == 0:   
            print("epoch: ",epoch)
            end2 = time.time()
            Print_loss(d_errs, g_errs, weakpoint, epoch)
            Print_smooth_loss(d_errs, g_errs, weakpoint, epoch)
            Print_G_accuracy(accuracy, weakpoint, epoch)
            Print_D_real_accuracy(D_real_accuracy, weakpoint, epoch)
            Print_smooth_D_real_accuracy(D_real_accuracy, weakpoint, epoch)
            Print_D_fake_accuracy(D_fake_accuracy, weakpoint, epoch)
            Print_smooth_D_fake_accuracy(D_fake_accuracy, weakpoint, epoch)
            Print_D_total_accuracy(D_combine_accuracy, weakpoint, epoch)
            Print_smooth_D_total_accuracy(D_combine_accuracy, weakpoint, epoch)
            Print_smooth_real_fake(D_real_accuracy, D_fake_accuracy, weakpoint, epoch)

            # Save model
            #d_model_name = "Discriminator_" + str(weakpoint) + "_" + str(epoch+1) + ".pth"
            g_model_name = "save_file/" + "Generator_" + str(weakpoint) + "_" + str(epoch+1) + ".pth"
            #torch.save(discriminator, d_model_name)
            torch.save(generator, g_model_name)
            #print("Success save D model: ", d_model_name)
            print("Success save G model: ", g_model_name)
            b =(end2 - start1)
            with open("save_file/GAN_10000_sec.txt","a+") as f:
                f.writelines(str(b))
                f.writelines("\n")
        # if os.path.isfile(fileName):             ############################################### 如果沒檢查到檔案(沒有收斂點) 將early_stop設為1000 (沒意義) 反之設為0 #############################################
        #     early_stop = 1000
        # else:
        #     early_stop = 0
        # if epoch >=15000 and early_stop == 0:    ############################################### 如果epoch >= 10000 但還是沒有拿到收斂點 就終止這個迴圈 ##########################################################
        #     break
    end = time.time()
    print('花費的時間:',round((end - start)/60, 2),'分')
    a = a =(end - start)
    with open("save_file/sec.txt","a+") as f:
            f.writelines(str(a))
            f.writelines("\n")

    fileName = 'save_file/end_loop.txt'
    if os.path.isfile(fileName):
        print('已儲存類別',weakpoint,'的收斂點')
        break