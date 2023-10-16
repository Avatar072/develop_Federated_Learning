import os
import pandas as pd
import numpy as np
import argparse


### 生成列名列表
column_names = ["principal_Component" + str(i) for i in range(1, 79)] + ["Label"]

### 檢查資料夾是否存在 回傳True表示沒存在
def CheckFolderExists (folder_name):
    if not os.path.exists(folder_name):
        return True
    else:
        return False
    
### 檢查檔案是否存在
def CheckFileExists (file):
   if os.path.isfile(file):
    print(f"{file} 是一個存在的檔案。")
    return True
   else:
    print(f"{file} 不是一個檔案或不存在。")
    return False
    
### Save data to csv
def SaveDataToCsvfile(df, folder_name, filename):
    # 抓取當前工作目錄名稱
    current_directory = os.getcwd()
    print("當前工作目錄", current_directory)
    # folder_name = filename + "_folder"
    print("資料夾名稱", folder_name)
    generatefolder(folder_name)
    csv_filename = os.path.join(current_directory, 
                                folder_name, filename + ".csv")
    print("存檔位置跟檔名", csv_filename)
    df.to_csv(csv_filename, index=False)

### 建立一個資料夾
def generatefolder(folder_name):
    if folder_name is None:
        folder_name = "my_AnalyseReportfolder"

    file_not_exists  = CheckFolderExists(folder_name)
    print("file_not_exists",file_not_exists)
    # 使用os.path.exists()檢文件夹是否存在
    if file_not_exists:
        # 如果文件夹不存在，就创建它
        os.makedirs(folder_name)
        print(f"資料夾 '{folder_name}' 創建。")
    else:
        print(f"資料夾 '{folder_name}' 已存在，不需再創建。")

### 合併DataFrame成csv
def mergeDataFrameAndSaveToCsv(trainingtype, x_train,y_train, filename, epochs):
    # 创建两个DataFrame分别包含x_train和y_train
    df_x_train = pd.DataFrame(x_train)
    df_y_train = pd.DataFrame(y_train)

    # 使用concat函数将它们合并
    generateNewdata = pd.concat([df_x_train, df_y_train], axis=1)

    # 保存合并后的DataFrame为CSV文件
    if trainingtype == "GAN":
        generateNewdata.columns = column_names
        SaveDataToCsvfile(generateNewdata, f"{trainingtype}_data_{filename}", f"{trainingtype}_data_{filename}_epochs_{epochs}")
    else:
        SaveDataToCsvfile(generateNewdata, f"{filename}_epochs_{epochs}")

def ParseCommandLineArgs(commands):
    
    # e.g
    # python BaseLine.py -h
    # python BaseLine.py --dataset train_half1
    # python BaseLine.py --dataset train_half2
    # python BaseLine.py --epochs 100
    # python BaseLine.py --dataset train_half1 --epochs 100
    # python DoGAN.py --dataset train_half1 --epochs 10 --weaklabel 8
    # default='train_half1'
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Federated Learning Client')

    # 添加一个参数来选择数据集
    parser.add_argument('--dataset', type=str, choices=['train_half1', 'train_half2'], default='train_half1',
                        help='Choose the dataset for training (train_half1 or train_half2)')

    # 添加一个参数来设置训练的轮数
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

     # 添加一个参数来设置训练的轮数
    parser.add_argument('--weaklabel', type=int, default=8, help='encode of weak label')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据输入的命令列表来确定返回的参数
    if 'dataset' in commands and 'epochs' in commands and 'weaklabel' in commands:
        return args.dataset, args.epochs, args.weaklabel
    elif 'dataset' in commands and 'epochs' in commands:
        return args.dataset, args.epochs
    elif 'dataset' in commands:
        return args.dataset
    elif 'epochs' in commands:
        return args.epochs

# 测试不同的命令
print(ParseCommandLineArgs(['dataset']))
print(ParseCommandLineArgs(['epochs']))
print(ParseCommandLineArgs(['dataset', 'epochs']))
print(ParseCommandLineArgs(['dataset', 'epochs', 'label']))

def ChooseTrainDatastes(filepath, my_command):
    # 加载选择的数据集
    if my_command == 'train_half1':
        print("Training with train_half1")
        train_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'train_df_half1.csv'))
        x_train = np.array(train_dataframe.iloc[:, :-1])
        y_train = np.array(train_dataframe.iloc[:, -1])
        client_str = "client1"
        
    elif my_command == 'train_half2':
        print("Training with train_half2")
        train_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'train_df_half2.csv'))
        x_train = np.array(train_dataframe.iloc[:, :-1])
        y_train = np.array(train_dataframe.iloc[:, -1])
        client_str = "client2"
        
    
    # 返回所需的數據或其他變量
    return x_train, y_train, client_str


def ChooseTestDataSet(filepath):
    test_dataframe = pd.read_csv(os.path.join(filepath, 'data', 'test_dataframes.csv'))
    x_test = np.array(test_dataframe.iloc[:, :-1])
    y_test = np.array(test_dataframe.iloc[:, -1])
    
    return x_test, y_test
