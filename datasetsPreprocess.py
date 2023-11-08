import warnings
import os
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mytoolfunction import SaveDataToCsvfile,printFeatureCountAndLabelCountInfo
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,spiltweakLabelbalance,SaveDataframeTonpArray,generatefolder

#############################################################################  variable  ###################
filepath = "D:\\Labtest20230911\\data"

today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", today)
#############################################################################  variable  ###################

#############################################################################  funcion宣告與實作  ###########

# 加载CICIDS 2017数据集
def writeData(file_path):
    # 读取CSV文件并返回DataFrame
    df = pd.read_csv(file_path,encoding='cp1252',low_memory=False)
    return df

### merge多個DataFrame
def mergeData(folder_path):
    # 创建要合并的DataFrame列表
    dataframes_to_merge = []

    # 添加每个CSV文件的DataFrame到列表
    dataframes_to_merge.append(writeData(folder_path + "\\Monday-WorkingHours.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Friday-WorkingHours-Morning.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Tuesday-WorkingHours.pcap_ISCX.csv"))
    dataframes_to_merge.append(writeData(folder_path + "\\Wednesday-workingHours.pcap_ISCX.csv"))

    # 检查特征名是否一致
    if check_column_names(dataframes_to_merge):
        # 特征名一致，可以进行合并
        result = pd.concat(dataframes_to_merge)
        # 使用clearDirtyData函数获取要删除的行的索引列表
        result = clearDirtyData(result)
        
        # 使用DataFrame的drop方法删除包含脏数据的行
        #result = result.drop(list_to_drop)
        return result
    else:
        # 特征名不一致，需要处理这个问题
        print("特征名不一致，请检查并处理特征名一致性")
        return None

### 检查要合并的多个DataFrame的特征名是否一致
def check_column_names(dataframes):
    # 获取第一个DataFrame的特征名列表
    reference_columns = list(dataframes[0].columns)

    # 检查每个DataFrame的特征名是否都与参考特征名一致
    for df in dataframes[1:]:
        if list(df.columns) != reference_columns:
            return False

    return True


### 检查CSV文件是否存在，如果不存在，则合并数据并保存到CSV文件中
def ChecktotalCsvFileIsexists(file):
    if not os.path.exists(file):
        # 如果文件不存在，执行数据合并    
        # data = mergeData("D:\\Labtest20230911\\data\\MachineLearningCVE")
        data = mergeData(filepath + "\\TrafficLabelling")#完整的資料
        
        data = clearDirtyData(data)
       
        if data is not None:
            # 去除特征名中的空白和小于ASCII 32的字符
            data.columns = data.columns.str.replace(r'[\s\x00-\x1F]+', '', regex=True)
            # 保存到CSV文件，同时将header设置为True以包括特征名行
            data.to_csv(file, index=False, header=True)
            last_column_index = data.shape[1] - 1
            Label_counts = data.iloc[:, last_column_index].value_counts()
            print(Label_counts)
            print(f"共有 {len(Label_counts)} 个不同的标签")
            print("mergeData complete")
    else:
        print(f"文件 {file} 已存在，不执行合并和保存操作。")


### label encoding
# def label_Encoding(label):
#     label_encoder = preprocessing.LabelEncoder()
#     mergecompelete_dataset[label] = label_encoder.fit_transform(mergecompelete_dataset[label])
#     mergecompelete_dataset[label].unique()

### do Label Encoding
def DoLabelEncoding(df):
    #將 Source IP、Source Port、Destination IP、Destination Port、Timestamp 等特徵做 one-hot 形式轉換
    # Flow ID	 Source IP	 Source Port	 Destination IP	 Destination Port	 Protocol	 Timestamp
    df['FlowID'] = df['FlowID'].str.strip()
    df = df.drop('FlowID', axis=1)
    df['Label'] = df['Label'].str.strip()
    df['SourceIP'] = df['SourceIP'].str.strip()
    df['SourcePort'] = df['SourcePort'].str.strip()
    df['DestinationIP'] = df['DestinationIP'].str.strip()
    df['Protocol'] = df['Protocol'].str.strip()
    df['Timestamp'] = df['Timestamp'].str.strip()
    label_Encoding('Label',df)
    label_Encoding('SourceIP',df)
    label_Encoding('SourcePort',df)
    label_Encoding('DestinationIP',df)
    label_Encoding('DestinationPort',df)
    label_Encoding('Protocol',df)
    label_Encoding('Timestamp',df)
    # 保存編碼后的 DataFrame 回到 CSV 文件
    df.to_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv", index=False)
    return df
    
### label Encoding And Replace the number of greater than 10,000
def ReplaceMorethanTenthousandQuantity(df):
  
    # 超過提取10000行的只取10000，其餘保留 
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv")
    # 获取每个标签的出现次数
    label_counts = df['Label'].value_counts()
    # 打印提取后的DataFrame
    print(label_counts)
    # 创建一个空的DataFrame来存储结果
    extracted_df = pd.DataFrame()

    # 获取所有不同的标签
    unique_labels = df['Label'].unique()

    # 遍历每个标签
    for label in unique_labels:
        # 选择特定标签的行
        label_df = df[df['Label'] == label]
    
        # 如果标签的数量超过1万，提取前1万行；否则提取所有行
        if len(label_df) > 10000:
            label_df = label_df.head(10000)
    
        # 将结果添加到提取的DataFrame中
        extracted_df = pd.concat([extracted_df, label_df])

    # 将更新后的DataFrame保存到文件
    SaveDataToCsvfile(extracted_df, "./data/dataset_AfterProcessed","total_encoded_updated_10000")

    # 打印修改后的结果
    print(extracted_df['Label'].value_counts())


# CheckCsvFileIsexists檢查file存不存在，若file不存在產生新檔
ChecktotalCsvFileIsexists(filepath + "\\dataset_AfterProcessed\\total_original.csv")
# Loading datasets after megre complete
mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_original.csv")
mergecompelete_dataset = DoLabelEncoding(mergecompelete_dataset)
ReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
# Loading datasets after label_Encoding
mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded_updated_10000.csv")

### extracting features
mergecompelete_dataset=mergecompelete_dataset.iloc[:,:-1]
# 列出要排除的列名
columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
# 使用条件选择不等于这些列名的列
mergecompelete_dataset = mergecompelete_dataset[[col for col in mergecompelete_dataset.columns if col not in columns_to_exclude]]
mergecompelete_dataset=mergecompelete_dataset.values
# scaler = preprocessing.StandardScaler()# 資料標準化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(mergecompelete_dataset)
scaler.fit(mergecompelete_dataset)
mergecompelete_dataset=scaler.transform(mergecompelete_dataset)
# 保留MinMaxScaler後的結果
SaveDataToCsvfile(mergecompelete_dataset, "./data/dataset_AfterProcessed","total_encoded_updated_10000")

# ### Do PCA
# number_of_components=78 # 改成跟資料集feature一樣
# pca = PCA(n_components=number_of_components)
# columns_array=[]
# for i in range (number_of_components):
#     columns_array.append("principal_Component"+str(i+1))
    
# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents
#               , columns = columns_array)

# finalDf = pd.concat([principalDf, mergecompelete_dataset[['Label']]], axis = 1)
# mergecompelete_dataset=finalDf

# split mergecompelete_dataset
train_dataframes, test_dataframes = train_test_split(mergecompelete_dataset, test_size=0.2, random_state=42)#test_size=0.4表示将数据集分成测试集的比例为40%
printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes)


# train_df_half1, train_df_half2 = train_test_split(train_dataframes, test_size=0.5)
# 分別取出Label等於8、9、13、14的數據 對半分
train_label_8, test_label_8 = spiltweakLabelbalance(8,mergecompelete_dataset,0.4)
train_label_9, test_label_9 = spiltweakLabelbalance(9,mergecompelete_dataset,0.5)
train_label_13, test_label_13 = spiltweakLabelbalance(13,mergecompelete_dataset,0.5)
train_label_14, test_label_14 = spiltweakLabelbalance(14,mergecompelete_dataset,0.5)

# 刪除Label相當於8、9、13、14的行
test_dataframes = test_dataframes[~test_dataframes['Label'].isin([8, 9,13, 14])]
train_dataframes = train_dataframes[~train_dataframes['Label'].isin([8, 9,13,14])]
# 合併Label8、9、13、14回去
test_dataframes = pd.concat([test_dataframes, test_label_8, test_label_9, test_label_13, test_label_14])
train_dataframes = pd.concat([train_dataframes,train_label_8, train_label_9,train_label_13,train_label_14])

label_counts = test_dataframes['Label'].value_counts()
print("test_dataframes\n", label_counts)
label_counts = train_dataframes['Label'].value_counts()
print("train_dataframes\n", label_counts)

# split train_dataframes各一半
train_half1,train_half2 = splitdatasetbalancehalf(train_dataframes)

# 找到train_df_half1和train_df_half2中重复的行
duplicates = train_half2[train_half2.duplicated(keep=False)]

# 删除train_df_half2中与train_df_half1重复的行
train_df_half2 = train_half2[~train_half2.duplicated(keep=False)]

# train_df_half1和train_df_half2 detail information
printFeatureCountAndLabelCountInfo(train_half1, train_df_half2)

SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/{today}", f"train_dataframes_{today}")
SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/{today}", f"test_dataframes_{today}")
SaveDataToCsvfile(train_half1, f"./data/dataset_AfterProcessed/{today}", f"train_half1_{today}")
SaveDataToCsvfile(train_half2,  f"./data/dataset_AfterProcessed/{today}", f"train_half2_{today}") 

SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/{today}", "test",today)
SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/{today}", "train",today)
SaveDataframeTonpArray(train_half1, f"./data/dataset_AfterProcessed/{today}", "train_half1", today)
SaveDataframeTonpArray(train_half2, f"./data/dataset_AfterProcessed/{today}", "train_half2", today)


