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
from sklearn.preprocessing import OneHotEncoder
from mytoolfunction import SaveDataToCsvfile,printFeatureCountAndLabelCountInfo
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,spiltweakLabelbalance,SaveDataframeTonpArray,generatefolder
from mytoolfunction import spiltweakLabelbalance_afterOnehot

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
    # df = pd.read_csv(file_path)
    # 找到不包含NaN、Infinity和"inf"值的行
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
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
        
        # data = clearDirtyData(data)
       
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
def label_Encoding(label):
    label_encoder = preprocessing.LabelEncoder()
    mergecompelete_dataset[label] = label_encoder.fit_transform(mergecompelete_dataset[label])
    mergecompelete_dataset[label].unique()

def OneHot_Encoding(feature, data):
    # 建立 OneHotEncoder 實例
    onehotencoder = OneHotEncoder()
    # 使用 OneHotEncoder 進行 One-Hot 編碼
    onehot_encoded_data = onehotencoder.fit_transform(data[[feature]]).toarray()
     # 將 One-Hot 編碼的結果轉換為 DataFrame，並指定欄位名稱
    onehot_encoded_data = pd.DataFrame(onehot_encoded_data, columns=[f"{feature}_{i}" for i in range(onehot_encoded_data.shape[1])])
    # 合併原始 DataFrame 與 One-Hot 編碼的結果
    data = pd.concat([data, onehot_encoded_data], axis=1)
    # 刪除原始特徵列
    data = data.drop(feature, axis=1)
    return data
### do Label Encoding
# def DoLabelEncoding(df):
#     #將 Source IP、Source Port、Destination IP、Destination Port、Timestamp 等特徵做 one-hot 形式轉換
#     # Flow ID	 Source IP	 Source Port	 Destination IP	 Destination Port	 Protocol	 Timestamp
#     df = df.drop('FlowID', axis=1)
#     label_Encoding('Label',df)
#     label_Encoding('SourceIP',df)
#     label_Encoding('SourcePort',df)
#     label_Encoding('DestinationIP',df)
#     label_Encoding('DestinationPort',df)
#     label_Encoding('Protocol',df)
#     label_Encoding('Timestamp',df)
    # 保存編碼后的 DataFrame 回到 CSV 文件
    # df.to_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv", index=False)
    # return df
    
### label Encoding And Replace the number of greater than 10,000
def ReplaceMorethanTenthousandQuantity(df):
  
    # 超過提取10000行的只取10000，其餘保留 
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv")
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_original.csv")
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
    # SaveDataToCsvfile(extracted_df, "./data/dataset_AfterProcessed","total_encoded_updated_10000")

    # 打印修改后的结果
    print(extracted_df['Label'].value_counts())
    return extracted_df


# CheckCsvFileIsexists檢查file存不存在，若file不存在產生新檔
ChecktotalCsvFileIsexists(filepath + "\\dataset_AfterProcessed\\total_original.csv")
# Loading datasets after megre complete
mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_original.csv")
# DoLabelEncoding(mergecompelete_dataset)
mergecompelete_dataset = ReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
mergecompelete_dataset = mergecompelete_dataset.drop('FlowID', axis=1)
label_Encoding('SourceIP')
label_Encoding('SourcePort')
label_Encoding('DestinationIP')
label_Encoding('DestinationPort')
label_Encoding('Protocol')
label_Encoding('Timestamp')
label_Encoding('Label')
# mergecompelete_dataset.to_csv(filepath + "\\dataset_AfterProcessed\\total_encoded_updated_10000.csv", index=False)
mergecompelete_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded_updated_10000.csv")


### extracting features
#除了Label外的特徵
crop_dataset=mergecompelete_dataset.iloc[:,:-1]
# 列出要排除的列名
columns_to_exclude = ['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp']
# 使用条件选择不等于这些列名的列
doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
# print(doScalerdataset.info)
# print(mergecompelete_dataset.info)
# print(undoScalerdataset.info)
X=doScalerdataset
X=X.values
# scaler = preprocessing.StandardScaler() #資料標準化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler.fit(X)
X=scaler.transform(X)


## 重新合並MinMax後的特徵
number_of_components=77 # 原84個的特徵，扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp' 'Label' | 84-7 =77
columns_array=[]
for i in range (number_of_components):
    columns_array.append("principal_Component"+str(i+1))
    
principalComponents = X
principalDf = pd.DataFrame(data = principalComponents
              , columns = columns_array)

finalDf = pd.concat([undoScalerdataset,principalDf, mergecompelete_dataset[['Label']]], axis = 1)
print(finalDf)
mergecompelete_dataset=finalDf
# 保留MinMaxScaler後的結果
# SaveDataToCsvfile(mergecompelete_dataset, "./data/dataset_AfterProcessed","total_encoded_updated_10000_After_minmax")

# 做one hot
mergecompelete_dataset = OneHot_Encoding('SourceIP', mergecompelete_dataset)
mergecompelete_dataset = OneHot_Encoding('SourcePort', mergecompelete_dataset)
mergecompelete_dataset = OneHot_Encoding('DestinationIP', mergecompelete_dataset)
mergecompelete_dataset = OneHot_Encoding('DestinationPort', mergecompelete_dataset)
mergecompelete_dataset = OneHot_Encoding('Protocol', mergecompelete_dataset)
mergecompelete_dataset = OneHot_Encoding('Timestamp', mergecompelete_dataset)

# mergecompelete_dataset = OneHot_Encoding('Label', mergecompelete_dataset)
SaveDataToCsvfile(mergecompelete_dataset, "./data/dataset_AfterProcessed","total_encoded_updated_10000_After_minmax_onehot")

# split mergecompelete_dataset
train_dataframes, test_dataframes = train_test_split(mergecompelete_dataset, test_size=0.2, random_state=42)#test_size=0.4表示将数据集分成测试集的比例为40%
# printFeatureCountAndLabelCountInfo(train_dataframes, test_dataframes)


# Label encode mode  分別取出Label等於8、9、13、14的數據 對半分
train_label_8, test_label_8 = spiltweakLabelbalance(8,mergecompelete_dataset,0.4)
train_label_9, test_label_9 = spiltweakLabelbalance(9,mergecompelete_dataset,0.5)
train_label_13, test_label_13 = spiltweakLabelbalance(13,mergecompelete_dataset,0.5)
train_label_14, test_label_14 = spiltweakLabelbalance(14,mergecompelete_dataset,0.5)

# # 刪除Label相當於8、9、13、14的行
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
###########################################################one hot mode################################################################################################
# one hot mode 分別取出Label等於8、9、13的數據 對半分
# def ifspiltweakLabelbalance_AfterOneHot(test_dataframes,train_dataframes):
#     test_label_8,train_label_8 = spiltweakLabelbalance_afterOnehot('Label_8',mergecompelete_dataset,0.5)
#     test_label_9,train_label_9  = spiltweakLabelbalance_afterOnehot('Label_9',mergecompelete_dataset,0.5)
#     test_label_13,train_label_13   = spiltweakLabelbalance_afterOnehot('Label_13',mergecompelete_dataset,0.5)
#     # 取Label不是於Label_8、Label_9、Label_13的列
#     test_dataframes = test_dataframes[(test_dataframes['Label_8'] != 1) & 
#                                       (test_dataframes['Label_9'] != 1) &
#                                       (test_dataframes['Label_13'] != 1)
#                                       ]
    
#     train_dataframes = train_dataframes[(train_dataframes['Label_8'] != 1) & 
#                                         (train_dataframes['Label_9'] != 1) &
#                                         (train_dataframes['Label_13'] != 1)
#                                         ]
    
#     #存回原本的test_dataframes和train_dataframes
#     test_dataframes = pd.concat([test_dataframes, test_label_8, test_label_9, test_label_13])
#     train_dataframes = pd.concat([train_dataframes,train_label_8, train_label_9,train_label_13])
#     # 保存新的 DataFrame 到文件
#     test_dataframes.to_csv("./data/test_test.csv", index=False)
#     train_dataframes.to_csv("./data/test_train.csv", index=False)

#     return test_dataframes, train_dataframes



# def pintLabelcountAfterOneHot(dfname1,dfname2,test_dataframes,train_dataframes):
#     for i in range(0,15):
#         print(f"{str(dfname1)} Label_{i} count",len(test_dataframes[test_dataframes[f'Label_{i}'] == 1]))
#         print(f"{str(dfname2)} Label_{i} count",len(train_dataframes[train_dataframes[f'Label_{i}'] == 1]))
    
# def spilt_half_train_dataframes_AfterOneHot(train_dataframes):
#     df = pd.DataFrame(train_dataframes)

#     # 初始化兩個 DataFrame 以存儲結果
#     train_half1 = pd.DataFrame()
#     train_half2 = pd.DataFrame()

#     # 分割每個標籤
#     for i in range(0,15):
#         label_name = f'Label_{i}'
#         label_data = df[label_name]
#         label_half1, label_half2 = train_test_split(df[label_data == 1], test_size=0.5, random_state=42)

#         # 將每個標籤的一半添加到對應的 DataFrame
#         train_half1 = pd.concat([train_half1, label_half1], axis=0)
#         train_half2 = pd.concat([train_half2, label_half2], axis=0)

#     # 打印存儲結果
#     # print("train_half1:\n", train_half1)
#     # print("\ntrain_half2:\n", train_half2)
#     train_half1.to_csv("./data/train_half1.csv", index=False)
#     train_half2.to_csv("./data/train_half2.csv", index=False)
    
#     return train_half1, train_half2


# test_dataframes, train_dataframes = ifspiltweakLabelbalance_AfterOneHot(test_dataframes,train_dataframes)
# train_half1, train_half2 = spilt_half_train_dataframes_AfterOneHot(train_dataframes)
# pintLabelcountAfterOneHot("test_dataframes","train_dataframes",test_dataframes,train_dataframes)
# pintLabelcountAfterOneHot("train_half1","train_half2",train_half1,train_half2)
# SaveDataToCsvfile(train_dataframes, f"./data/dataset_AfterProcessed/{today}", f"train_dataframes_{today}")
# SaveDataToCsvfile(test_dataframes,  f"./data/dataset_AfterProcessed/{today}", f"test_dataframes_{today}")
# SaveDataToCsvfile(train_half1, f"./data/dataset_AfterProcessed/{today}", f"train_half1_{today}")
# SaveDataToCsvfile(train_half2,  f"./data/dataset_AfterProcessed/{today}", f"train_half2_{today}") 

# SaveDataframeTonpArray(test_dataframes, f"./data/dataset_AfterProcessed/{today}", "test",today)
# SaveDataframeTonpArray(train_dataframes, f"./data/dataset_AfterProcessed/{today}", "train",today)
# SaveDataframeTonpArray(train_half1, f"./data/dataset_AfterProcessed/{today}", "train_half1", today)
# SaveDataframeTonpArray(train_half2, f"./data/dataset_AfterProcessed/{today}", "train_half2", today)


