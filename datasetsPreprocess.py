import warnings
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from mytoolfunction import SaveDataToCsvfile,printFeatureCountAndLabelCountInfo
from mytoolfunction import clearDirtyData,label_Encoding,splitdatasetbalancehalf,spiltweakLabelbalance,SaveDataframeTonpArray

#############################################################################  variable  ###################
filepath = "D:\\Labtest20230911\\data"
#############################################################################  variable  ###################

#############################################################################  funcion宣告與實作  ###########

# 加载CICIDS 2017数据集
def writeData(file_path):
    # 读取CSV文件并返回DataFrame
    df = pd.read_csv(file_path)
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
        data = mergeData(filepath + "\\MachineLearningCVE")
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

    
### label Encoding And Replace the number of greater than 10,000
def LabelEncodingAndReplaceMorethanTenthousandQuantity(df):
    # 将 CSV 中的所有列中的 � 替换为空白（使用 Unicode 编码）
    # label Encoding
    df['Label'] = df['Label'].str.replace('\ufffd', '', regex=False)
    label_Encoding('Label',df)
    # df['DestinationPort'] = df['DestinationPort'].astype(str)# 這邊Label encoding失敗，DestinationPort拉出來到mytoolfunction.py單獨從做一次
    # label_Encoding('DestinationPort',df)
    # 保存編碼后的 DataFrame 回到 CSV 文件
    df.to_csv(filepath + "\\total_encoded.csv", index=False)
  
    # 超過提取10000行的只取10000，其餘保留 
    df_temp = pd.read_csv(filepath + "\\total_encoded.csv")
    # 获取每个标签的出现次数
    label_counts = df_temp['Label'].value_counts()
    # 打印提取后的DataFrame
    print(label_counts)
    # 创建一个空的DataFrame来存储结果
    extracted_df = pd.DataFrame()

    # 获取所有不同的标签
    unique_labels = df_temp['Label'].unique()

    # 遍历每个标签
    for label in unique_labels:
        # 选择特定标签的行
        label_df = df_temp[df_temp['Label'] == label]
    
        # 如果标签的数量超过1万，提取前1万行；否则提取所有行
        if len(label_df) > 10000:
            label_df = label_df.head(10000)
    
        # 将结果添加到提取的DataFrame中
        extracted_df = pd.concat([extracted_df, label_df])

    # 将更新后的DataFrame保存到文件
    SaveDataToCsvfile(extracted_df, "./data","total_encoded_updated")

    # 打印修改后的结果
    print(extracted_df['Label'].value_counts())


# CheckCsvFileIsexists檢查file存不存在，若file不存在產生新檔
ChecktotalCsvFileIsexists(filepath + "\\total_original.csv")
# Loading datasets after megre complete
mergecompelete_dataset = pd.read_csv(filepath + "\\total_original.csv")
# label Encoding And generate MapTable
LabelEncodingAndReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
# Loading datasets after label_Encoding
mergecompelete_dataset = pd.read_csv(filepath + "\\total_encoded_updated.csv")

### extracting features
X=mergecompelete_dataset.iloc[:,:-1]
X=X.values
scaler = preprocessing.StandardScaler()# 資料標準化
scaler.fit(X)
X=scaler.transform(X)

### Do PCA
number_of_components=78 # 改成跟資料集feature一樣
pca = PCA(n_components=number_of_components)
columns_array=[]
for i in range (number_of_components):
    columns_array.append("principal_Component"+str(i+1))
    
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
              , columns = columns_array)

finalDf = pd.concat([principalDf, mergecompelete_dataset[['Label']]], axis = 1)
mergecompelete_dataset=finalDf

# split mergecompelete_dataset各一半
train_dataframes, test_dataframes = train_test_split(mergecompelete_dataset, test_size=0.2, random_state=42)#test_size=0.4表示将数据集分成测试集的比例为40%

# SaveDataToCsvfile(train_dataframes, "./data","train_dataframes")
# SaveDataToCsvfile(test_dataframes, "./data","test_dataframes")
# train_dataframes和test_dataframes information
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

SaveDataToCsvfile(train_dataframes, "./data", "train_dataframes_20231101")
SaveDataToCsvfile(test_dataframes,  "./data", "test_dataframes_20231101")
SaveDataToCsvfile(train_half1, "./data", "train_half1_20231101")
SaveDataToCsvfile(train_half2,  "./data", "train_half2_20231101") 
SaveDataframeTonpArray(test_dataframes, "test", "20231101")
SaveDataframeTonpArray(train_half1, "train_half1", "20231101")
SaveDataframeTonpArray(train_half2, "train_half2", "20231101")


