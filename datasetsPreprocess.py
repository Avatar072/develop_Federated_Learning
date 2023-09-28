import warnings
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA

#############################################################################  variable  ###################
filepath = "D:\\Labtest20230911\\data"
#############################################################################  variable  ###################

#############################################################################  funcion宣告與實作  ###########
### for sorting the labeled data based on support
def sortingFunction(data):
    return data.shape[0]

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

# ## 清除CIC-IDS数据集中的脏数据，第一行特征名称和包含NaN、Infinity、包含空白或小于ASCII 32的字符等数据的行数
# def clearDirtyData(df):
#     # 检查第一列特征名中是否包含空白或第一个字符小于ASCII 32
#     first_column = df.columns[0]
#     is_dirty = first_column.isspace() or ord(first_column[0]) < 32

#     if (df == 'Nan').any().any() or (df == 'Infinity').any().any() or (df == 'inf').any().any():
#         # 找到包含NaN或Infinity值的行，并将其索引添加到dropList
#         dropList = df[is_dirty | (df == 'Nan').any(axis=1) | (df == 'Infinity').any(axis=1) | (df == 'inf').any(axis=1)].index.tolist()
#     else:
#         # 处理列不存在NaN或Infinity值的情况
#         dropList = []

#     return dropList

def clearDirtyData(df):
    # 检查第一列特征名中是否包含空白或第一个字符小于ASCII 32
    first_column = df.columns[0]
    is_dirty = first_column.isspace() or ord(first_column[0]) < 32

    # 将"inf"值替换为NaN
    df.replace("inf", np.nan, inplace=True)

    # 找到包含NaN、Infinity和"inf"值的行，并将其索引添加到dropList
    nan_inf_rows = df[df.isin([np.nan, np.inf, -np.inf]).any(axis=1)].index.tolist()

    # 将第一行特征名称所在的索引添加到dropList
    if is_dirty:
        nan_inf_rows.append(0)

    # 去重dropList中的索引
    dropList = list(set(nan_inf_rows))

    # 删除包含脏数据的行
    df_clean = df.drop(dropList)

    return df_clean



### 检查CSV文件是否存在，如果不存在，则合并数据并保存到CSV文件中
def CheckCsvFileIsexists(file):
    if not os.path.exists(file):
        # 如果文件不存在，执行数据合并    
        filepath
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


### 标签编码用于名义值
def label_Encoding(label):
    label_encoder = preprocessing.LabelEncoder()
    mergecompelete_dataset[label] = label_encoder.fit_transform(mergecompelete_dataset[label])
    mergecompelete_dataset[label].unique()


def findStringCloumn(dataFrame):
        string_columns = dataFrame.select_dtypes(include=['object'])
        for column in string_columns.columns:
            print(f"{dataFrame} 中数据类型为 'object' 的列: {column}")
            print(string_columns[column].value_counts())
            print("\n")

### print information 
def printFeatureCountAndLabelCount(dataFrame1, dataFrame2):
    num_features_train_half1 = dataFrame1.shape[1] - 1
    label_counts = dataFrame1['Label'].value_counts()
    print("Train_df_half1 的特征数:", num_features_train_half1)
    print("Train_df_half1 的label数:", len(label_counts))
    print("Train_df_half1 的除了最后一列标签列之外的所有列,即选择特征数据:\n", dataFrame1.iloc[:,:-1])
    findStringCloumn(dataFrame1)

    num_features_train_half2 = dataFrame2.shape[1] - 1 
    label_counts2 = dataFrame2['Label'].value_counts()
    print("Train_df_half2 的特征数:", num_features_train_half2)
    print("Train_df_half2 的label数:", len(label_counts2))
    print("Train_df_half2 的除了最后一列标签列之外的所有列,即选择特征数据:\n", dataFrame2.iloc[:,:-1])
    findStringCloumn(dataFrame2)
    
### Save np 

# 指定要创建的文件夹名称
def generatefolder():
    folder_name = "my_AnalyseReportfolder"
    # 使用os.path.exists()检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 如果文件夹不存在，就创建它
        os.makedirs(folder_name)
        print(f"文件夹 '{folder_name}' 已创建。")
    else:
        print(f"文件夹 '{folder_name}' 已存在，无需创建。")

### label Encoding And Replace the number of greater than 10,000
def LabelEncodingAndReplaceMorethanTenthousandQuantity(df):
    # 将 CSV 中的所有列中的 � 替换为空白（使用 Unicode 编码）
    # label Encoding
    df['Label'] = df['Label'].str.replace('\ufffd', '', regex=False)
    label_Encoding('Label')
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
    extracted_df.to_csv(filepath + '\\total_encoded_updated.csv', index=False)

    # 打印修改后的结果
    print(extracted_df['Label'].value_counts())


def DoTrainAndTestChaneTypeTonpArray():
    #择除了最后一列标签列之外的所有列，即选择特征数据
    x_train_half1 = np.array(train_df_half1.iloc[:,:-1])
    y_train_half1 = np.array(train_df_half1.iloc[:,-1])

    x_train_half2 = np.array(train_df_half2.iloc[:,:-1])
    y_train_half2 = np.array(train_df_half2.iloc[:,-1])

    x_test = np.array(test_dataframes.iloc[:,:-1])
    y_test = np.array(test_dataframes.iloc[:,-1])   

    #np.save
    np.save('x_train_half1.npy', x_train_half1)
    np.save('x_train_half2.npy', x_train_half2)
    np.save('y_train_half1.npy', y_train_half1)
    np.save('y_train_half2.npy', y_train_half2)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

    # Save train data
    train_df_half1.to_csv(filepath + "\\train_df_half1.csv", index=False)
    train_df_half2.to_csv(filepath + "\\train_df_half2.csv", index=False)

    


# CheckCsvFileIsexists檢查file存不存在，若file不存在產生新檔
CheckCsvFileIsexists(filepath + "\\total_original.csv")
# 加载合并后的数据集
mergecompelete_dataset = pd.read_csv(filepath + "\\total_original.csv")
# label Encoding And generate MapTable
LabelEncodingAndReplaceMorethanTenthousandQuantity(mergecompelete_dataset)
# 加载label_Encoding后的数据集
# mergecompelete_dataset = pd.read_csv(filepath + "\\total_encoded.csv")
mergecompelete_dataset = pd.read_csv(filepath + "\\total_encoded_updated.csv")

### extracting features
X=mergecompelete_dataset.iloc[:,:-1]
X=X.values
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


scaler = preprocessing.StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

### do PCA
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

# split各一半
train_dataframes, test_dataframes = train_test_split(mergecompelete_dataset, test_size=0.4, random_state=42)#test_size=0.4表示将数据集分成测试集的比例为40%

train_dataframes.to_csv(filepath + "\\train_dataframes.csv", index=False)
test_dataframes.to_csv(filepath + "\\test_dataframes.csv", index=False)

common_indices = np.intersect1d(train_dataframes.index, test_dataframes.index)
print("train_dataframes 和 test_dataframes 的索引交集数量:", len(common_indices))
print("train_dataframes 和 test_dataframes 的是否相同:", train_dataframes.equals(test_dataframes)) 
# 获取整个数据集的行数
mergecompelete_dataset_total_rows = len(mergecompelete_dataset)
print("整個資料集的行数\n",mergecompelete_dataset_total_rows)
train_df_half1, train_df_half2 = train_test_split(train_dataframes, test_size=0.5)


# 找到train_df_half1和train_df_half2中重复的行
duplicates = train_df_half2[train_df_half2.duplicated(keep=False)]

# 删除train_df_half2中与train_df_half1重复的行
train_df_half2 = train_df_half2[~train_df_half2.duplicated(keep=False)]




printFeatureCountAndLabelCount(train_df_half1, train_df_half2)




# 确保 train_df_half1 和 train_df_half2 中没有重叠的样本
intersection = len(set(train_df_half1.index) & set(train_df_half2.index))
print("train_df_half1 和 train_df_half2 的索引交集数量:", intersection)
print("train_df_half1 和 train_df_half2 的是否相同:", train_df_half1.equals(train_df_half2)) 

