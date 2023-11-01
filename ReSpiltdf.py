import warnings
import random
import os
import time
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from mytoolfunction import SaveDataToCsvfile, SaveDataframeTonpArray,generatefolder

#############################################################################  variable  ###################
filepath = "D:\\Labtest20230911\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data產生天日期的資料夾
generatefolder(filepath + "\\", today)
# Loading datasets after label_Encoding
mergecompelete_dataset = pd.read_csv(filepath + "\\total_encoded_updated.csv")

### extracting features
X=mergecompelete_dataset.iloc[:,:-1]
X=X.values
scaler = preprocessing.StandardScaler()
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

label_counts = mergecompelete_dataset['Label'].value_counts()
print(label_counts)

# # split mergecompelete_dataset各一半
train_dataframes, test_dataframes = train_test_split(mergecompelete_dataset, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
# label_counts = test_dataframes['Label'].value_counts()
# print("test_dataframes\n",label_counts)
# label_counts = train_dataframes['Label'].value_counts()
# print("train_dataframes\n",label_counts)


# 分別取出Label等於8、9、13、14的數據
label_8_data = mergecompelete_dataset[mergecompelete_dataset['Label'] == 8]
label_9_data = mergecompelete_dataset[mergecompelete_dataset['Label'] == 9]
label_13_data = mergecompelete_dataset[mergecompelete_dataset['Label'] == 13]
label_14_data = mergecompelete_dataset[mergecompelete_dataset['Label'] == 14]

# 使用train_test_split分別劃分取Label相等8、9、13、14的數據
train_label_8, test_label_8 = train_test_split(label_8_data, test_size=0.4, random_state=42)
train_label_9, test_label_9 = train_test_split(label_9_data, test_size=0.5, random_state=42)
train_label_13, test_label_13 = train_test_split(label_13_data, test_size=0.5, random_state=42)
train_label_14, test_label_14 = train_test_split(label_14_data, test_size=0.5, random_state=42)

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



# 建立叫stratified_split的StratifiedShuffleSplit 對象
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
# 使用分層劃分資料集
def splitdatasetbalancehalf():
    for train_indices, test_indices in stratified_split.split(train_dataframes, train_dataframes['Label']):
        df1 = train_dataframes.iloc[train_indices]
        df2 = train_dataframes.iloc[test_indices]
        label_counts = df1['Label'].value_counts()
        label_counts2 = df2['Label'].value_counts()
        print("train_half1\n",label_counts)
        print("train_half2\n",label_counts2)

    return df1,df2

train_half1,train_half2 = splitdatasetbalancehalf()

SaveDataToCsvfile(train_dataframes, f"./data/{today}", f"train_dataframes_{today}")
SaveDataToCsvfile(test_dataframes, f"./data/{today}", f"test_dataframes_{today}")
SaveDataToCsvfile(train_half1, f"./data/{today}", f"train_half1_{today}")
SaveDataToCsvfile(train_half2,  f"./data/{today}", f"train_half2_{today}") 
SaveDataframeTonpArray(test_dataframes, "test", today)
SaveDataframeTonpArray(train_dataframes, "total_train", today)
SaveDataframeTonpArray(train_half1, "train_half1", today)
SaveDataframeTonpArray(train_half2, "train_half2", today)