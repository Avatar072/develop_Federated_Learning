import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mytoolfunction import SaveDataToCsvfile, SaveDataframeTonpArray,generatefolder
def zeroMean(dataMat):
    # 求列均值
    meanVal = np.mean(dataMat, axis=0)
    # 求列差值
    newData = dataMat - meanVal
    return newData, meanVal

# 對初始數據進行降維處理
def pcaa(dataMat, percent):
    newData, meanVal = zeroMean(dataMat)
    print("Percentage: ", percent)

    # 求協方差矩陣
    covMat = np.cov(newData, rowvar=0) #cov為計算協方差，
    print("協方差矩陣的shape: ", covMat.shape)
    print("以下為協方差矩陣:")
    print(covMat)

    # 求特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print("eigVals: ", eigVals.shape)
    print(eigVals)
    print("\n")
    print("eigVects: ", eigVects.shape)
    print(eigVects)
    print("\n")
    

    # 抽取前n個特征向量
    n = percentage2n(eigVals, percent) #透過方差百分比決定 n 值
    print("數據降低到：" + str(n) + '維')

    # 將特征值按從小到大排序
    eigValIndice = np.argsort(eigVals)
    print("特徵值由小至大排序: ", eigValIndice)
    # 取最大的n個特征值的下標
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    # 取最大的n個特征值的特征向量
    print("決定選取的特徵值: ", n_eigValIndice)
    n_eigVect = eigVects[:, n_eigValIndice]
    # 取得降低到n維的數據
    lowDataMat = newData * n_eigVect

    ##---------------------------------###
    # 抽取前n+1個特征向量
    n1 = percentage2n(eigVals, percent) + 1 #透過方差百分比決定 n+1 值
#     print("選擇n+1，數據降低到：" + str(n1) + '維')

    # 將特征值按從小到大排序
    eigValIndice_1 = np.argsort(eigVals)
    print("特徵值由小至大排序: ", eigValIndice_1)
    # 取最大的n+1個特征值的下標
    n_eigValIndice_1 = eigValIndice_1[-1:-(n + 2):-1]
    # 取最大的n+1個特征值的特征向量
    print("決定選取的特徵值: ", n_eigValIndice_1)
    n_eigVect_1 = eigVects[:, n_eigValIndice_1]

    # 取得降低到n+1維的數據
    lowDataMat_1 = newData * n_eigVect_1

    reconMat_1 = (lowDataMat_1 * n_eigVect_1.T) + meanVal

    return lowDataMat, lowDataMat_1, eigVals

# 通過方差百分比確定抽取的特征向量的個數
def percentage2n(eigVals, percentage):
    # 按降序排序
    sortArray = np.sort(eigVals)[-1::-1]
    # 求和
    arraySum = sum(sortArray)

    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * percentage:
            return num

def covariance_bar(eigVals):
    Index = np.argsort(eigVals)[-1::-1]
    Value = np.sort(eigVals)[-1::-1]
    arraySum = sum(Value)
    aaa = np.zeros(shape=len(eigVals))
    j = 0
    for i in Value:
        temp = 0
        temp = np.round((i/arraySum), 4)
        #aaa = np.append(aaa, temp)
        aaa[j] = temp
        j = j + 1
    return Index, aaa
#############################################################################  variable  ###################
filepath = "D:\\Labtest20230911\\data"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\After_PCA產生天日期的資料夾
generatefolder(filepath + "\\", "After_PCA")
generatefolder(filepath + "\\After_PCA\\", today)

# Loading datasets after label_Encoding
mergecompelete_dataset = pd.read_csv(filepath + "\\total_encoded_updated.csv")
print(type(mergecompelete_dataset))
print(mergecompelete_dataset.shape)

X=mergecompelete_dataset.iloc[:,:-1]
X=X.values
# scaler = preprocessing.StandardScaler() #歸一化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler.fit(X)
X=scaler.transform(X)
##########  正規化  ########

raw_train = mergecompelete_dataset.drop("Label",axis=1)
# scaler = preprocessing.StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler.fit(raw_train)
raw_train=scaler.transform(raw_train)

print(type(raw_train))
print(raw_train.shape)

fin = pcaa(raw_train, 1)
data1 = fin[0]
data2 = fin[1]
eigen = fin[2]

a, b = covariance_bar(eigen)
aa = a.tolist()
#print(aa)
label_x = []
for i in aa:
    temp_list = "F" + str(i)
    label_x.append(temp_list)
x = np.arange(len(a))
fig = plt.figure(figsize=(30,5))    # 设置画布大小
plt.bar(x, b)
plt.xticks(x,label_x)
plt.xlabel('Feature')
plt.ylabel('Cov percentage')
plt.title('Covariance percentage of Each Feature')
# plt.show()
plt.savefig(filepath + "\\After_PCA\\" +today+f"\\Feature choose_{today}.jpg")
plt.close("all")

print("---------PCA n result-----------")
print("N shape: ", data1.shape)
# 將 numpy.matrix 轉換為 numpy.ndarray
data1 = np.array(data1)
print(type(data1))
# 將 numpy.ndarray 轉換為pandas DataFrame
data1 = pd.DataFrame(data1)
finalDf = pd.concat([data1, mergecompelete_dataset[['Label']]], axis = 1)

data1=finalDf
#將PCA完的data1資料8 2分
train_dataframes, test_dataframes = train_test_split(data1, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%
print(len(train_dataframes))
print(type(train_dataframes))
print(len(test_dataframes))

# 分別取出Label等於8、9、13、14的數據
# 先取消做14
label_8_data = data1[data1['Label'] == 8]
label_9_data = data1[data1['Label'] == 9]
label_13_data = data1[data1['Label'] == 13]
label_14_data = data1[data1['Label'] == 14]

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
print("test_dataframes\n", test_dataframes.shape)
label_counts = train_dataframes['Label'].value_counts()
print("train_dataframes\n", label_counts)
print("train_dataframes\n", train_dataframes.shape)

SaveDataToCsvfile(train_dataframes, f"./data/After_PCA/{today}", f"train_dataframes_{today}")
SaveDataToCsvfile(test_dataframes, f"./data/After_PCA/{today}", f"test_dataframes_{today}")

x_train = np.array(train_dataframes.iloc[:,:-1])
y_train = np.array(train_dataframes.iloc[:,-1])

x_test = np.array(test_dataframes.iloc[:,:-1])
y_test = np.array(test_dataframes.iloc[:,-1])

np.save(f'./data/After_PCA/{today}/x_train_1', x_train)
np.save(f'./data/After_PCA/{today}/x_test_1', x_test)
np.save(f'./data/After_PCA/{today}/y_train_1', y_train)
np.save(f'./data/After_PCA/{today}/y_test_1', y_test)
print("---------Finished PCA n txt save--------\n")