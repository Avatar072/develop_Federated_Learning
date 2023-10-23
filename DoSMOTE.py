
import torch
import numpy as np
import matplotlib.pyplot as plt
import imblearn # Oversample with SMOTE and random undersample for imbalanced dataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from mytoolfunction import  ChooseLoadNpArray, ChooseTrainDatastes, ParseCommandLineArgs

filepath = "D:\\Labtest20230911\\"
######################## Choose Dataset ##############################
# 根据命令行参数选择数据集
# python DoSMOTE.py --dataset train_half1
# python DoSMOTE.py --dataset train_half1 --epochs 10000 --weaklabel 13
file, num_epochs, weakLabel= ParseCommandLineArgs(["dataset", "epochs","weaklabel"])
print(f"Dataset: {file}")

x_train, y_train, client_str = ChooseTrainDatastes(filepath, file)
# x_train, y_train, client_str = ChooseLoadNpArray(filepath, file)
#feature and label count
# print(x_train.shape[1])
# print(len(np.unique(y_train)))

# 打印原始类别分布
counter = Counter(y_train)
print(counter)

# 遍历每个标签并绘制对应的数据点
# 创建颜色映射
cmap = plt.get_cmap('tab20', lut=len(np.unique(y_train)))
for label in np.unique(y_train):
    label_indices = np.where(y_train == label)[0]
    plt.scatter(x_train[label_indices, 0], x_train[label_indices, 1], label=f'Label {label}', cmap=cmap)

# 添加图例
plt.legend()
plt.show()

################## DO SMOTE ##################
desired_sample_count = 150
sampling_strategy = {13: desired_sample_count}  # 你可以设置desired_sample_count为你希望的样本数量
k_neighbors = 1  # 调整k_neighbors的值
# oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
# 參數說明：

# ratio：用於指定重抽樣的比例，如果指定字元型的值，可以是'minority'，表示對少數類別的樣本進行抽樣、'majority'，表示對多數類別的樣本進行抽樣、
# 'not minority'表示採用欠採樣方法、'all'表示採用過採樣方法，
# 默認為'auto'，等同於'all'和'not minority';如果指定字典型的值，其中鍵為各個類別標籤，值為類別下的樣本量;
# random_state：用於指定隨機數生成器的種子，預設為None，表示使用預設的隨機數生成器;
# k_neighbors：指定近鄰個數，預設為5個;
# m_neighbors：指定從近鄰樣本中隨機挑選的樣本個數，預設為10個;
# kind：用於指定SMOTE演算法在生成新樣本時所使用的選項，預設為'regular'，表示對少數類別的樣本進行隨機採樣，也可以是'borderline1'、'borderline2'和'svm';
# svm_estimator：用於指定SVM分類器，預設為sklearn.svm.SVC，該參數的目的是利用支援向量機分類器生成支援向量，然後再生成新的少數類別的樣本;

oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)

# 对指定的类别进行数据转换
X_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

# # 打印新的类别分布
counter = Counter(y_resampled)
print(counter)

# 获取所有不同的标签值
unique_labels = np.unique(y_resampled)

# 创建颜色映射
cmap = plt.get_cmap('tab20', lut=len(unique_labels))

# 遍历每个标签并绘制对应的数据点
for label in unique_labels:
    label_indices = np.where(y_resampled == label)[0]
    plt.scatter(X_resampled[label_indices, 0], X_resampled[label_indices, 1], label=f'Label {label}', cmap=cmap)

# 添加图例
plt.legend()
plt.show()