
import torch
import numpy as np
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

# x_train, y_train, client_str = ChooseTrainDatastes(filepath, file)
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file)
#feature and label count
# print(x_train.shape[1])
# print(len(np.unique(y_train)))

# 打印原始类别分布
counter = Counter(y_train)
print(counter)

# # 创建SMOTE对象，设置参数
oversample = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)

# # 进行数据转换
# X_resampled, y_resampled =x_train,y_train
X_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

# # 打印新的类别分布
counter = Counter(y_resampled)
print(counter)

# 绘制样本点
for label, _ in counter.items():
    row_ix = where(y_resampled == label)[0]
    pyplot.scatter(X_resampled[row_ix, 0], X_resampled[row_ix, 1], label=str(label))

pyplot.legend()
pyplot.show()