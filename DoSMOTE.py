
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
# 根據參數選擇dataset
# python DoSMOTE.py --dataset train_half1
# python DoSMOTE.py --dataset train_half1 --epochs 10000 --weaklabel 13
file, num_epochs, weakLabel= ParseCommandLineArgs(["dataset", "epochs","weaklabel"])
print(f"Dataset: {file}")
print(f"weakLabel: {weakLabel}")
x_train, y_train, client_str = ChooseTrainDatastes(filepath, file)
# x_train, y_train, client_str = ChooseLoadNpArray(filepath, file)
#feature and label count
# print(x_train.shape[1])
# print(len(np.unique(y_train)))

# 打印原始类别分布
counter = Counter(y_train)
print(counter)

# 遍歷每個Label並繪製對應的數據點
# 創建顏色映射
cmap_original = plt.get_cmap('tab20', lut=len(np.unique(y_train)))
# 畫原始dataset分布
# for label in np.unique(y_train):
#     label_indices = np.where(y_train == label)[0]
#     plt.scatter(x_train[label_indices, 0], x_train[label_indices, 1], label=f'Label {label}', cmap=cmap_original)

# plt.legend()
# plt.show()

################## DO SMOTE ##################
desired_sample_count = 1500
sampling_strategy = {weakLabel: desired_sample_count}   # weakLabel和設置desired_sample_count為希望稱生成的樣本數量
                                                        # SMOTE進行過抽樣時，所請求的樣本數應不小於原始類別中的樣本數。
k_neighbors = 5  # 调整k_neighbors的值  label 8要設因為樣本只有2個
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

oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)

# 對指定的類別進行資料轉換
X_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

# # 列印新的Label分佈
counter = Counter(y_resampled)
print(counter)

# 獲取所有不同的Label值
unique_labels = np.unique(y_resampled)

# 創建顏色映射
cmap_smote = plt.get_cmap('tab20', lut=len(unique_labels))

# 遍歷每個標籤並繪制對應的數據點
# 繪制原始數據集
# for label in np.unique(y_train):
#     label_indices = np.where(y_train == label)[0]
#     plt.scatter(x_train[label_indices, 0], x_train[label_indices, 1], label=f'Label {label}', cmap=cmap_original)
# plt.legend()
# plt.show()

# 獲取原始數據中標籤為 weakLabel 的索引
label_indices_Original = np.where(y_train == weakLabel)[0]

# 獲取原始數據中標籤為 weakLabel 的數據點
x_train_label_Oringinal = X_resampled[label_indices_Original]

# 繪制原始數據中的標籤 weakLabel 數據點
plt.scatter(x_train_label_Oringinal[:, 0], 
            x_train_label_Oringinal[:, 1], 
            c='red', marker='o', s=100, 
            label=f'Original Samples (Label {weakLabel}: {len(x_train_label_Oringinal)})')
# 添加圖例
plt.legend()
plt.savefig(f"./GAN_data_train_half2/Original_Samples_Label_{weakLabel}.png")
plt.show()

# 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
label_indices_SMOTE = np.where(y_resampled == weakLabel)[0]

# 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
X_resampled_label_SMOTE = X_resampled[label_indices_SMOTE]

# 繪制SMOTE採樣後的數據中的標籤 weakLabel
plt.scatter(X_resampled_label_SMOTE[:, 0], 
            X_resampled_label_SMOTE[:, 1], 
            c='blue', marker='x', s=100, 
            label=f'SMOTE Samples (Label {weakLabel}: {len(X_resampled_label_SMOTE)})')
# 添加圖例
plt.legend()
plt.savefig(f"./GAN_data_train_half2/SMOTE_Samples_Label_{weakLabel}.png")
plt.show()

plt.scatter(x_train_label_Oringinal[:, 0], 
            x_train_label_Oringinal[:, 1], 
            c='red', marker='o', s=100, 
            label=f'Original Samples (Label {weakLabel}): {len(x_train_label_Oringinal)})')

plt.scatter(X_resampled_label_SMOTE[:, 0], 
            X_resampled_label_SMOTE[:, 1], 
            c='blue', marker='x', s=40, 
            label=f'SMOTE Samples (Label {weakLabel}: {len(X_resampled_label_SMOTE)})')


plt.legend()
plt.savefig(f"./GAN_data_train_half2/Original_and_SMOTE_Samples_Label_{weakLabel}.png")
plt.show()
np.save(f"{filepath}\\GAN_data_train_half2\\x_{file}_SMOTE_{weakLabel}.npy", X_resampled)
np.save(f"{filepath}\\GAN_data_train_half2\\y_{file}_SMOTE_{weakLabel}.npy", y_resampled)