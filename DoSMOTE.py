import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import imblearn # Oversample with SMOTE and random undersample for imbalanced dataset
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from mytoolfunction import  ChooseLoadNpArray, ChooseTrainDatastes, ParseCommandLineArgs,generatefolder

filepath = "D:\\Labtest20230911\\"
desired_sample_count = 4000
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# k_neighbors = 1  # 调整k_neighbors的值  label 8要設因為樣本只有2個
######################## Choose Dataset ##############################
# 根據參數選擇dataset
# python DoSMOTE.py --dataset train_half1
# python DoSMOTE.py --dataset total_train
# python DoSMOTE.py --dataset train_half1 --epochs 10000 --weaklabel 13
# python DoSMOTE.py --dataset train_half1 --method normal
# file, num_epochs, weakLabel= ParseCommandLineArgs(["dataset", "epochs","weaklabel"])
file, num_epochs,Choose_method = ParseCommandLineArgs(["dataset", "epochs", "method"])
print(f"Dataset: {file}")
print(f"Number of epochs: {num_epochs}")
print(f"Choose_method: {Choose_method}")
# ChooseLoadNpArray function  return x_train、y_train 和 client_str and Choose_method
x_train, y_train, client_str = ChooseLoadNpArray(filepath, file, Choose_method)
print(f"client_str: {client_str}")

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

### 一次SMOTE所有weaklabel
def DoALLWeakLabel(x_train,y_train):
    generatefolder(f'{filepath}' + '\\ALL_Label\\SMOTE\\' + f'{today}\\', client_str)

    # 对Label14进行SMOTE
    # sampling_strategy_14 = {14: desired_sample_count}
    x_train = x_train.real #去除复數 因為做完統計百分比PCA後會有
    # Assuming y_train contains the labels
    unique_labels = np.unique(y_train)

    # Choose a colormap with at least 15 distinct colors
    cmap = plt.get_cmap('tab20')

    for label in unique_labels:
        row_ix = np.where(y_train == label)[0]
        plt.scatter(x_train[row_ix, 0], x_train[row_ix, 1], label=f'{label}', color=cmap(label))

    plt.legend()
    plt.show()    

    sampling_strategy_9 = {9: 1000}
    oversample_9 = SMOTE(sampling_strategy=sampling_strategy_9, random_state=42)
    X_res, y_res = oversample_9.fit_resample(x_train, y_train)
    print("Label 9 SMOTE", Counter(y_res))
    # 对Label8进行SMOTE
    sampling_strategy_8 = {8: 1000}
    oversample_8 = SMOTE(sampling_strategy=sampling_strategy_8,k_neighbors = 2, random_state=42)
    X_res, y_res = oversample_8.fit_resample(X_res, y_res)
    print("Label 8 SMOTE", Counter(y_res))


     # # 獲取原始數據中標籤為 weakLabel 的索引
    label8_indices_Original = np.where(y_train == 8)[0]

    # 獲取原始數據中標籤為 weakLabel 的數據點
    x_train_label8_Oringinal = x_train[label8_indices_Original]

    # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
    label8_indices_SMOTE = np.where(y_res == 8)[0]

    # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
    X_resampled_label8_SMOTE = X_res[label8_indices_SMOTE]

    plt.scatter(x_train_label8_Oringinal[:, 0], 
            x_train_label8_Oringinal[:, 1], 
            c='red', marker='o', s=20, 
            label=f'Original Samples (Label {8}): {len(x_train_label8_Oringinal)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/SMOTE/{today}/{client_str}/Label8/SMOTE_Samples_Original_Label_{8}.png")
    plt.show()
    # 繪制SMOTE採樣後的數據中的標籤 weakLabel
    plt.scatter(X_resampled_label8_SMOTE[:, 0], 
                X_resampled_label8_SMOTE[:, 1], 
                c='blue', marker='x', s=36, 
                label=f'SMOTE Samples (Label {8}: {len(X_resampled_label8_SMOTE)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/SMOTE/{today}/{client_str}/Label8/BorederlineSMOTE_Samples_After_SMOTE_Label_{8}.png")
    plt.show()


    # # 獲取原始數據中標籤為 weakLabel 的索引
    label9_indices_Original = np.where(y_train == 9)[0]

    # 獲取原始數據中標籤為 weakLabel 的數據點
    x_train_label9_Oringinal = x_train[label9_indices_Original]

    # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
    label9_indices_SMOTE = np.where(y_res == 9)[0]

    # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
    X_resampled_label9_SMOTE = X_res[label9_indices_SMOTE]

    plt.scatter(x_train_label9_Oringinal[:, 0], 
            x_train_label9_Oringinal[:, 1], 
            c='red', marker='o', s=20, 
            label=f'Original Samples (Label {9}): {len(x_train_label9_Oringinal)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/SMOTE/{today}/{client_str}/Label9/SMOTE_Samples_Original_Label_{9}.png")
    plt.show()
    # 繪制SMOTE採樣後的數據中的標籤 weakLabel
    plt.scatter(X_resampled_label9_SMOTE[:, 0], 
                X_resampled_label9_SMOTE[:, 1], 
                c='blue', marker='x', s=36, 
                label=f'SMOTE Samples (Label {9}: {len(X_resampled_label9_SMOTE)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/SMOTE/{today}/{client_str}/Label9/BorederlineSMOTE_Samples_After_SMOTE_Label_{9}.png")
    plt.show()

    generatefolder(f'{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label8_and_Label9")
    # np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\Label9\\x_{file}_SMOTE_Label_9.npy", X_res)
    # np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\Label9\\y_{file}_SMOTE_Label_9.npy", y_res)
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\Label8_and_Label9\\x_{file}_SMOTE_Label8_and_Label9_{today}.npy", X_res)
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\Label8_and_Label9\\y_{file}_SMOTE_Label8_and_Label9_{today}.npy", y_res)


def DoALL_Label(x_train,y_train):
    generatefolder(filepath, "ALL_Label")
    generatefolder(f'{filepath}' + '\\ALL_Label\\', "SMOTE")
    generatefolder(f'{filepath}' + '\\ALL_Label\\SMOTE\\', today)
    # 对ALL　Label进行SMOTE
    # 打印原始类别分布
    # counter = Counter(y_train)
    # print(counter)
    for i in range(0, 15):# 因為有15個Label
        print(i)
        sampling_strategy_ALL = {i: 10000}
        oversample = SMOTE(sampling_strategy=sampling_strategy_ALL, random_state=42)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        print("ALL Label SMOTE", Counter(y_train))
    
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\x_{file}_SMOTE_ALL_Label.npy", x_train)
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\y_{file}_SMOTE_ALL_Label.npy", y_train)

def DoBorederlineSMOTE(x_train, y_train,choosekind,ChooseLable):
    generatefolder(filepath, "ALL_Label")
    generatefolder(f'{filepath}' + '\\ALL_Label\\', "BorederlineSMOTE")
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\', choosekind)
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\', today)
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\', client_str)
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\' + f'{client_str}\\', f'{ChooseLable}')
    x_train = x_train.real #去除复數 因為做完統計百分比PCA後會有
    # Assuming y_train contains the labels
    unique_labels = np.unique(y_train)

    # Choose a colormap with at least 15 distinct colors
    cmap = plt.get_cmap('tab20')

    for label in unique_labels:
        row_ix = np.where(y_train == label)[0]
        plt.scatter(x_train[row_ix, 0], x_train[row_ix, 1], label=f'{label}', color=cmap(label))

    plt.legend()
    plt.show()
    
    print('Original dataset shape %s' % Counter(y_train))
    sampling_strategy_8 = {8: 1000}
    sampling_strategy_9 = {9: 1000} 
    # BorderlineSMOTE
    oversample_8 = BorderlineSMOTE(sampling_strategy=sampling_strategy_8, kind=choosekind ,k_neighbors=1,m_neighbors =5,random_state=42)
    X_res, y_res = oversample_8.fit_resample(x_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    
    # # 獲取原始數據中標籤為 weakLabel 的索引
    label8_indices_Original = np.where(y_train == 8)[0]

    # 獲取原始數據中標籤為 weakLabel 的數據點
    x_train_label8_Oringinal = x_train[label8_indices_Original]

    # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
    label8_indices_SMOTE = np.where(y_res == 8)[0]

    # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
    X_resampled_label8_SMOTE = X_res[label8_indices_SMOTE]

    plt.scatter(x_train_label8_Oringinal[:, 0], 
            x_train_label8_Oringinal[:, 1], 
            c='red', marker='o', s=20, 
            label=f'Original Samples (Label {8}): {len(x_train_label8_Oringinal)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/BorederlineSMOTE/{choosekind}/{today}/{client_str}/Label8/BorederlineSMOTE_{choosekind}_Samples_Original_Label_{8}.png")
    plt.show()
    # 繪制SMOTE採樣後的數據中的標籤 weakLabel
    plt.scatter(X_resampled_label8_SMOTE[:, 0], 
                X_resampled_label8_SMOTE[:, 1], 
                c='blue', marker='x', s=36, 
                label=f'Borederline SMOTE {choosekind} Samples (Label {8}: {len(X_resampled_label8_SMOTE)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/BorederlineSMOTE/{choosekind}/{today}/{client_str}/Label8/BorederlineSMOTE_{choosekind}_Samples_AfterBorderLine_SMOTE_Label_{8}.png")
    plt.show()


    oversample_9 = BorderlineSMOTE(sampling_strategy=sampling_strategy_9, kind=choosekind ,k_neighbors=1,m_neighbors =10,random_state=42)
    X_res, y_res = oversample_9.fit_resample(X_res, y_res)
    print('Resampled dataset shape %s' % Counter(y_res))

    # # 獲取原始數據中標籤為 weakLabel 的索引
    label9_indices_Original = np.where(y_train == 9)[0]

    # 獲取原始數據中標籤為 weakLabel 的數據點
    x_train_label9_Oringinal = x_train[label9_indices_Original]

    # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
    label9_indices_SMOTE = np.where(y_res == 9)[0]

    # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
    X_resampled_label9_SMOTE = X_res[label9_indices_SMOTE]



    plt.scatter(x_train_label9_Oringinal[:, 0], 
            x_train_label9_Oringinal[:, 1], 
            c='red', marker='o', s=20, 
            label=f'Original Samples (Label {9}): {len(x_train_label9_Oringinal)})')
    plt.legend()
    plt.savefig(f"./ALL_Label/BorederlineSMOTE/{choosekind}/{today}/{client_str}/Label9/BorederlineSMOTE_{choosekind}_Samples_Original_Label_{9}.png")
    plt.show()
    # 繪制SMOTE採樣後的數據中的標籤 weakLabel
    plt.scatter(X_resampled_label9_SMOTE[:, 0], 
                X_resampled_label9_SMOTE[:, 1], 
                c='blue', marker='x', s=36, 
                label=f'Borederline SMOTE {choosekind} Samples (Label {9}: {len(X_resampled_label9_SMOTE)})')
    # 添加圖例
    plt.legend()
    plt.savefig(f"./ALL_Label/BorederlineSMOTE/{choosekind}/{today}/{client_str}/Label9/BorederlineSMOTE_{choosekind}_Samples_Label_{9}.png")
    plt.show()
    
    # np.save(f"{filepath}\\ALL_Label\\BorederlineSMOTE\\{choosekind}\\{today}\\{client_str}\\Label8_and_Label9\\x_{file}_BorederlineSMOTE_Label8_and_Label9_{today}.npy", X_res)
    # np.save(f"{filepath}\\ALL_Label\\BorederlineSMOTE\\{choosekind}\\{today}\\{client_str}\\Label8_and_Label9\\y_{file}_BorederlineSMOTE_Label8_and_Label9_{today}.npy", y_res)

# DoALLWeakLabel(x_train,y_train)
# DoALL_Label(x_train,y_train)
DoBorederlineSMOTE(x_train, y_train,"borderline-1","Label8_and_Label9")

###############################################
# #一次SMOTE只SMOTE一個weaklabel
# sampling_strategy = {weakLabel: desired_sample_count}   # weakLabel和設置desired_sample_count為希望稱生成的樣本數量
#                                                         # SMOTE進行過抽樣時，所請求的樣本數應不小於原始類別中的樣本數。

# oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
# # 參數說明：
# # ratio：用於指定重抽樣的比例，如果指定字元型的值，可以是'minority'，表示對少數類別的樣本進行抽樣、'majority'，表示對多數類別的樣本進行抽樣、
# # 'not minority'表示採用欠採樣方法、'all'表示採用過採樣方法，
# # 默認為'auto'，等同於'all'和'not minority';如果指定字典型的值，其中鍵為各個類別標籤，值為類別下的樣本量;
# # random_state：用於指定隨機數生成器的種子，預設為None，表示使用預設的隨機數生成器;
# # k_neighbors：指定近鄰個數，預設為5個;
# # m_neighbors：指定從近鄰樣本中隨機挑選的樣本個數，預設為10個;
# # kind：用於指定SMOTE演算法在生成新樣本時所使用的選項，預設為'regular'，表示對少數類別的樣本進行隨機採樣，也可以是'borderline1'、'borderline2'和'svm';
# # svm_estimator：用於指定SVM分類器，預設為sklearn.svm.SVC，該參數的目的是利用支援向量機分類器生成支援向量，然後再生成新的少數類別的樣本;

# oversample = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)

# # 對指定的類別進行資料轉換
# X_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

# # # 列印新的Label分佈
# counter = Counter(y_resampled)
# print(counter)

# # 獲取所有不同的Label值
# unique_labels = np.unique(y_resampled)

# # 創建顏色映射
# cmap_smote = plt.get_cmap('tab20', lut=len(unique_labels))

# # 遍歷每個標籤並繪制對應的數據點
# # 繪制原始數據集
# # for label in np.unique(y_train):
# #     label_indices = np.where(y_train == label)[0]
# #     plt.scatter(x_train[label_indices, 0], x_train[label_indices, 1], label=f'Label {label}', cmap=cmap_original)
# # plt.legend()
# # plt.show()

# # 獲取原始數據中標籤為 weakLabel 的索引
# label_indices_Original = np.where(y_train == weakLabel)[0]

# # 獲取原始數據中標籤為 weakLabel 的數據點
# x_train_label_Oringinal = X_resampled[label_indices_Original]

# # 繪制原始數據中的標籤 weakLabel 數據點
# plt.scatter(x_train_label_Oringinal[:, 0], 
#             x_train_label_Oringinal[:, 1], 
#             c='red', marker='o', s=100, 
#             label=f'Original Samples (Label {weakLabel}: {len(x_train_label_Oringinal)})')
# # 添加圖例
# plt.legend()
# plt.savefig(f"./GAN_data_train_half2/Original_Samples_Label_{weakLabel}.png")
# plt.show()

# # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
# label_indices_SMOTE = np.where(y_resampled == weakLabel)[0]

# # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
# X_resampled_label_SMOTE = X_resampled[label_indices_SMOTE]

# # 繪制SMOTE採樣後的數據中的標籤 weakLabel
# plt.scatter(X_resampled_label_SMOTE[:, 0], 
#             X_resampled_label_SMOTE[:, 1], 
#             c='blue', marker='x', s=100, 
#             label=f'SMOTE Samples (Label {weakLabel}: {len(X_resampled_label_SMOTE)})')
# # 添加圖例
# plt.legend()
# plt.savefig(f"./GAN_data_train_half2/SMOTE_Samples_Label_{weakLabel}.png")
# plt.show()

# plt.scatter(x_train_label_Oringinal[:, 0], 
#             x_train_label_Oringinal[:, 1], 
#             c='red', marker='o', s=100, 
#             label=f'Original Samples (Label {weakLabel}): {len(x_train_label_Oringinal)})')

# plt.scatter(X_resampled_label_SMOTE[:, 0], 
#             X_resampled_label_SMOTE[:, 1], 
#             c='blue', marker='x', s=40, 
#             label=f'SMOTE Samples (Label {weakLabel}: {len(X_resampled_label_SMOTE)})')


# plt.legend() # 作用是將圖例添加到當前的繪圖中
# plt.savefig(f"./GAN_data_train_half2/Original_and_SMOTE_Samples_Label_{weakLabel}.png")
# plt.show()
# np.save(f"{filepath}\\GAN_data_train_half2\\x_{file}_SMOTE_{weakLabel}.npy", X_resampled)
# np.save(f"{filepath}\\GAN_data_train_half2\\y_{file}_SMOTE_{weakLabel}.npy", y_resampled)