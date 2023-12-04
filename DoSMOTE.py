import torch
import datetime
import numpy as np
import pandas as pd
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
from mytoolfunction import splitdatasetbalancehalf,SaveDataToCsvfile,SaveDataframeTonpArray

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
# total_train smote後切一半
def spilttrainhalfAfterSMOTE(X_res,y_res):
    # 使用 pd.DataFrame 將 X_res 和 y_res 水平合併
    column_names = ["principal_Component" + str(i) for i in range(1, 64)] + ["Label"]
    combined_df = pd.DataFrame(np.column_stack((X_res, y_res)), columns=column_names)
    # 找到不包含NaN、Infinity和"inf"值的行
    combined_df = combined_df[~combined_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # combined_df.to_csv("./data/combined_df.csv", index=False)
    # 顯示合併後的 DataFrame
    print(combined_df)
    # split train_dataframes各一半
    train_half1,train_half2 = splitdatasetbalancehalf(combined_df)
    # 找到train_df_half1和train_df_half2中重复的行
    duplicates = train_half2[train_half2.duplicated(keep=False)]

    # 删除train_df_half2中与train_df_half1重复的行
    train_df_half2 = train_half2[~train_half2.duplicated(keep=False)]
    SaveDataToCsvfile(train_half1, f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", f"train_half1_AfterSMOTEspilt_{today}")
    SaveDataToCsvfile(train_half2,  f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", f"train_half2_AfterSMOTEspilt_{today}") 
    SaveDataframeTonpArray(train_half1, f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", "train_half1_AfterSMOTEspilt", today)
    SaveDataframeTonpArray(train_half2, f"./ALL_Label/SMOTE/{today}/{client_str}/{file}_AfterSMOTEspilthalf", "train_half2_AfterSMOTEspilt", today)

def SMOTEParameterSet(choose_strategy, choose_k_neighbors,x_train, y_train, Label_encode):
        
        oversample = SMOTE(sampling_strategy = choose_strategy, k_neighbors = choose_k_neighbors, random_state = 42)
        
        X_res, y_res = oversample.fit_resample(x_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # # 獲取原始數據中標籤為 weakLabel 的索引
        Label_indices_Original = np.where(y_train == Label_encode)[0]

        # 獲取原始數據中標籤為 weakLabel 的數據點
        x_train_Label_Oringinal = x_train[Label_indices_Original]

        # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
        Label_indices_SMOTE = np.where(y_res == Label_encode)[0]

        # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
        X_resampled_Label_SMOTE = X_res[Label_indices_SMOTE]

        plt.scatter(x_train_Label_Oringinal[:, 0], 
                x_train_Label_Oringinal[:, 1], 
                c='red', marker='o', s=20, 
                label=f'Original Samples (Label {Label_encode}): {len(x_train_Label_Oringinal)})')
        plt.legend()
        plt.savefig(f"./ALL_Label/SMOTE/{today}/{client_str}/Label{Label_encode}/SMOTE_Samples_Original_Label_{Label_encode}.png")

        plt.show()
        # 繪制SMOTE採樣後的數據中的標籤 weakLabel
        plt.scatter(X_resampled_Label_SMOTE[:, 0], 
                    X_resampled_Label_SMOTE[:, 1], 
                    c='blue', marker='x', s=36, 
                    label=f'SMOTE Samples (Label {Label_encode}: {len(X_resampled_Label_SMOTE)})')
        # 添加圖例
        plt.legend()
        plt.savefig(f"./ALL_Label/SMOTE/{today}/{client_str}/Label{Label_encode}/SMOTE_Samples_After_SMOTE_Label_{Label_encode}.png")   
        plt.show()
        return X_res, y_res

def DoALLWeakLabel(x_train,y_train, ChooseLabel, Choose_totaltrain):
    
    generatefolder(f'{filepath}' + '\\ALL_Label\\SMOTE\\' + f'{today}\\', client_str)
    generatefolder(f'{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', f'{ChooseLabel}')
    generatefolder(f'{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label8")
    generatefolder(f'{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label9")
    generatefolder(f'{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\', "Label13")

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
    # SMOTE到原本比數的兩倍
    sampling_strategy_Label8 = {8: 12}
    sampling_strategy_Label9 = {9: 36}
    sampling_strategy_Label13 = {13: 20}
    y_train = y_train.astype(int) 
     # Start Do SMOTE
    if(Choose_totaltrain != True):
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label8, 2, x_train, y_train, 8)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label9, 5, X_res, y_res, 9)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label13,4, X_res, y_res, 13)
    else: # k_neighbors  use default 5
        print("use total train generate SMOTE data")
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label8, 5, x_train, y_train, 8)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label9, 5, X_res, y_res, 9)
        X_res, y_res = SMOTEParameterSet(sampling_strategy_Label13,5, X_res, y_res, 13)
        y_res = y_res.astype(int) 
        spilttrainhalfAfterSMOTE(X_res,y_res)

    print('After SMOTE dataset shape %s' % Counter(y_res)) 
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\{ChooseLabel}\\x_{file}_SMOTE_{ChooseLabel}_{today}.npy", X_res)
    np.save(f"{filepath}\\ALL_Label\\SMOTE\\{today}\\{client_str}\\{ChooseLabel}\\y_{file}_SMOTE_{ChooseLabel}_{today}.npy", y_res)


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


def BorderLineParameterSet(choose_strategy, choosekind, choose_k_neighbors,choose_m_neighbors,x_train, y_train, Label_encode):
        
        oversample = BorderlineSMOTE(sampling_strategy = choose_strategy, kind = choosekind, 
                                        k_neighbors = choose_k_neighbors, m_neighbors = choose_m_neighbors,
                                        random_state = 42)
        
        X_res, y_res = oversample.fit_resample(x_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # # 獲取原始數據中標籤為 weakLabel 的索引
        Label_indices_Original = np.where(y_train == Label_encode)[0]

        # 獲取原始數據中標籤為 weakLabel 的數據點
        x_train_Label_Oringinal = x_train[Label_indices_Original]

        # 找到SMOTE採樣後的數據中標籤 weakLabel 的索引
        Label_indices_SMOTE = np.where(y_res == Label_encode)[0]

        # 獲取SMOTE採樣後的數據中標籤 weakLabel 的數據點
        X_resampled_Label_SMOTE = X_res[Label_indices_SMOTE]

        plt.scatter(x_train_Label_Oringinal[:, 0], 
                x_train_Label_Oringinal[:, 1], 
                c='red', marker='o', s=20, 
                label=f'Original Samples (Label {Label_encode}): {len(x_train_Label_Oringinal)})')
        plt.legend()
        plt.savefig(f"./ALL_Label/BorederlineSMOTE/{choosekind}/{today}/{client_str}/Label{Label_encode}/BorederlineSMOTE_{choosekind}_Samples_Original_Label_{Label_encode}.png")
        plt.show()
        # 繪制SMOTE採樣後的數據中的標籤 weakLabel
        plt.scatter(X_resampled_Label_SMOTE[:, 0], 
                    X_resampled_Label_SMOTE[:, 1], 
                    c='blue', marker='x', s=36, 
                    label=f'Borederline SMOTE {choosekind} Samples (Label {Label_encode}: {len(X_resampled_Label_SMOTE)})')
        # 添加圖例
        plt.legend()
        plt.savefig(f"./ALL_Label/BorederlineSMOTE/{choosekind}/{today}/{client_str}/Label{Label_encode}/BorederlineSMOTE_{choosekind}_Samples_Label_{Label_encode}.png")
        plt.show()
        return X_res, y_res


def DoBorederlineSMOTE(x_train, y_train,choosekind,ChooseLable):
    # 產生存檔分類用資料夾
    generatefolder(filepath, "ALL_Label")
    generatefolder(f'{filepath}' + '\\ALL_Label\\', "BorederlineSMOTE")
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\', choosekind)
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\', today)
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\', client_str)
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\' + f'{client_str}\\', f'{ChooseLable}')
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\' + f'{client_str}\\', "Label8")
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\' + f'{client_str}\\', "Label9")
    generatefolder(f'{filepath}' + '\\ALL_Label\\BorederlineSMOTE\\'+ f'{choosekind}\\' + f'{today}\\' + f'{client_str}\\', "Label13")

    x_train = x_train.real #去除复數 因為做完統計百分比PCA後會有
    # 將標籤列轉換為整數型別
    y_train = y_train.astype(int)
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
    sampling_strategy_Label8 = {8: 2000}
    sampling_strategy_Label9 = {9: 2000}
    sampling_strategy_Label13 = {13: 2000} 
    # Start Do BorderlineSMOTE
    # y_res = y_res.astype(int)
    y_train = y_train.astype(int)
    X_res, y_res = BorderLineParameterSet(sampling_strategy_Label8, choosekind, 5, 10, x_train, y_train, 8)
    X_res, y_res = BorderLineParameterSet(sampling_strategy_Label9, choosekind, 5, 10, X_res, y_res, 9)
    X_res, y_res = BorderLineParameterSet(sampling_strategy_Label13, choosekind, 5, 10, X_res, y_res, 13)

    print('Afterr BorderLine SMOTE dataset shape %s' % Counter(y_res))

    np.save(f"{filepath}\\ALL_Label\\BorederlineSMOTE\\{choosekind}\\{today}\\{client_str}\\{ChooseLable}\\x_{file}_BorederlineSMOTE_{choosekind}_{ChooseLable}_{today}.npy", X_res)
    np.save(f"{filepath}\\ALL_Label\\BorederlineSMOTE\\{choosekind}\\{today}\\{client_str}\\{ChooseLable}\\y_{file}_BorederlineSMOTE_{choosekind}_{ChooseLable}_{today}.npy", y_res)

# DoALLWeakLabel(x_train,y_train,"Label8_and_Label9_Label13", True)
# DoALL_Label(x_train,y_train)
DoBorederlineSMOTE(x_train, y_train,"borderline-1","Label8_and_Label9_Label13")
# DoBorederlineSMOTE(x_train, y_train,"borderline-2","Label8_and_Label9_Label13")

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