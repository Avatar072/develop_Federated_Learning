import os

### 檢查資料夾是否存在
def CheckFolderExists (folder_name):
    if not os.path.exists(folder_name):
        return False
    else:
        return True
    
### 檢查檔案是否存在
def CheckFileExists (file):
   if os.path.isfile(file):
        print(f"{file} 是一個存在的檔案。")
        return True
   else:
        print(f"{file} 不是一個檔案或不存在。")
        return False
    
### Save data to csv
def SaveDataToCsvfile(df, filename):
    # 抓取當前工作目錄名稱
    current_directory = os.getcwd()
    print("當前工作目錄", current_directory)
    folder_name = filename + "_folder"
    generatefolder(folder_name)
    csv_filename = os.path.join(current_directory, 
                                folder_name, filename + ".csv")
    print("存檔位置跟檔名", csv_filename)
    df.to_csv(csv_filename, index=False)

### 指定要建的資料夾名稱
def generatefolder(folder_name):
    if folder_name is None:
        folder_name = "my_AnalyseReportfolder"

    file_exists  = CheckFolderExists(folder_name)
    # 使用os.path.exists()檢文件夹是否存在
    if file_exists:
        # 如果文件夹不存在，就创建它
        os.makedirs(folder_name)
        print(f"資料夾 '{folder_name}' 創建。")
    else:
        print(f"資料夾 '{folder_name}' 已存在，不需再創建。")