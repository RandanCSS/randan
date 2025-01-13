# coding: utf-8

'''
A module for formatting to a dataframe files of the formats: CSV, Excel and JSON. It facilitates working with data from social media
'''

# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from google_drive_downloader import GoogleDriveDownloader
        import os, pandas
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '').replace('_', '')
        if '.' in module: module = module.split('.')[1] 
        print('Пакет', module,
              'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

def getFolder():
    folder = input('--- С какой директорией хотите работать? Укажите полный путь к ней, включая её саму:')
    folder = folder.replace('"', '') # поскольку в Windows путь из Проводника закавычен
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    return folder, slash

def getFolderFile(*arg): # арки: fileName и folder
    # print(*arg)
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if len(arg) == 0:
        # print(0)
        folderFile = input('--- С каким файлом хотите работать? Укажите полный путь, включая название файла:')
        folderFile = folderFile.replace('"', '') # поскольку в Windows путь из Проводника закавычен
        # print(folderFile)
        fileName = folderFile.split(slash)[-1]
        # print(fileName)
        folder = slash.join(folderFile.split(slash)[:-1]) # из полного пути убрать имя файла
        folder += slash
        # print(folder)

    if len(arg) == 1:
        # print(1)
        fileName = arg[0]
        fileName = fileName.replace('"', '') # поскольку в Windows путь из Проводника закавычен
        # print(fileName)
        folder = ''
        # print(folder)

    if len(arg) == 2:
        # print(2)
        fileName = arg[0]
        fileName = fileName.replace('"', '') # поскольку в Windows путь из Проводника закавычен
        # print(fileName)
        folder = arg[1]
        folder = folder.replace('"', '') # поскольку в Windows путь из Проводника закавычен
        folder += slash
        # print(folder)
    return fileName, folder, slash

def excel2df(*arg):
    # print(*arg)
    fileName, folder, slash = getFolderFile(*arg)
    folderFile = folder + fileName
    # print(folderFile)
    error = None
    try:
        df = pandas.read_excel(folderFile, index_col=0)
    except FileNotFoundError:
        errorDescription = sys.exc_info()
        error = str(errorDescription[1]).replace("No module named '", '').replace("'", '').replace('_', '')
        print('error:', error)     
    
    if error == None:
        display(df.head())
        print('Число столбцов:', df.shape[1], ', число строк', df.shape[0], '\n')
    else:
        df = None
    return df, error, fileName, folder, slash    

def files2df(*arg):
    # print(*arg)
    print('Эта функция предназначена для оформления в датафрейм таблицы из файла формата Excel'
          , 'и присоединения к ней связанных ключом (id) таблиц из файлов форматов CSV, Excel и JSON,'
          , 'расположенных в той же директории')
    df, error, fileName, folder, slash  = excel2df(*arg)
    
    # print('fileName', fileName) # для отладки
    # print('folder', folder) # для отладки
    if len(folder) == 0: # значит, из excel2df олный путь передан в fileName
        folder = slash.join(fileName.split(slash)[:-1]) + slash # из полного пути убрать имя файла
        fileName = fileName.split(slash)[-1] # из полного пути оставить только имя файла

    fileNameS = os.listdir(folder)
    print(fileNameS)
    fileNameS.remove(fileName)      
    
    formatS = ['XLSX', 'CSV', 'JSON']
    for frmt in formatS:
        print('\n--- Если требуется найти ещё файл формата', frmt, 'для присоединения, то нажмите Enter'
              , '\n--- Если НЕ требуется, то введите любой символ и нажмите Enter')    
        if len(input()) == 0:
            fileNamesForImport = []
            for fileName in fileNameS:
                if ('.' + frmt.lower()) in fileName:
                    print('--- Найден файл', fileName, '. Если он подходит, то нажмите Enter'
                          , '\n--- Если искать дальше, то введите любой символ и нажмите Enter')
                    if len(input()) == 0:
                        fileNamesForImport.append(fileName)
                        print('--- Файл', fileName, 'учтён; если требуется найти ещё файл формата', frmt, ', то нажмите Enter'
                              , '\n--- Если НЕ требуется, то введите любой символ и нажмите Enter')
                        if len(input()) > 0:
                            break

            if len(fileNamesForImport) > 0:
                # print('fileNamesForImport', fileNamesForImport) # для отладки
                # print('folder', folder) # для отладки
                # print('fileName', fileName) # для отладки
                error = None
                try:
                    for fileName in fileNamesForImport:
                        if 'xlsx' in fileName: df = df.merge(pandas.read_excel(f'{folder}{fileName}', index_col=0), on='id', how='outer')
                        if 'csv' in fileName: df = df.merge(pandas.read_csv(f'{folder}{fileName}'), on='id', how='outer')
                        if 'json' in fileName: df = df.merge(pandas.read_json(f'{folder}{fileName}'), on='id', how='outer')
                        # if 'json' in fileName: df = pandas.concat([df, pandas.read_json(f'{folder}{fileName}')], axis=1)
                except FileNotFoundError:
                    errorDescription = sys.exc_info()
                    error = str(errorDescription[1])
                    print('error:', error) 
            else:
                print('Файлы форматов', formatS, 'не найдены в директории\n')
    if error == None:
        display(df.head())
        print('Число столбцов:', df.shape[1], ', число строк', df.shape[0], '\n')
    else:
        df = None
    return df, error, folder

def googleDriver2local(file_id, dest_path):
    GoogleDriveDownloader.download_file_from_google_drive(file_id=file_id, dest_path=dest_path)
    error = None
    try:
        df = pandas.read_excel(dest_path, index_col=0)
    except FileNotFoundError:
        errorDescription = sys.exc_info()
        error = str(errorDescription[1]).replace("No module named '", '').replace("'", '').replace('_', '')
        print('error:', error) 
        
    if error == None:
        display(df.head())
        print('Число столбцов:', df.shape[1], ', число строк', df.shape[0], '\n')
    else:
        df = None
    return df, error
