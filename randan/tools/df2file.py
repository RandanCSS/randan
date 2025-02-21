#!/usr/bin/env python
# coding: utf-8

'''
A module for saving a dataframe to a file of one of the formats: CSV, Excel and JSON. It facilitates working with data from social media
'''

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from randan.tools import varPreprocessor
        import os, pandas
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[1] 
        print('Пакет', module,
              'НЕ прединсталлируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталлирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print('Пакет', module
                  , 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,'
                  , 'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
            break

def df2file(dfIn, *arg): # арки: fileName и folder
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС

# ********** Выяснить поданные аргументы
    if len(arg) == 0:
        # print('len(arg) == 0')
        fileName = ''
        folder = ''

    if len(arg) == 1:
        # print('len(arg) == 1')
        fileName = arg[0]
        folder = ''

    if len(arg) == 2:
        # print('len(arg) == 2')
        fileName = arg[0]
        folder = arg[1]
        folder += slash

    if fileName == '':
        fileName = input('--- Впишите имя сохраняемого файла и нажмите Enter:')
    
    if folder != '':
        print(f'Директория, в которую сохраняю файл "{fileName}":', os.getcwd() + slash + folder)
    elif slash in fileName: # если директория содерджится в fileName
        folder = slash.join(fileName.split(slash)[:-1])
        fileName = fileName.split(slash)[-1]
        print(f'Директория, в которую сохраняю файл "{fileName}":', os.getcwd() + slash + folder)
    else:
        folder = input('--- Впишите директорию, в которую сохранить файл (если имя файла уже содержит путь к нему, то не вписывайте ничего) и нажмите Enter:')

    if slash != folder[-1]:
        folder += slash
# ********** Выяснить расширение сохраняемого файла
    # print('Имя сохраняемого файла:', fileName)
    fileFormatChoice = fileName.split('.')[-1]
    if (fileFormatChoice != 'xlsx') & (fileFormatChoice != 'csv') & (fileFormatChoice != 'json'):
        while True:
            print('--- Если хотите сохранить датафрейм в файл Excel, нажмите Enter;'
                  , '\n--- если же хотите в файл формата CSV или JSON, впишите букву "c" или "j" соответственно и нажмите Enter')
            fileFormatChoice = input()
            # print(folder + fileName.capitalize() + fileFormatChoice)
            if len(fileFormatChoice) == 0:
                fileFormatChoice = '.xlsx'
                break
            elif fileFormatChoice == 'c':
                fileFormatChoice = '.csv'
                break
            elif fileFormatChoice == 'j':
                fileFormatChoice = '.json'
                break
            else:
                print('--- Вы ввели что-то не то; попробуйте, пожалуйста, ещё раз..')
        fileName += fileFormatChoice

# ********** В зависимости от расширения сохраняемого файла выполнить сохранение
    # print('Расширение файла:', fileFormatChoice)
    # print('folder', folder) # для отладки
    # print('fileName', fileName) # для отладки
    if fileFormatChoice == 'xlsx':
        attempt = 0
        while True:
            try:
                dfIn.to_excel(folder + fileName)
                # print(folder + fileName)
                break
            except:
                errorDescription = sys.exc_info()
                print(errorDescription[1])
                if 'IllegalCharacterError' in str(errorDescription[0]):
                    module = 'xlsxwriter'
                    print('Для устранения ошибки требуется пакет', {module}, 'поэтому он будет инсталирован сейчас\n')
                    check_call([sys.executable, "-m", "pip", "install", module])
                    attempt += 1
                    if  attempt == 10:
                        print('Пакет', module
                              , 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,'
                              , 'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
                        break
                else: break
    if fileFormatChoice == 'csv':
        dfIn.to_csv(folder + fileName)   
    if fileFormatChoice == 'json':
        dfIn.to_json(folder + fileName)

def df2fileShell(complicatedNamePart, dfIn, fileFormatChoice, method, coLabFolder, currentMoment):
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    folder = currentMoment + complicatedNamePart
    if coLabFolder == None:
        print('Сохраняю выгрузку метода', method, '                              ') #, f'в директорию "{folder}"'
        if os.path.exists(folder) == False:
            print('Такой директории не существовало, поэтому она создана')
            os.makedirs(folder)
        # else:
            # print('Эта директория существует')
    else:
        print('Сохраняю выгрузку метода', method, '                              ') #, f'в директорию "{os.getcwd() + slash + coLabFolder + slash + folder}"'
        if os.path.exists(os.getcwd() + slash + coLabFolder + slash + folder) == False:
            print('Такой директории не существовало, поэтому она создана')
            os.makedirs(os.getcwd() + slash + coLabFolder + slash + folder)
        # else:
            # print('Эта директория существует')
    
    # df2file(itemS) # при такой записи имя сохранаяемого файла и директория, в которую сохранить, вводятся вручную
    # print('При сохранении возможно появление обширного предупреждения UserWarning: Ignoring URL.'
    #       , 'Оно вызвано слишком длинными URL-адресами в датафрейме и не является проблемой; его следует пролистать и перейти к диалоговому окну' )

    # Проверка всех столбцов на наличие в их ячейках JSON-формата
    columnsToJSON = varPreprocessor.jsonChecker(dfIn)

    # print('folder', folder) # для отладки
    if len(columnsToJSON) > 0:
        print('В выгрузке метода', method, 'есть столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
              , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')

        columnsToJSON.append('id')
        if 'from_id' in dfIn.columns: columnsToJSON.append('from_id')
        if 'owner_id' in dfIn.columns: columnsToJSON.append('owner_id')

        df2file(dfIn[columnsToJSON], f'{folder}_{method}_JSON_varS.json', folder if coLabFolder == None else coLabFolder + slash + folder)
        columnsToJSON.remove('id')
        if 'from_id' in columnsToJSON: columnsToJSON.remove('from_id')
        if 'owner_id' in columnsToJSON: columnsToJSON.remove('owner_id')

        df2file(dfIn.drop(columnsToJSON, axis=1), f'{folder}_{method}_Other_varS{fileFormatChoice}', folder if coLabFolder == None else coLabFolder + slash + folder)
    else: df2file(dfIn, f'{folder}_{method}{fileFormatChoice}', folder if coLabFolder == None else coLabFolder + slash + folder)
