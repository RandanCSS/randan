#!/usr/bin/env python
# coding: utf-8

'''
A module for saving a dataframe to a file of one of the formats: CSV, Excel and JSON. It facilitates working with data from social media
'''

# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        import os, pandas
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[1] 
        print('Пакет', module,
              'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

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
        print('Директория, в которую сохраняю файл:', folder)
    elif slash in fileName: # если директория содерджится в fileName
        folder = slash.join(fileName.split(slash)[:-1])
        print('Директория, в которую сохраняю файл:', folder)
        fileName = fileName.split(slash)[-1]
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
    if fileFormatChoice == 'xlsx':
        # try:
        dfIn.to_excel(folder + fileName)
        # print(folder + fileName)
        # except pandas.errors.IllegalCharacterError:
        #     print(sys.exc_info()[1])
        #     module = 'xlsxwriter'
        #     print('Для устранения ошибки требуется пакет', {module}, 'поэтому он будет инсталирован сейчас\n')
        #     check_call([sys.executable, "-m", "pip", "install", module])
    if fileFormatChoice == 'csv':
        dfIn.to_csv(folder + fileName)   
    if fileFormatChoice == 'json':
        dfIn.to_json(folder + fileName)
