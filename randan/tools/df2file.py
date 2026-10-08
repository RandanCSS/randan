#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for exporting a dataframe to a file of one of the formats: CSV, Excel and JSON
(RU) Модуль для экспорта датафрейма в файл одного из флорматов: CSV, Excel и JSON
'''

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from randan.tools import textPreprocessor, varPreprocessor
        import os, pandas
        break # выход из цикла for attempt in range(3)

    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0] 
        print(
f'''Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 3
'''
              )
        check_call([sys.executable, '-m', 'pip', 'install', module])
        if attempt == 3: print(
f'''Пакет {module} НЕ удалось импортировать за {attempt} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
        )

def df2file(df, *arg): # арки: fileName и folder
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
        fileName = input('--- Впишите имя сохраняемого файла (с расширением) и нажмите Enter:')
    
    if folder != '':
        print(f"Директория, в которую сохраняю файл '{fileName}':", os.getcwd() + slash + folder)

    elif slash in fileName: # если директория содерджится в fileName
        folder = slash.join(fileName.split(slash)[:-1])
        fileName = fileName.split(slash)[-1]
        print(f"Директория, в которую сохраняю файл '{fileName}':", os.getcwd() + slash + folder)
    else:
        folder = input('--- Впишите директорию, в которую сохранить файл (если имя файла уже содержит путь к нему, то не вписывайте ничего) и нажмите Enter:')

    if folder and not folder.endswith(slash):
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
                fileFormatChoice = 'xlsx'
                break
            elif fileFormatChoice == 'c':
                fileFormatChoice = 'csv'
                break
            elif fileFormatChoice == 'j':
                fileFormatChoice = 'json'
                break
            else:
                print('--- Вы ввели что-то не то; попробуйте, пожалуйста, ещё раз..')
        fileName += '.' + fileFormatChoice

# ********** В зависимости от расширения сохраняемого файла выполнить сохранение
    # print('Расширение файла:', fileFormatChoice)
    # print('folder', folder) # для отладки
    # print('fileName', fileName) # для отладки
    if fileFormatChoice == 'xlsx':
        textColS = df.select_dtypes(include=['object', 'string']).columns
        for column in textColS: df[column] = df[column].apply(textPreprocessor.dropControlCharacters)
            # чистка текстов от control characters (недопустимых при экспорте в файлы формата типа Excel)

        for attempt in range(1, 4):
            try:
                df.to_excel(folder + fileName, engine='xlsxwriter', index=False)
                # print(folder + fileName)
                break # выход из цикла for attempt in range(3)

            except:
                errorDescription = sys.exc_info()
                print(errorDescription[1])
                if 'IllegalCharacterError' in str(errorDescription[0]):
                    module = 'xlsxwriter'
                    print('Для устранения ошибки требуется пакет', {module}, 'поэтому он будет инсталирован сейчас\n')
                    check_call([sys.executable, "-m", "pip", "install", module])
                    if attempt == 3: print(
f'''Пакет {module} НЕ удалось импортировать за {attempt} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                    )

    if fileFormatChoice == 'csv':
        textColS = df.select_dtypes(include=['object', 'string']).columns
        for column in textColS: df[column] = df[column].apply(textPreprocessor.dropControlCharacters)
            # чистка текстов от control characters (недопустимых при экспорте в файлы формата типа Excel)

        df.to_csv(folder + fileName, encoding='utf-8-sig', index=False)

    if fileFormatChoice == 'json': df.to_json(folder + fileName, force_ascii=False, indent=4, orient='records')

def df2fileShell(complicatedNamePart, currentMoment, dfIn, fileFormatChoice, folder, method):
    # slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    # folder = currentMoment + complicatedNamePart
    # if coLabFolder == None:
    #     print('Сохраняю выгрузку метода', method, '                              ') #, f'в директорию "{folder}"'
    #     if os.path.exists(folder) == False:
    #         print('Такой директории не существовало, поэтому она создана')
    #         os.makedirs(folder)
    #     # else:
    #         # print('Эта директория существует')
    # else:
    #     print('Сохраняю выгрузку метода', method, '                              ') #, f'в директорию "{os.getcwd() + slash + coLabFolder + slash + folder}"'
    #     if os.path.exists(os.getcwd() + slash + coLabFolder + slash + folder) == False:
    #         print('Такой директории не существовало, поэтому она создана')
    #         os.makedirs(os.getcwd() + slash + coLabFolder + slash + folder)
    #     # else:
    #         # print('Эта директория существует')

    # Проверка всех столбцов на наличие в их ячейках JSON-формата
    columnsToJSON = varPreprocessor.jsonChecker(dfIn)

    print('folder', folder) # для отладки
    if len(columnsToJSON) > 0:
        print('В выгрузке метода', method, 'есть столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
              , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')

        columnsToJSON.append('id')
        if 'from_id' in dfIn.columns: columnsToJSON.append('from_id')
        if 'owner_id' in dfIn.columns: columnsToJSON.append('owner_id')

        print('columnsToJSON:', columnsToJSON) # для отладки
        df2file(dfIn[columnsToJSON], f'{folder}_{method}_JSON_varS.json', folder)
        columnsToJSON.remove('id')
        if 'from_id' in columnsToJSON: columnsToJSON.remove('from_id')
        if 'owner_id' in columnsToJSON: columnsToJSON.remove('owner_id')

        df2file(dfIn.drop(columnsToJSON, axis=1), f'{folder}_{method}_Other_varS.{fileFormatChoice}', folder)
    else: df2file(dfIn, f'{folder}_{method}.{fileFormatChoice}', folder)
