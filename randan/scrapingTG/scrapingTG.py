#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies scraping TG data
(RU) Модуль для упрощения скрапинга данных из TG
'''

# 0. Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# 0.0 В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        import pandas
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print(
f'''Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 10
'''
              )
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break

# 1. Авторские функции
# 1.0 парсинга столбца messages
def parser_messages_TG(folder_fileName):
    df = pandas.read_json(folder_fileName)
    df = pandas.json_normalize(df['messages']) if len(df['messages']) > 0 else pandas.DataFrame(columns='messages')
    return df

# 1.1 парсинга ячеек столбца text
def parser_text_TG(cellContent):
    # text = ''
    if type(cellContent) == list:
        text = ' '.join(pandas.json_normalize(cellContent)['text'].dropna().tolist())
        # for element in cellContent:
        #     # print('element:', element)        
        #     if type(element) == str:
        #         text += ' ' + element
        #         # print('    text:', text)
        #     elif type(element) == dict:
        #         text += ' ' + ' '.join(pandas.json_normalize(element)['text'].tolist())
        #         # print('    text:', text)
    elif type(cellContent) == str: text = cellContent # ... обработан случай, если в ячейке просто текст
    return text
