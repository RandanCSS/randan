#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies scraping TG data
(RU) Модуль для упрощения скрапинга данных из TG
'''

# Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# subprocess & sys -- эти пакеты обычно предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
from subprocess import check_call, CalledProcessError
import sys

# --- остальные модули и пакеты
MAX_ATTEMPTS = 3
attempt = 1

while True:
    try:
        import pandas
        break # выход из цикла while True

    except (ImportError, ModuleNotFoundError) as excptn_1:
        module = (excptn_1.name or '').split('.')[0]
        if attempt > MAX_ATTEMPTS:
            print(
f'Пакет {module} НЕ удалось импортировать за {MAX_ATTEMPTS} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт')

            raise

        print(
f'Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас. Попытка № {attempt} из {MAX_ATTEMPTS}')

        try: check_call([sys.executable, '-m', 'pip', 'install', module, '--quiet', '--disable-pip-version-check'])
        except CalledProcessError as excptn_2:
            print(f"Не удалось установить {module}. {type(excptn_2).__name__}: {str(excptn_2).split('Stacktrace:')[0].strip()}")
            raise

        attempt += 1

# 1. Функции..
# 1.0 .. парсинга столбца messages
def parser_messages_TG(folder_fileName):
    df = pandas.read_json(folder_fileName)
    df = pandas.json_normalize(df['messages']) if len(df['messages']) > 0 else pandas.DataFrame(columns='messages')
    return df

# 1.1 .. парсинга ячеек столбца text
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
