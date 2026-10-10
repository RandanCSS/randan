#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for adapting the current script to the CoLab file system
(RU) Модуль для адаптации текущего скрипта к файловой системе CoLab
'''

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
        from google.colab import drive
        import numpy, os
        break # выход из цикла while True

    except ModuleNotFoundError as excptn_1:
        module = excptn_1.name.split('.')[0]
        if attempt > MAX_ATTEMPTS:
            print(
f'Пакет {module} НЕ удалось импортировать за {MAX_ATTEMPTS} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт'
            )

            raise

        print(
f'Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас. Попытка № {attempt} из {MAX_ATTEMPTS}'
        )

        try: check_call([sys.executable, '-m', 'pip', 'install', module, '--quiet', '--disable-pip-version-check'])
        except CalledProcessError as excptn_2:
            print(f"Не удалось установить {module}. {type(excptn_2).__name__}: {str(excptn_2).split('Stacktrace:')[0].strip()}")
            raise

        attempt += 1
    
def coLabAdaptor():
    attempt = 0
    folderCoLab = None
    colabMode = False
    while True:
        try:
            from google.colab import drive
            print('Похоже, я исполняюсь в CoLab, поэтому сейчас появится окно с просьбой открыть доступ для сохранения результатов работы на Ваш Google Drive\n')
            colabMode = True
            drive.mount('/content/drive')
            folderCoLab = '/content/drive/MyDrive/Colab Notebooks'
            break
        except ModuleNotFoundError:
            attempt += 1
            if attempt == 2:
                # print('Похоже, я исполняюсь не в CoLab\n')
                break
    return folderCoLab

def folderCoLab_folderIn_comparison(folder)
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if not folder:
        folderCoLab = coLabAdaptor.coLabAdaptor() # либо '/content/drive/MyDrive/Colab Notebooks' , либо None
        if folderCoLab: folder = folderCoLab

    if folder: folder += slash
    return folder

