#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for adapting the current script to the CoLab file system
(RU) Авторский модуль для адаптации текущего скрипта к файловой системе CoLab
'''

def coLabAdaptor():
    attempt = 0
    coLabFolder = None
    colabMode = False
    while True:
        try:
            from google.colab import drive
            print('Похоже, я исполняюсь в CoLab, поэтому сейчас появится окно с просьбой открыть доступ для сохранения результатов работы на Ваш Google Drive\n')
            colabMode = True
            from google.colab import drive
            drive.mount('/content/drive')
            coLabFolder = 'drive/MyDrive/Colab Notebooks'
            break
        except ModuleNotFoundError:
            attempt += 1
            if attempt == 2:
                # print('Похоже, я исполняюсь не в CoLab\n')
                break
    return coLabFolder
