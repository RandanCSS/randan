#!/usr/bin/env python
# coding: utf-8

'''
A module for editing one dataframe (df_editing) within its specific column based on the same column from another dataframe (df_standard)
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from pandas import DataFrame
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
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

def dictionariesHarmonizer(df_editing, df_standard, columnName):
    df_editing = df_editing.copy() # df_editing -- датафрейм, редактируемый в столбце columnName на основе того же столбца датафрейма df_standard

    # Шаг № 1. Грубая сверка
    df_editing_matching = df_editing[df_editing[columnName].isin(df_standard[columnName])]
    df_editing_New_1 = df_editing[df_editing[columnName].isin(df_standard[columnName]) != True]

    # Шаг № 2. Тонкая сверка
    rowS_toDrop = []
    df_editing_New_2 = df_editing_New_1.copy()
    elementS_editing = df_editing_New_1[columnName]
    for element_editing in elementS_editing:
        # print('element_editing:', element_editing, end='\r') # для отладки
        for element_standard in df_standard[columnName]:
            # print('element_standard:', element_standard, end='\r') # для отладки
            if element_standard in element_editing:
                df_editing_New_1.loc[df_editing_New_1[columnName] == element_editing, columnName] = element_standard # заменить element_editing на element_standard ,
                    # что обеспечивает совместимость обрабатываемых тут строчек df_editing_New_1 и df_standard
                df_editing_New_2 = df_editing_New_2[df_editing_New_2[columnName] != element_editing] 
    return df_editing_matching, df_editing_New_1, df_editing_New_2
    # df_editing_New_1 -- часть редактируемого датафрейма (df_editing), которая не прошла грубую сверку, но прошла тонкую сверку
    # df_editing_New_2 -- часть редактируемого датафрейма (df_editing), которая не прошла ни грубую, ни тонкую сверку
