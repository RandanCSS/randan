#!/usr/bin/env python
# coding: utf-8

'''
A module for editing one dataframe (df_small) within its specific column based on the same column from another dataframe (df_large)
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

def dictionariesHarmonizer(df_large, dfIn_Small, columnName):
    df_small = dfIn_Small.copy() # df_small -- датафрейм, редактируемый в столбце columnName на основе того же столбца датафрейма df_large

    # Шаг № 1. Грубая сверка
    df_small_matching = df_small[df_small[columnName].isin(df_large[columnName])]
    df_small_New_1 = df_small[df_small[columnName].isin(df_large[columnName]) != True]

    # Шаг № 2. Тонкая сверка
    rowS_toDrop = []
    df_small_New_2 = df_small_New_1.copy()
    elementS_Small = df_small_New_1[columnName]
    for element_Small in elementS_Small:
        for element_Large in df_large[columnName]:
            if element_Large in element_Small:
                df_small_New_1.loc[df_small_New_1[columnName] == element_Small, columnName] = element_Large # заменить element_Small на element_Large ,
                    # что обеспечивает совместимость обрабатываемых тут строчек df_small_New_1 b df_large
                df_small_New_2 = df_small_New_2[df_small_New_2[columnName] != element_Small] 
    return df_small_matching, df_small_New_1, df_small_New_2
    # df_small_New_1 -- часть редактируемого датафрейма (df_small), которая не прошла грубую сверку, но прошла тонкую сверку
    # df_small_New_2 -- часть редактируемого датафрейма (df_small), которая не прошла ни грубую, ни тонкую сверку
