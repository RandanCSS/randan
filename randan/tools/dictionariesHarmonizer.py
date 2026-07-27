#!/usr/bin/env python
# coding: utf-8

'''
A module for editing one dataframe (df_forEditing) within its specific column based on the same column from another dataframe (df_standard)
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from pandas import DataFrame
        break # выход из цикла for attempt in range(3)

    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        print(
f'''Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 3
'''
              )
        check_call([sys.executable, "-m", "pip", "install", module])
        if  attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break

def dictionariesHarmonizer(df_forEditing_in, df_standard, columnName):
    df_forEditing = df_forEditing_in.copy() # df_forEditing -- датафрейм, редактируемый в столбце columnName на основе того же столбца датафрейма df_standard

    # Шаг № 1. Грубая сверка
    df_forEditing_matching = df_forEditing[df_forEditing[columnName].isin(df_standard[columnName])] # совпадающие строки датафрейма
    df_forEditing_new_1 = df_forEditing[df_forEditing[columnName].isin(df_standard[columnName]) != True] # несовпадающие строки датафрейма

    # Шаг № 2. Тонкая сверка
    rowS_detected = [] # только эти строки датафрейма останутся в df_forEditing_new_1
    df_forEditing_new_2 = df_forEditing_new_1.copy()
    elementS_forEditing = df_forEditing_new_1[columnName]

    # Два цикла для сверки поячеечно столбцов df_forEditing_new_1[columnName] и df_standard[columnName]
    for element_forEditing in elementS_forEditing:
        # print('element_forEditing:', element_forEditing) # для отладки , end='\r'
        for element_standard in df_standard[columnName]:
            # print('element_standard:', element_standard) # для отладки , end='\r'
    
            if element_standard in element_forEditing:
                # print('element_standard in element_forEditing:', element_standard in element_forEditing) # для отладки , end='\r'
                rowS_detected.extend(df_forEditing_new_1[df_forEditing_new_1[columnName] == element_forEditing].index)
                df_forEditing_new_1.loc[df_forEditing_new_1[columnName] == element_forEditing, columnName] = element_standard # заменить element_forEditing на element_standard ,
                    # что обеспечивает совместимость обрабатываемых тут ячеек df_forEditing_new_1[columnName] и df_standard[columnName]

                df_forEditing_new_2 = df_forEditing_new_2[df_forEditing_new_2[columnName] != element_forEditing]

    rowS_detected = list(set(rowS_detected))
    rowS_detected.sort
    df_forEditing_new_1 = df_forEditing_new_1.loc[rowS_detected, :]

    return df_forEditing_matching, df_forEditing_new_1, df_forEditing_new_2
    # df_forEditing_new_1 -- часть редактируемого датафрейма (df_forEditing), которая не прошла грубую сверку, но прошла тонкую сверку
    # df_forEditing_new_2 -- часть редактируемого датафрейма (df_forEditing), которая не прошла ни грубую, ни тонкую сверку
