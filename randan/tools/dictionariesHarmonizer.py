#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for editing one dataframe (df_forEditing) within its specific column based on the same column from another dataframe (df_standard)
(RU) Можель для изменения некоторого столбца одного датафрейма (df_forEditing) на основе такого же столбца другого датафрейма (df_standard)
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
        from pandas import DataFrame
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
