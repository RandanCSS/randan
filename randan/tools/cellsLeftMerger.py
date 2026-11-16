#!/usr/bin/env python
# coding: utf-8

'''
(EN) A proprietary module that facilitates left-merging a source dataframe into a target dataframe on a specified merge column
(RU) Авторский модуль для упрощения операции левостороннего присоединения датафрейма-донора к датафрейму-реципиенту по специальному столбцу 
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
        from IPython.display import display
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

def cellsLeftMerger(df_source, df_target_in, merge_column):
    df_target = df_target_in.copy()
    df_target = df_target.reset_index().rename(columns={'index': 'indexOriginal'}) # сохранить индекс как новый столбец indexOriginal
    df_target = df_target.merge(df_source, how='left', on=merge_column, suffixes=('', '_drop'))

    columnS_toDrop = []
    for column in df_target.columns:
        if '_drop' in column:
            # Замена старых значений новыми только там, где новые не NaN
            try: df_target[column.replace('_drop', '')] = df_target[column].combine_first(df_target[column.replace('_drop', '')])
            except: # учесть потенциальную ошибку year 0 is out of range: 0000-00-00, at position 0
                df_target[column] = df_target[column].astype(str)
                df_target[column.replace('_drop', '')] = df_target[column.replace('_drop', '')].astype(str)
                df_target[column.replace('_drop', '')] = df_target[column].combine_first(df_target[column.replace('_drop', '')])

            columnS_toDrop.append(column)

    df_target = df_target.drop(columnS_toDrop, axis=1)
    df_target = df_target.set_index('indexOriginal') # восстановить индекс из столбца indexOriginal
    return df_target
