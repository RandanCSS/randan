# coding: utf-8

'''
A proprietary module that facilitates left-merging a source dataframe into a target dataframe on a specified merge column
Авторский модуль для упрощения операции левостороннего присоединения датафрейма-донора к датафрейму-реципиенту по специальному столбцу 
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from IPython.display import display
        from randan.tools import coLabAdaptor # авторский модуль для..
            # (а) адаптации текущего скрипта к файловой системе CoLab

        import pandas
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
        check_call([sys.executable, "-m", "pip", "install", module])
        if  attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break # выход из цикла for attempt in range(3)

coLabFolder = coLabAdaptor.coLabAdaptor()

def cellsLeftMerger(df_source, df_target_in, merge_column):
  df_target = df_target_in.copy()
  df_target = df_target.reset_index().rename(columns={'index': 'indexOriginal'}) # сохранить индекс как новый столбец indexOriginal
  df_target = df_target.merge(df_source, how='left', on=merge_column, suffixes=('', '_drop'))

  columnS_toDrop = []
  for column in df_target.columns:
      if '_drop' in column:
          df_target[column.replace('_drop', '')] = df_target[column].combine_first(df_target[column.replace('_drop', '')])
              # замена старых значений новыми только там, где новые не NaN

          columnS_toDrop.append(column)

  df_target = df_target.drop(columnS_toDrop, axis=1)
  df_target = df_target.set_index('indexOriginal') # восстановить индекс из столбца indexOriginal
  return df_target
