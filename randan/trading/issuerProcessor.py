# Авторский модуль для операций с названиями эмитентов торгуемых на МосБирже облигаций

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from randan.tools import coLabAdaptor, textPreprocessor # авторские модули для (а) адаптации текущего скрипта к файловой системе CoLab и (б) предобработки нестандартизированнрого текста
        import os, pandas, re
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

coLabFolder = coLabAdaptor.coLabAdaptor()

# 1. Авторские функции
    # извлечения из SECNAME торгуемых на МосБирже облигаций названий их эмитентов
def issuerExtractor(dfIn):
    df = dfIn.copy()
    df['Эмитент'] = df['SECNAME'].str.replace('_', ' ').str.replace('-', ' ')
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r' [БЗ][OОPР]П?.+', '', text))
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r'Б\d+', '', text))
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r' 0\d+.*', '', text))
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r' \d+ обл\.?', '', text))
    df['Эмитент'] = df['Эмитент'].apply(textPreprocessor.simbolsCleaner)
    df['Эмитент'] = df['Эмитент'].str.replace('ПАО ', '').str.replace(' ПАО', '')\
        .str.replace('АО ', '').str.replace(' АО', '')\
        .str.replace('ООО ', '').str.replace(' ООО', '')\
        .str.replace('"', '')
    df['Эмитент'] = df['Эмитент'].apply(lambda text: text if text[0] != ' ' else text[1:])
    df['Эмитент'] = df['Эмитент'].apply(lambda text: text if text[-1] != ' ' else text[:-1])
    df['Эмитент'] = df['Эмитент'].str.split(' Б ').str[0]
    df['Эмитент'] = df['Эмитент'].str.lower()
    df.loc[df['Эмитент'].str.contains('офз'), 'Эмитент'] = 'минфин рф'
    df.loc[(df['Эмитент'].str.contains('воз')) & (df['Эмитент'].str.contains('рф')), 'Эмитент'] = 'минфин рф'
    return df
