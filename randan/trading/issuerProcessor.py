# Авторский модуль для операций с эмитентами торгуемых на МосБирже облигаций

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from randan.tools import coLabAdaptor, textPreprocessor # авторские модули для
        # (а) адаптации текущего скрипта к файловой системе CoLab и
        # (б) предобработки нестандартизированнрого текста
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

# Авторские функции..
    # .. компановки информации об эмитентах торгуемых на МосБирже облигаций в датафрейм (словарь)
def issuersComposer(bondS):
    issuerS = bondS[['Эмитент']].sort_values('Эмитент').drop_duplicates().reset_index(drop=True) # заготовка для датафрейма с актуальными эмитентами
    for row_issuerS in issuerS.index:
    # for row_issuerS in issuerS.index[:1]: # для отладки
        # print('row_issuerS:', row_issuerS) # для отладки
        rowS_bondS = bondS[bondS['Эмитент'] == issuerS['Эмитент'][row_issuerS]].index # обрабатываемые строчки bondS
        secNameS = bondS['SECNAME'][rowS_bondS].tolist()
        # print('secNameS:', secNameS) # для отладки
        issuerS.loc[row_issuerS, 'Count'] = len(secNameS)
        issuerS.loc[row_issuerS, 'SecNameS'] = ''
        issuerS.at[row_issuerS, 'SecNameS'] = secNameS

        if 'Bond D Rating' in bondS.columns:
            ratingS = bondS['Bond D Rating'][rowS_bondS].dropna().tolist()
            # print('ratingS:', ratingS) # для отладки
            ratingS_mean = bondS['Bond D Rating'][rowS_bondS].dropna().mean()
            # print('ratingS_mean:', ratingS_mean) # для отладки
            issuerS.loc[row_issuerS, 'RatingS'] = ''
            issuerS.at[row_issuerS, 'RatingS'] = ratingS[0] if len(ratingS) == 1 else ratingS
            issuerS.loc[row_issuerS, 'Issuer D Rating'] = ratingS_mean

    # display('issuerS:', issuerS) # для отладки
    return issuerS

    # .. извлечения из SECNAME торгуемых на МосБирже облигаций названий их эмитентов
def issuerExtractor(dfIn):
    df = dfIn.copy()
    df['Эмитент'] = df['SECNAME'].str.replace('_', ' ').str.replace('-', ' ')
    df['Эмитент'] = df['Эмитент'].str.replace('ЯНАО', 'ЯНАвОк')
    df.loc[df['Эмитент'].str.contains('ИОС'), 'Эмитент'] = 'СберИОС'
    df['Эмитент'] = df['Эмитент'].apply(lambda cellContent: re.sub(r' [БЗ][OОPР]П?.+', '', cellContent))
    df['Эмитент'] = df['Эмитент'].apply(lambda cellContent: re.sub(r'Б\d+', '', cellContent))
    df['Эмитент'] = df['Эмитент'].apply(lambda cellContent: re.sub(r' 0\d+.*', '', cellContent))
    df['Эмитент'] = df['Эмитент'].apply(lambda cellContent: re.sub(r' \d+ обл\.?', '', cellContent))
    df['Эмитент'] = df['Эмитент'].apply(textPreprocessor.simbolsCleaner)
    df['Эмитент'] = df['Эмитент'].str.replace('ПАО ', '').str.replace(' ПАО', '')\
        .str.replace('АО ', '').str.replace(' АО', '')\
        .str.replace('ООО ', '').str.replace(' ООО', '')\
        .str.replace('"', '')
    df['Эмитент'] = df['Эмитент'].apply(lambda cellContent: cellContent if cellContent[0] != ' ' else cellContent[1:])
    df['Эмитент'] = df['Эмитент'].apply(lambda cellContent: cellContent if cellContent[-1] != ' ' else cellContent[:-1])
    df['Эмитент'] = df['Эмитент'].str.split(' Б ').str[0]
    df['Эмитент'] = df['Эмитент'].str.lower()
    df.loc[df['Эмитент'].str.contains('офз'), 'Эмитент'] = 'минфин рф'
    df.loc[(df['Эмитент'].str.contains('воз')) & (df['Эмитент'].str.contains('рф')), 'Эмитент'] = 'минфин рф'
    return df
