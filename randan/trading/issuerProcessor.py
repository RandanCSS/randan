# Авторский модуль для операций с эмитентами торгуемых на МосБирже облигаций

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from randan.tools import coLabAdaptor, dictionariesHarmonizer, textPreprocessor # авторские модули для
        # (а) адаптации текущего скрипта к файловой системе CoLab
        # (б) редактирования столбца одного датафрейма на основе того же столбца другого датафрейма
        # (в) предобработки нестандартизированнрого текста

        import os, pandas, re
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
            rowS_bondS_withoutRating = bondS[(bondS['Эмитент'] == issuerS['Эмитент'][row_issuerS])\
                & (bondS['Bond D Rating'] == 'Рейтинг не присвоен или неизвестен, или отозван')].index # обрабатываемые строчки bondS

            rowS_bondS_withoutRating = list(rowS_bondS_withoutRating)
            rowS_bondS = list(rowS_bondS)
            rowS_bondS = [row_bondS for row_bondS in rowS_bondS if row_bondS not in rowS_bondS_withoutRating]
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

def issuerNameProcessor(bondS_in, issuerS):
    bondS = bondS_in.copy()

    # 1.1.0 Предобработка названий облигаций
    bondS_isna = bondS[bondS['SECNAME'].isna()]
    # display('bondS_isna:', bondS_isna) # для отладки

    # Создать столбец 'Эмитент'
    bondS_notna = issuerExtractor(bondS[bondS['SECNAME'].notna()])
        # bondS['SECNAME'].notna() , потому что отсекаются вышедшие из обращения облигации и акции

    # display('bondS_notna:', bondS_notna) # для отладки

    # 1.1.1 Поиск эмитентов, которые до сих пор отсутствуют в словарях; их внесение в словарь
    bondS_new_2 = bondS_notna.copy() # на каждой итерации bondS_New_2 будет сокращаться
    bondS_notna = pandas.DataFrame()

    if len(bondS_new_2) > 0:
        # for column in issuerS.columns[:1]: # для отладки
        for column in issuerS.columns: # issuerS -- это 'Словарь эмитентов' (без информации о рейтинге),
                    # в котором несколько столбцов, в названии которых слово 'Эмитент'
                # Причём более общие наименования одного и того же эмитента находятся в более правых столбцах
                    # и, как следствие, при объединении столбцов в один -- попадают в более нижние ячейки

            if 'Эмитент' in column:
                issuerS_byColumn = issuerS[issuerS[column].notna()][[column]] # текущий столбец эмитентов
                if len(issuerS_byColumn) > 0:
                    issuerS_byColumn = issuerS_byColumn.rename(columns={column: 'Эмитент'})
                    # display('issuerS_byColumn:', issuerS_byColumn) # для отладки

                    bondS_matching, bondS_new_1, bondS_new_2 =\
                        dictionariesHarmonizer.dictionariesHarmonizer(bondS_new_2, issuerS_byColumn, 'Эмитент')
                        # bondS_new_1 -- часть редактируемого датафрейма (df_editing), которая не прошла грубую сверку, но прошла тонкую сверку
                        # bondS_new_2 -- часть редактируемого датафрейма (df_editing), которая не прошла ни грубую, ни тонкую сверку
                    # display('bondS_new_2:', bondS_new_2) # для отладки

                    bondS_notna = pandas.concat([bondS_notna, bondS_matching, bondS_new_1])
                    # display('bondS_notna:', bondS_notna) # для отладки

            else: break

    if len(bondS_new_2) > 0:
        issuerS_new = issuersComposer(bondS_new_2)
        print('Эмитенты, до сих пор отсутствующие в словарях (их следует внести в Словарь эмитентов.xlsx):')
        display(issuerS_new)
        issuerS_new.to_excel('Новые эмитенты.xlsx', index=False)
        print(
'''До внесения их в словарь скрипт приостанавливается. После внесения перезапустите скрипт сначала или с текущего чанка
А сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'''
              )
        input()
        sys.exit()

    bondS_notna = pandas.concat([bondS_notna, bondS_new_2])
    # display('bondS_notna:', bondS_notna) # для отладки

    bondS = pandas.concat([bondS_notna, bondS_isna]) # теперь в bondS у каждой облигации указан эмитент с названием, соотнесённым со Словарём эмитентов
        # (все эти эмитенты представлены в issuerS)

    # display('bondS:', bondS) # для отладки

    return bondS
