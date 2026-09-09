# coding: utf-8

'''
A module to import and process bonds' feachures from the Moscow Exchange
Модуль для выгрузки характеристик торгуемых на МосБирже акций, облигаций, фьючерсов
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        # from bs4 import BeautifulSoup
        from IPython.display import display
        from tqdm import tqdm
        import os, pandas, requests, time, traceback, warnings # , re
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

        check_call([sys.executable, '-m', 'pip', 'install', module])
        if  attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )

# 1. Вспомогательные функции..
# .. выгрузки таблиц -- фрагментов данных формата JSON из БД МосБиржи
def json2df(columnS_forComparisom, headers, pause, sectionOfJson, url):
    df = pandas.DataFrame()
    df_additional_previous = pandas.DataFrame()
    start = 0
    while True:
        # print('start:', start, '                    ', end='\r') # для отладки
        params = {'start': start} # 'limit': 100,

        try: data_json = requests.get(url, headers=headers, params=params).json()
        except Exception as excptn:
            print('Exception 1 в json2df') # для отладки
            print(f'{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}') # для отладки
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой
            time.sleep(pause)

        df_additional = pandas.DataFrame(columns=data_json[sectionOfJson]['columns'], data=data_json[sectionOfJson]['data'])
        # df_additional = df_additional.fillna('Нет данных')
        # display('df_additional:', df_additional) # для отладки

        try:
            if (len(df_additional) == 0) | ((df_additional[columnS_forComparisom] != df_additional_previous).sum().sum() == 0):
                    # во второй части условия проверяется наличие различия между датафреймами хотя бы в одной ячейке

                # print('Похоже, df_additional == df_additional_previous; завершаю цикл') # для отладки
                break

        except Exception as excptn:
            # print('Exception 2 в json2df') # для отладки
            # print(f'{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}') # для отладки
            # print(traceback.format_exc()) # показ точной строчки кода с ошибкой
            # print('Похоже, df_additional != df_additional_previous; продолжаю итерировать') # для отладки
            pass

        df = pandas.concat([df, df_additional])
        df_additional_previous = df_additional[columnS_forComparisom]
        start += len(df_additional)

    # break # для отладки
    df = pandas.DataFrame(columns=data_json[sectionOfJson]['columns'], data=data_json[sectionOfJson]['data'])
    # display('df:', df) # для отладки
    return df

# .. работы с дубликатами в рамках одного и того же SECID
def securities_marketdata_df_duplicated_withinIsin_processor(securities_marketdata_df_duplicated_withinIsin):
    conditionS = [
        (((securities_marketdata_df_duplicated_withinIsin['BOARDNAME'].str.contains('Акции и ДР', case=False)) |\
          (securities_marketdata_df_duplicated_withinIsin['BOARDNAME'].str.contains('облигации', case=False))) &\
         (securities_marketdata_df_duplicated_withinIsin['BOARDNAME'].str.contains('- безадрес.', case=False))),
        (securities_marketdata_df_duplicated_withinIsin['CURRENCYID'] == 'SUR'),
        (securities_marketdata_df_duplicated_withinIsin['CURRENCYID'] == 'CNY')
        ]

    # for condition in conditionS[:2]: # для отладки
    for condition in conditionS:
        # print('condition:', condition) # для отладки
        if (sum(condition) > 0) & (sum(condition) < len(securities_marketdata_df_duplicated_withinIsin)):
            # тут и ниже первая часть условия проверыет применимость всего условия; вторая часть условия: если в рассматриваемом столбце
                # не только проверяемые тексты, то строки securities_marketdata_df_duplicated_withinIsin с остальными значениями не нужны

            securities_marketdata_df_duplicated_withinIsin = securities_marketdata_df_duplicated_withinIsin[condition]

    return securities_marketdata_df_duplicated_withinIsin

# .. работы со срезом securities_marketdata_df , содержащим дублирующиеся по SECID строки
def securities_marketdata_df_duplicates_processor(columnS_withDateTime, securities_marketdata_df, securities_marketdata_df_duplicated):
    securities_marketdata_df_notDuplicated =\
        securities_marketdata_df[~securities_marketdata_df['SECID'].isin(securities_marketdata_df_duplicated['SECID'])]

    # display('securities_marketdata_df_notDuplicated:', securities_marketdata_df_notDuplicated) # для отладки

    # print(len(securities_marketdata_df_duplicated) + len(securities_marketdata_df_notDuplicated) == len(securities_marketdata_df))

    securities_marketdata_df_duplicated_secidS = list(securities_marketdata_df_duplicated['SECID'].unique())
    securities_marketdata_df_duplicated_secidS.sort()
    # print(securities_marketdata_df_duplicated_secidS) # для отладки

    for secid in tqdm(securities_marketdata_df_duplicated_secidS):
        # print('secid:', secid) # для отладки

        securities_marketdata_df_duplicated_withinIsin =\
            securities_marketdata_df_duplicated[securities_marketdata_df_duplicated['SECID'] == secid]

        securities_marketdata_df_duplicated_withinIsin =\
            securities_marketdata_df_duplicated_withinIsin_processor(securities_marketdata_df_duplicated_withinIsin)

        columnsWithDifferences = securities_marketdata_df_duplicated_withinIsin.columns[
            securities_marketdata_df_duplicated_withinIsin.nunique() > 1
            ].tolist()

        # print('columnsWithDifferences:', columnsWithDifferences) # для отладки

        conditionNext = True

        # либо выбрать наиболее свежую запись по одному из столбцов с DateTime
        for column_withDateTime in columnS_withDateTime:
            if column_withDateTime in columnsWithDifferences:
                securities_marketdata_df_duplicated_withinIsin =\
                    securities_marketdata_df_duplicated_withinIsin.sort_values(column_withDateTime).iloc[[-1], :]

                conditionNext = False
                break

        if conditionNext:# либо усреднить значения по столбцам columnsWithDifferences ,..
            print('columnsWithDifferences:', columnsWithDifferences) # для отладки
            display(securities_marketdata_df_duplicated_withinIsin[columnsWithDifferences]) # для отладки

            meanS = securities_marketdata_df_duplicated_withinIsin[columnsWithDifferences].mean()
            # display('meanS:', meanS) # для отладки

            for column in columnsWithDifferences: # .. импутировать их в securities_marketdata_df_duplicated_withinIsin и..
                securities_marketdata_df_duplicated_withinIsin[column] = meanS[column]

            securities_marketdata_df_duplicated_withinIsin = securities_marketdata_df_duplicated_withinIsin.drop_duplicates()
                # .. удалить дубликаты

        if len(securities_marketdata_df_duplicated_withinIsin) != 1:
            print('secid инструментов, имеющих после обработки дубликатов 0 записей или более 1 записи:', secid) # для отладки
            display(securities_marketdata_df_duplicated_withinIsin[columnsWithDifferences] if columnsWithDifferences else securities_marketdata_df_duplicated_withinIsin)

        securities_marketdata_df_notDuplicated = pandas.concat([securities_marketdata_df_notDuplicated, securities_marketdata_df_duplicated_withinIsin])

    # display('securities_marketdata_df_notDuplicated:', securities_marketdata_df_notDuplicated) # для отладки
    return securities_marketdata_df_notDuplicated

# 2. Основная функция
def getMoExData(folder=coLabFolder,
                market='bonds',
                pause=0.1,
                plusNotTraded=False,
                returnDfs=False):
    '''
    Функция умеет выгружать характеристики торгуемых на МосБирже акций, облигаций, фьючерсов. Дополнительно выгружается словарь полей БД МосБиржи.

    Parameters
    ----------
       folder : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

       market : str -- если интересуют облигации, подходит значение по умолчанию 'bonds' , если фьючерсы, впишите 'forts' , если акции, впишите 'shares'
        pause : float -- длительность приостановки исполнения скрипта в секундах
plusNotTraded : bool -- в случае True функция возвращает и неторгуемые securities
    returnDfs : bool -- в случае True функция возвращает итоговые датафреймы boardS, columnsDescriptionS и securities_marketdata_df строго в такой последовательности
    '''
    headers = {'User-Agent': 'Mozilla/5.0'}

    # Блок, поскольку folder многократно используется внутри функции в формулах
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    # if folder: print('folder до:', folder) # для отладки
    if (folder == None) | (folder == ''): folder = ''
    else: folder += slash
    # if folder: print('folder после:', folder) # для отладки

# 2.0 Проверка наличия комплекта файлов и вопрос про необходимость его обновления
    path_boards = folder + market + ' Boards.xlsx'
    path_columnsDescriptions = folder + market + ' Columns descriptions.xlsx'
    path_securities_marketdata = folder + market + ' Securities and Marketdata.xlsx'
    if os.path.exists(path_boards) & os.path.exists(path_columnsDescriptions) & os.path.exists(path_securities_marketdata):

        print(
f'''--- Комплект файлов:
'{path_boards}' -- режимы торгов
'{path_columnsDescriptions}' -- словарь полей БД МосБиржи
'{path_securities_marketdata}' -- доступные инструменты (securities) и их финансовые данные (marketdata или marketdata_yields)
существует; если хотите обновить этот комплект, просто нажмите Enter (это недолго)
--- Если хотите НЕ обновить, то нажмите пробел и затем Enter'''
              )

        decision = input()
        if decision:
            print('Использую существующий комплект')
            if returnDfs:
                boardS = pandas.read_excel(path_boards)
                columnsDescriptionS = pandas.read_excel(path_columnsDescriptions)
                securities_marketdata_df = pandas.read_excel(path_securities_marketdata)
                return boardS, columnsDescriptionS, securities_marketdata_df

# 2.1 Если нет комплекта
# 2.1.0 Формирование файла с режимами торгов boardS
    print('Создаю файл с режимами торгов')
    if (market == 'bonds') | (market == 'shares'): url = f'https://iss.moex.com/iss/engines/stock/markets/{market}'
    if market == 'forts': url = f'https://iss.moex.com/iss/engines/futures/markets/{market}'
    boardS = json2df(['id'], headers, pause, 'boards', url + '.json')
    boardS.to_excel(path_boards, index=False)
    # display('boardS 1:', boardS) # для отладки

    if not plusNotTraded: boardS = boardS[boardS['is_traded'].astype(int) == 1]
    # display('boardS 2:', boardS) # для отладки

# 2.1.1 Формирование словаря полей БД МосБиржи и файла с доступными инструментами (securities)
        # и их финансовыми данными (marketdata или marketdata_yields)
    print(
'Создаю словарь полей БД МосБиржи и файл с доступными инструментами (securities) и их финансовыми данными (marketdata или marketdata_yields)'
        )

    columnsDescriptionS = pandas.DataFrame()

    sectionOfJson_list = ['securities']
    if market == 'bonds': sectionOfJson_list.append('marketdata_yields')
    if (market == 'forts') | (market == 'shares'): sectionOfJson_list.append('marketdata')

    securities_marketdata_df = pandas.DataFrame()
    for sectionOfJson in tqdm(sectionOfJson_list):
        # print('sectionOfJson:', sectionOfJson) # для отладки

        # <Формирование словаря полей БД МосБиржи>
        columnsDescriptionS_additional = json2df(['id'], headers, pause, sectionOfJson, url + '.json')
        # columnsDescriptionS_additional = pseudojson2df(headerS, index, url)
        columnsDescriptionS_additional.loc[:, 'data id'] = sectionOfJson
        # columnsDescriptionS_additional.loc[:, 'data id'] = index

        if columnsDescriptionS.empty:
            print("columnsDescriptionS пуст")
            columnsDescriptionS = columnsDescriptionS_additional.copy()

        else: columnsDescriptionS = pandas.concat([columnsDescriptionS, columnsDescriptionS_additional], ignore_index=True)
        # </Формирование словаря полей БД МосБиржи>

        # <Формирование файла с доступными securities в интересующих режимах торгов>
        securities_marketdata_df_additional_1 = pandas.DataFrame()
        securities_marketdata_df_additional_2 = json2df(['SECID', 'BOARDID'], headers, pause, sectionOfJson, url + f'/securities.json')
        securities_marketdata_df_additional_1 = pandas.concat([securities_marketdata_df_additional_1, securities_marketdata_df_additional_2],
                                                              ignore_index=True)

        if securities_marketdata_df.empty:
            print("securities_marketdata_df пуст")
            securities_marketdata_df = securities_marketdata_df_additional_1.copy()

        else:
            securities_marketdata_df =\
                securities_marketdata_df.merge(securities_marketdata_df_additional_1, how='left', on='SECID', suffixes=('', '_drop'))

            securities_marketdata_df =\
                securities_marketdata_df[[column for column in securities_marketdata_df.columns if not column.endswith('_drop')]]

        # display('securities_marketdata_df 1:', securities_marketdata_df) # для отладки

    # </Формирование файла с доступными securities в интересующих режимах торгов>

    columnsDescriptionS = columnsDescriptionS.drop_duplicates(['id', 'name'], ignore_index=True)
    # display('columnsDescriptionS:', columnsDescriptionS) # для отладки

    print("boardS['boardid']:", boardS['boardid'])
    securities_marketdata_df = securities_marketdata_df[securities_marketdata_df['BOARDID'].isin(boardS['boardid'])]
        # учёт желаемых режимов торгов (аргумент plusNotTraded )

    columnS_withDateTime = ['BUYBACKDATE',
                            'CALLOPTIONDATE',
                            'DATEYIELDFROMISSUER',
                            'IMTIME',
                            'ISSUECAPITALIZATION_UPDATETIME',
                            'LASTDELDATE',
                            'LASTTRADEDATE',
                            'MATDATE',
                            'NEXTCOUPON',
                            'OFFERDATE',
                            'PREVDATE',
                            'PUTOPTIONDATE',
                            'SETTLEDATE',
                            'SYSTIME',
                            'TIME',
                            'TRADE_SESSION_DATE',
                            'TRADEDATE',
                            'TRADEMOMENT',
                            'UPDATETIME',
                            'YIELDDATE',
                            'ZCYCMOMENT']

    for column_withDateTime in columnS_withDateTime:
        if column_withDateTime in securities_marketdata_df.columns:
            securities_marketdata_df[column_withDateTime] = pandas.to_datetime(securities_marketdata_df[column_withDateTime], errors='coerce')

    # securities_marketdata_df['SYSTIME'] = pandas.to_datetime(securities_marketdata_df['SYSTIME'])

    # if market == 'bonds':
    #     securities_marketdata_df['URL MoEx'] = 'https://www.moex.com/ru/issue.aspx?code=' + securities_marketdata_df['ISIN']
    #     securities_marketdata_df['TRADEMOMENT'] = pandas.to_datetime(securities_marketdata_df['TRADEMOMENT'])
    #     securities_marketdata_df['ZCYCMOMENT'] = pandas.to_datetime(securities_marketdata_df['ZCYCMOMENT'])

    # display('securities_marketdata_df 2:', securities_marketdata_df) # для отладки

    securities_marketdata_df_duplicated =\
    securities_marketdata_df[securities_marketdata_df.duplicated(
        ['ISIN', 'REGNUMBER', 'SECID', 'SECNAME', 'SHORTNAME'] if 'ISIN' in securities_marketdata_df and 'REGNUMBER' in securities_marketdata_df else ['SECID', 'SECNAME', 'SHORTNAME'],
        keep=False)
        ]

    # display('securities_marketdata_df_duplicated:', securities_marketdata_df_duplicated) # для отладки

    if len(securities_marketdata_df_duplicated) > 0:
        print('Работаю со срезом securities_marketdata_df , содержащим дублирующиеся по ISIN строки')
        securities_marketdata_df = securities_marketdata_df_duplicates_processor(columnS_withDateTime,
                                                                                 securities_marketdata_df,
                                                                                 securities_marketdata_df_duplicated)

    columnsDescriptionS.to_excel(path_columnsDescriptions, index=False)
    securities_marketdata_df.to_excel(path_securities_marketdata, index=False)
    if returnDfs: return boardS, columnsDescriptionS, securities_marketdata_df

# Схема API MoEx

# в market == 'bonds' | market == 'shares' : url = 'https://iss.moex.com/iss/engines/stock/markets/' + market
# в market == 'forts' : url = 'https://iss.moex.com/iss/engines/futures/markets/' + market

# url + '.json' -- тут boards и columnsDescriptionS остальных sectionOfJson
# columnS_forComparisom = ['id'] # столбцы, по которым сравниваются df_additional_previous и df_additional

# url + '/boards/{board}/securities.json' -- тут securities и финансовые столбцы..

    # .. в market == 'bonds' в marketdata_yields

    # .. в market == 'forts' | market == 'shares' в marketdata

# и там, и там columnS_forComparisom = ['SECID', 'BOARDID'] # столбцы, по которым сравниваются df_additional_previous и df_additional

# board = 'TQCB'
# board = 'RFUD'
# board = 'TQBR'

# columnS_forComparisom = ['id']
# columnS_forComparisom = ['SECID', 'BOARDID']

# market = 'bonds'
# market = 'forts'
# market = 'shares'

# sectionOfJson = 'boards'
# sectionOfJson = 'marketdata_yields'
# sectionOfJson = 'marketdata'
# sectionOfJson = 'securities'

# url = 'https://iss.moex.com/iss/engines/futures/markets/' + market
# url = 'https://iss.moex.com/iss/engines/stock/markets/' + market
# url += '.json'
# url += f'/boards/{board}/securities.json'

# https://iss.moex.com/iss/reference/
# https://iss.moex.com/iss/engines/stock/markets/qnv
# https://iss.moex.com/iss/engines/stock/markets/bonds/boards/tqcb/securities
