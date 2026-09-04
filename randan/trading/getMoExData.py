# coding: utf-8

'''
A module to import and process bonds' feachures from the Moscow Exchange
Модуль для выгрузки характеристик торгуемых на МосБирже облигаций
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
        # from tqdm import tqdm
        from randan.tools import coLabAdaptor # авторский модуль для адаптации текущего скрипта к файловой системе CoLab
        import os, pandas, requests, traceback, warnings # , re
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

coLabFolder = coLabAdaptor.coLabAdaptor()

# 1. Авторские функции..
    # .. выгрузки имён полей БД МосБиржи
# def getColumnNameS(text):
#     columnS = BeautifulSoup(text, features='xml').find_all('column')
#     # print('columnS:', columnS) # для отладки
#     columnNameS = []
#     for column in columnS:
#         columnNameS.append(column.get('name'))
#     return columnNameS

    # .. выгрузки таблиц -- фрагментов данных формата JSON из БД МосБиржи
def json2df(columnS_forComparisom, headers, sectionOfJson, url):
    df = pandas.DataFrame()
    df_additional_previous = pandas.DataFrame()
    start = 0
    while True:
        # print('start:', start, '                    ', end='\r') # для отладки
        params = {'start': start} # 'limit': 100,

        data_json = requests.get(url, headers=headers, params=params).json()
        df_additional = pandas.DataFrame(columns=data_json[sectionOfJson]['columns'], data=data_json[sectionOfJson]['data'])
        # df_additional = df_additional.fillna('Нет данных')
        # display('df_additional:', df_additional) # для отладки

        try:
            if (len(df_additional) == 0) | ((df_additional[columnS_forComparisom] != df_additional_previous).sum().sum() == 0):
                    # во второй части условия проверяется наличие различия между датафреймами хотя бы в одной ячейке

                # print('Похоже, df_additional == df_additional_previous; завершаю цикл') # для отладки
                break

        except Exception as excptn:
            # print('Exception') # для отладки
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
# def pseudojson2df(headerS, index, url):
#     df = pandas.DataFrame()
#     text = re.findall(r'<data.+?/data>', requests.get(url, headers=headerS).text, re.DOTALL)[index]
#     # print('text:', text) # для отладки
#     columnNameS = getColumnNameS(text)
#     rowS = BeautifulSoup(text, features='xml').find_all('row')
#     # print('rowS:', rowS) # для отладки
#     i = 0
#     for row in tqdm(rowS):
#         i += 1
#         for column in columnNameS:
#             df.loc[i, column] = row.get(column)
#     # display('df:', df)
#     return df

# 2. Основная функция
def getMoExData(folder=coLabFolder,
                market='bonds',
                plusNotTraded=False,
                returnDfs=False):
    '''
    Функция умеет выгружать характеристики торгуемых на МосБирже облигаций, причём не дефолтные (далее -- Д) и не повышенного инвестиционного риска (далее -- ПИР). Дополнительно выгружается словарь полей БД МосБиржи. Также она умеет выгружать фьючерсы

    Parameters
    ----------
       folder : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

       market : str -- если интересуют облигации, подходит значение по умолчанию 'bonds' , если фьючерсы, впишите 'forts' , если акции, впишите 'shares'
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

# Формирование файла с режимами торгов
# 2.0 Если нет файла с режимами торгов
    print('Создаю файл с режимами торгов')
    if (market == 'bonds') | (market == 'shares'): url = f'https://iss.moex.com/iss/engines/stock/markets/{market}'
    if market == 'forts': url = f'https://iss.moex.com/iss/engines/futures/markets/{market}'
    boardS = json2df(['id'], headers, 'boards', url + '.json')
    # boardS = pseudojson2df(headers, 0, url)
    # display('boardS:', boardS) # для отладки

    # display('boardS 1:', boardS) # для отладки
    # if market == 'bonds': boardS = boardS[boardS['title'].str.contains('облигации ', case=False)] # если облигации: нужны именно облигации
    # display('boardS 2:', boardS) # для отладки
    if not plusNotTraded: boardS = boardS[boardS['is_traded'].astype(int) == 1]
    # display('boardS 3:', boardS) # для отладки

# 2.1 Формирование файла с доступными securities
    decision = ''
    goC = True
    securities_marketdata_df = pandas.DataFrame()
    path_1 = folder + market + ' Securities and Marketdata.xlsx'
    if os.path.exists(path_1):
        print(
f'''--- Файл с доступными securities и финансовой информацией '{path_1}' существует; если НЕ хотите обновить этот файл, просто нажмите Enter
--- Если хотите, то нажмите пробел и затем Enter'''
              )

        decision = input()

        if decision: print('Создаю новый файл', path_1)
        else:
            print('Использую существующий файл', path_1)
            securities_marketdata_df = pandas.read_excel(path_1)
            goC = False

# 2.1.0 Формирование словаря полей БД МосБиржи и файла с доступными securities в интересующих режимах торгов
    print('Создаю файл со словарём полей БД МосБиржи')
    columnsDescriptionS = pandas.DataFrame()
    sectionOfJson_list = ['securities']
    if market == 'bonds': sectionOfJson_list.append('marketdata_yields')
    if (market == 'forts') | (market == 'shares'): sectionOfJson_list.append('marketdata')
    # if market == 'bonds': indeceS = [2, 8]
    # if market == 'forts': indeceS = [2, 3]
    # for index in indeceS:
    for sectionOfJson in sectionOfJson_list:
        # print('sectionOfJson:', sectionOfJson) # для отладки
        # <Формирование словаря полей БД МосБиржи>
        columnsDescriptionS_additional = json2df(['id'], headers, sectionOfJson, url + '.json')
        # columnsDescriptionS_additional = pseudojson2df(headerS, index, url)
        columnsDescriptionS_additional.loc[:, 'data id'] = sectionOfJson
        # columnsDescriptionS_additional.loc[:, 'data id'] = index
        columnsDescriptionS = pandas.concat([columnsDescriptionS, columnsDescriptionS_additional], ignore_index=True)
        # </Формирование словаря полей БД МосБиржи>

        # <Формирование файла с доступными securities в интересующих режимах торгов>
        if goC:
            securities_marketdata_df_additional_1 = pandas.DataFrame()
            for board in boardS['boardid']:
                print('board:', board)
                securities_marketdata_df_additional_2 = json2df(['SECID', 'BOARDID'], headers, sectionOfJson, url + f'/boards/{board}/securities.json')
                securities_marketdata_df_additional_2['board'] = board
                securities_marketdata_df_additional_1 = pandas.concat([securities_marketdata_df_additional_1, securities_marketdata_df_additional_2], ignore_index=True)

            if len(securities_marketdata_df) > 0:
                securities_marketdata_df = securities_marketdata_df.merge(securities_marketdata_df_additional_1, how='left', on='SECID', suffixes=('', '_drop'))
                securities_marketdata_df = securities_marketdata_df[[column for column in securities_marketdata_df.columns if not column.endswith('_drop')]]
                # print('securities_marketdata_df.columns:', securities_marketdata_df.columns) # для отладки

            else: securities_marketdata_df = securities_marketdata_df_additional_1.copy()
        # </Формирование файла с доступными securities в интересующих режимах торгов>

    columnsDescriptionS = columnsDescriptionS.drop_duplicates(['id', 'name'], ignore_index=True)
    # display('columnsDescriptionS:', columnsDescriptionS) # для отладки

    path_2 = market + ' Columns descriptions.xlsx'
    columnsDescriptionS.to_excel(path_2, index=False)

    if os.path.exists(path_2.replace('.xlsx', ' selected.xlsx')):
        columnsDescriptionS = pandas.read_excel(path_2.replace('.xlsx', ' selected.xlsx'))

    # display('columnsDescriptionS:', columnsDescriptionS) # для отладки
    columnsDescriptionS = columnsDescriptionS[columnsDescriptionS['name'] !='BOARDID']
    columnsDescriptionS = columnsDescriptionS[columnsDescriptionS['name'].notna()]
    columnsDescriptionS = columnsDescriptionS['name'].drop_duplicates().tolist()
    if market == 'bonds': columnsDescriptionS.append('URL MoEx')
    # securities_marketdata_df = securities_marketdata_df.groupby('SECID', as_index=False).first()
    # print('securities_marketdata_df.columns:', securities_marketdata_df.columns) # для отладки
    if market == 'bonds': securities_marketdata_df['URL MoEx'] = 'https://www.moex.com/ru/issue.aspx?code=' + securities_marketdata_df['ISIN']
    # print('securities_marketdata_df.columns:', securities_marketdata_df.columns) # для отладки
    securities_marketdata_df = securities_marketdata_df[columnsDescriptionS]
    securities_marketdata_df.to_excel(path_1, index=False)
    # display(securities_marketdata_df) # для отладки

    # securities_marketdata_df =\
    #     securities_marketdata_df.drop_duplicates(['ISIN', 'REGNUMBER', 'SECID', 'SECNAME', 'SHORTNAME'], ignore_index=True)
    #         # костыль

    if returnDfs: return boardS, columnsDescriptionS, securities_marketdata_df
    warnings.filterwarnings('ignore')
    print("Скрипт исполнен. Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть")
    input()
    sys.exit()
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
