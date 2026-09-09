# coding: utf-8

'''
A module to import and process bonds' feachures from the SPb Exchange
Модуль для выгрузки характеристик торгуемых на СПб Бирже облигаций
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

        from randan.tools import cellsLeftMerger, coLabAdaptor, scrapingTools #  модули для
            # (а) упрощения операции левостороннего присоединения датафрейма-донора к датафрейму-реципиенту по специальному столбцу
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) упрощения скрапинга

        from tqdm import tqdm
        import os, pandas, requests, time, traceback
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

# 1. Вспомогательная функция..
# .. выгрузки таблиц -- фрагментов данных формата JSON из БД СПб Биржи
def get_json_df(body, headers, pause, url):
    try:
        response = requests.post(url, headers=headers, json=body, verify=False)
        response.raise_for_status()  # проверка на ошибки HTTP
        data = response.json().get('instruments', [])
        df = pandas.DataFrame(data=data)
        # display('data:', data) # для отладки
        return df

    except Exception as excptn:
        print('Exception в get_json_df') # для отладки
        print(f'{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}') # для отладки
        print(traceback.format_exc()) # показ точной строчки кода с ошибкой

# 2. Основная функция
def getSPbExData(folder=coLabFolder,
                 market='bonds',
                 pause=0.1,
                 plusNotTraded=False,
                 returnDfs=False,
                 tToken=None):
    '''
    Функция умеет выгружать характеристики торгуемых на СПб Бирже облигаций.

    Parameters
    ----------
       folder : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

       market : str -- если интересуют облигации, подходит значение по умолчанию 'bonds' , если фьючерсы, впишите 'forts' , если акции, впишите 'shares'
        pause : float -- длительность приостановки исполнения скрипта в секундах
plusNotTraded : bool -- в случае True функция возвращает и неторгуемые securities
    returnDfs : bool -- в случае True функция возвращает итоговые датафреймы boardS, columnsDescriptionS и securities_marketdata_df строго в такой последовательности
    '''

    # Блок, поскольку folder многократно используется внутри функции в формулах
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    # if folder: print('folder до:', folder) # для отладки
    if not folder: folder = ''
    else: folder += slash
    # if folder: print('folder после:', folder) # для отладки

# 2.0 Проверка наличия файла Securities and Marketdata SPbEx.xlsx и вопрос про необходимость его обновления
    path_securities_marketdata = folder + market + ' Securities and Marketdata SPbEx.xlsx'
    if os.path.exists(path_securities_marketdata):

        print(
f'''--- Файл:
'{path_securities_marketdata}' -- доступные инструменты (securities) и их финансовые данные (marketdata или marketdata_yields)
существует; если хотите обновить этот комплект, просто нажмите Enter (это недолго)
--- Если хотите НЕ обновить, то нажмите пробел и затем Enter'''
              )

        decision = input()
        if decision:
            print('Использую существующий комплект')
            if returnDfs:
                securities_marketdata_df = pandas.read_excel(path_securities_marketdata)
                return securities_marketdata_df

# 2.1 Если нет комплекта
# 2.1.0 Поиск Т-токена
    if not tToken:
        rootNameS = os.listdir(folder if folder else None)
        if 'tToken.txt' in rootNameS:
            tToken = scrapingTools.containerImport(folder + 'tToken.txt', str)
            print('Проверяю наличие файла tToken.txt с Т-токеном, гипотетически сохранёнными при первом запуске скрипта')
            print(f'Нашёл файл tToken.txt; далее буду использовать Т-токен {tToken} из него:')

        else:
            print(
'''--- НЕ нашёл файл tToken.txt . Введите в окно Ваш Т-токен для https://developer.tbank.ru/invest/api . После ввода нажмите Enter'''
                  )

            while True:
                tToken = input('Введите в окно Ваш Т-токен и нажмите Enter')
                if len(tToken) > 0:
                    print('-- далее будет использован этот Т-токен')
                    break

                else:
                    print('--- Вы ничего НЕ ввели. Попробуйте ещё раз..')

            scrapingTools.containerExport('tToken.txt', tToken)

    headers = {'Authorization': f'Bearer {tToken}', 'Content-Type': 'application/json'}

# 2.1.1 Формирование файла с доступными инструментами (securities) и их финансовыми данными (marketdata или marketdata_yields)
    # <Формирование файла с доступными securities в интересующих режимах торгов>
    print('Создаю файл с доступными инструментами (securities) и их финансовыми данными (marketdata или marketdata_yields)')
    body = {}
    securitieS = get_json_df(body, headers, pause, 'https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/Bonds')

    valueS = ['INSTRUMENT_VALUE_UNSPECIFIED', # неопределенное значение (обычно не используется)
              'INSTRUMENT_VALUE_LAST_PRICE', # последняя цена
              'INSTRUMENT_VALUE_LAST_PRICE_DEALER', # цена последней сделки дилера
              'INSTRUMENT_VALUE_CLOSE_PRICE', # цена закрытия
              'INSTRUMENT_VALUE_EVENING_SESSION_PRICE', # цена вечерней сессии
              'INSTRUMENT_VALUE_OPEN_INTEREST', # открытый интерес
              'INSTRUMENT_VALUE_THEOR_PRICE', # теоретическая цена
              'INSTRUMENT_VALUE_YIELD'] # YTM

    marketdata_df = pandas.DataFrame()
    for isin in tqdm(securitieS['isin']):
        body = {'instrumentId': figi, values: valueS}
        marketdata_df_additional = get_json_df(body, headers, pause, 'https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetMarketValues')
        marketdata_df = pandas.concat([marketdata_df, marketdata_df_additional])

    # display('marketdata_df 1:', marketdata_df) # для отладки

    securities_marketdata_df = cellsLeftMerger.cellsLeftMerger(marketdata_df,
                                                               securitieS,
                                                               'isin') # следует мёрджить по ISIN

    if returnDfs: return securities_marketdata_df
