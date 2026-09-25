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
        # !pip install grpcio-tools requests pandas
        from grpc_tools import protoc

        from google.protobuf import descriptor_pb2 # для чтения binary descriptor
        from io import StringIO
        from IPython.display import display

        from randan.tools import cellsLeftMerger, coLabAdaptor, scrapingTools #  модули для
            # (а) упрощения операции левостороннего присоединения датафрейма-донора к датафрейму-реципиенту по специальному столбцу
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) упрощения скрапинга

        from tqdm import tqdm
        import grpc_tools, os, pandas, requests, subprocess, time, traceback
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

# 1. Вспомогательная функция для..
# .. выгрузки таблиц -- фрагментов данных формата JSON из БД СПб Биржи
def get_json_df(body, headers, pause, url, max_retries=3):
    for attempt in range(max_retries): # цикл попыток
        try:
            response = requests.post(url, headers=headers, json=body, timeout=30, verify=False)
            response.raise_for_status()  # проверка на ошибки HTTP
            
            data = response.json().get('instruments', [])
            df = pandas.DataFrame(data=data)
            return df

        except requests.exceptions.HTTPError as excptn:
            status_code = excptn.response.status_code
            
            # Если ошибка 500, 502, 503, 504 или 429 (лимит запросов) -- попробовать снова
            if status_code in [429, 500, 502, 503, 504]:
                # Ждем: твоя пауза + экспоненциальная задержка (1с, 2с, 4с)
                wait_time = pause + (2 ** attempt) 
                print(f'Ошибка {status_code}. Жду {wait_time} сек и пробую снова (попытка {attempt + 1}/{max_retries})...')
                time.sleep(wait_time)

            else: # если ошибка другая (например 400 Bad Request) -- нет смысла пробовать снова
                print('Exception в get_json_df (фатальная ошибка HTTP)')
                print(f'{type(excptn).__name__}: {str(excptn).split("Stacktrace:")[0].strip()}')
                return pandas.DataFrame() # заглушка, чтобы не сломать concat в вызывающем цикле
                
        except Exception as excptn:
            # Любая другая ошибка (например, оборвалось соединение)
            print('Exception в get_json_df (сетевая ошибка)')
            print(traceback.format_exc().split('Stacktrace:')[0].strip())
            wait_time = pause + (2 ** attempt)
            time.sleep(wait_time)

    # Если все попытки исчерпаны
    print(f'Не удалось получить данные после {max_retries} попыток. Пропускаю батч.')
    return pandas.DataFrame() # заглушка, чтобы не сломать concat в вызывающем цикле

# .. чтения и парсинга (десериализации) файла схемы .proto
def proto2df(section):

# 1. Клонирование репозитория
    REPO_DIR = 'investAPI' # REPO_DIR -- директория для клона репозитория
    if not os.path.isdir(REPO_DIR): # создать клон репозитория
        subprocess.run(
            ['git', 'clone', '--depth', '1', 'https://github.com/RussianInvestments/investAPI.git', REPO_DIR],
            check=True,
        )

    try: PROTO_DIR = os.path.abspath(os.path.join(REPO_DIR, 'src', 'docs', 'contracts')) # абсолютный путь к папке контрактов внутри репозитория
    except Exception as excptn:
        print('Exception в proto2df') # для отладки
        print(f'{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}') # для отладки
        print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой    
        PROTO_DIR = os.path.abspath(os.path.join(REPO_DIR, 'src', 'proto')) # абсолютный путь к папке контрактов внутри репозитория

    GRPC_INCLUDE = os.path.abspath(os.path.join(os.path.dirname(grpc_tools.__file__), '_proto'))
        # получить путь к встроенным протобуфам grpcio-tools (чтобы он находил google/protobuf/*.proto ) 

# 2. Компиляция текстового файла схемы .proto (например, marketdata.proto) в бинарный файл дескриптора OUT_PB
    OUT_PB = f'{section}.pb' # имя дескриптора

    args = ['protoc', # имя программы (формальность)
            f'-I{PROTO_DIR}', # ищет локальные импорты Т-Банка (включая их папку google/)
            f'-I{GRPC_INCLUDE}', # ищет стандартные google/protobuf/*.proto
            '--include_source_info', # сохраняет комментарии в бинарный дескриптор
            f'--descriptor_set_out={OUT_PB}', # куда положить результат (.pb)
            os.path.join(PROTO_DIR, f'{section}.proto')] # ЧТО компилировать

    # <Перехват stderr, чтобы увидеть ошибку, если protoc упадёт>
    old_stderr = sys.stderr
    sys.stderr = stderr_capture = StringIO()
    return_code = protoc.main(args) # компиляция
    error_message = stderr_capture.getvalue()
    sys.stderr = old_stderr # вернуть stderr на место

    if return_code != 0:
        print('❌ ОШИБКА PROTOC:\n', error_message)
        raise RuntimeError(f'protoc failed with code {return_code}')

    else:
        print('✅ Компиляция успешна!', os.path.getsize(OUT_PB), 'bytes')    
    # </Перехват stderr, чтобы увидеть ошибку, если protoc упадёт>

# 3. Чтение и парсинг бинарного дескриптора OUT_PB и запись в контейнер fdS
    # Распаковка бинарного дескриптора (схемы API) в объекты Python
    fdS = descriptor_pb2.FileDescriptorSet() # инициализация контейнера
    with open(OUT_PB, "rb") as file: fdS.ParseFromString(file.read()) # чтение и парсинг (десериализация)

# 4 Распаковка дерева в таблицу
    data = []

    # Маппинг числовых типов протобуфа в строки
    TYPE_MAP = {1: 'double', 2: 'float', 3: 'int64', 4: 'uint64', 5: 'int32',
                6: 'fixed64', 7: 'fixed32', 8: 'bool', 9: 'string', 10: 'group',
                11: 'message', 12: 'bytes', 13: 'uint32', 14: 'enum',
                15: 'sfixed32', 16: 'sfixed64', 17: 'sint32', 18: 'sint64'}

    # Функция для сборки словаря комментариев из source_code_info дескриптора
    # Protobuf хранит комментарии по числовым путям (path), например [4, 0, 2, 1]
    def build_comments_map(file_desc):
        comments = {}
        for location in file_desc.source_code_info.location:
            partS = []
            if location.leading_comments: partS.append(location.leading_comments.strip())
            if location.trailing_comments: partS.append(location.trailing_comments.strip())
            comment_text = ' '.join(partS)
            if comment_text: comments[tuple(location.path)] = comment_text
        return comments

    # Рекурсивная функция для извлечения enum (перечислений)
    def extract_enum(comments_map, enum_desc, parent_path, path_prefix):
        for val_index, val_desc in enumerate(enum_desc.value):
            # Путь к значению enum: [..., 2(val_index)]
            path = tuple(path_prefix + [2, val_index])
            comment = comments_map.get(path, "")
            
            data.append({'Message_Path': parent_path,
                         'Message': enum_desc.name,
                         'Field_ID': val_desc.number,
                         'Field_Name': val_desc.name,
                         'Field_Type': 'EnumValue',
                         'Comment': comment})

    # Рекурсивная функция для прохода по всем message (сообщениям) (включая вложенные)
    def extract_message(comments_map, msg_desc, parent_path, path_prefix):
        current_path = f'{parent_path}.{msg_desc.name}' if parent_path else msg_desc.name
    
        for field_index, field in enumerate(msg_desc.field):
            # Путь к полю: [..., 2(field_index)]
            path = tuple(path_prefix + [2, field_index])
            comment = comments_map.get(path, '')

            # Определить тип поля
            if field.type_name: # для message и enum (типы 11 и 14) имя хранится тут
                field_type = field.type_name

            else: field_type = TYPE_MAP.get(field.type, 'unknown')

            # Если массив (repeated)
            if field.label == descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED:
                field_type = f'repeated {field_type}'

            data.append({'Message_Path': current_path,
                         'Message': msg_desc.name,
                         'Field_ID': field.number,
                         'Field_Name': field.name,
                         'Field_Type': field_type,
                         'Comment': comment})
    
        # Рекурсивный проход по вложенным message (внутри Message путь 3 - nested_type)
        for nested_index, nested_desc in enumerate(msg_desc.nested_type):
            extract_message(comments_map, nested_desc, current_path, path_prefix + [3, nested_index])
         
        # Рекурсивный проход по вложенным enum (внутри Message путь 4 - enum_type)
        for enum_index, enum_desc in enumerate(msg_desc.enum_type):
            extract_enum(comments_map, enum_desc, current_path, path_prefix + [4, enum_index])

    # Проход по всем файлам в схеме API (например, marketdata.proto и его зависимости)
    for file_desc in fdS.file:
        comments_map = build_comments_map(file_desc)
    
        # Top-level сообщения (путь 4 -- message_type)
        for msg_index, msg_desc in enumerate(file_desc.message_type):
            extract_message(comments_map, msg_desc, '', [4, msg_index])
    
        # Top-level enum (путь 5 -- enum_type)
        for enum_index, enum_desc in enumerate(file_desc.enum_type):
            extract_enum(comments_map, enum_desc, enum_desc.name, [5, enum_index])

    df = pandas.DataFrame(data)
    return df

# .. парсинга ячеек со словарём с целой частью числа (units) и дробной его частью (nano)
def singleJsonParcer(series, row):
    df = pandas.json_normalize(series[row])
    # display('df:', df) # для отладки
    df['nano'] = df['nano'].abs()
    df['units'] = df['units'].astype(str) + '.' + df['nano'].astype(str)
    df['units'] = df['units'].astype(float)
    df = df.drop('nano', axis=1)

    df = df.rename(columns={'units': row}) # поменять имя столбца на значение securitieS_row
    df = df.T
    df = df.rename(columns={0: series.name}) # поменять имя столбца на значение securitieS_row
    return df

# .. парсинга ячеек столбца values датафрейма marketdata_df
def valuesParcer(marketdata_df, marketdata_df_row):
    df = pandas.json_normalize(marketdata_df['values'][marketdata_df_row])
    # display('df:', df) # для отладки
    df['value.nano'] = df['value.nano'].abs()
    df['value.units'] = df['value.units'].astype(str) + '.' + df['value.nano'].astype(str)
    df['value.units'] = df['value.units'].astype(float)
    df = df.drop('value.nano', axis=1)

    df['time'] = pandas.to_datetime(df['time'], format="mixed", utc=True)
    date_time_mean = df['time'].mean()
    # print('Усреднённые даты и время', date_time_mean) # для отладки
    df = df.drop('time', axis=1)

    df = df.set_index('type').rename_axis(None) # сделать столбец type индексом без собственного заголовка
    df = df.rename(columns={'value.units': marketdata_df_row}) # поменять имя столбца value.units на значение marketdata_df_row
    df = df.T
    df.loc[:, 'Усреднённые даты и время'] = date_time_mean
    df['Усреднённые даты и время'] = pandas.to_datetime(df['Усреднённые даты и время'], utc=True)
    return df

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
'{path_securities_marketdata}' -- доступные инструменты (securities) и их финансовые данные (marketdata)
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
            print(f'Нашёл файл tToken.txt; далее буду использовать Т-токен {tToken} из него')

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

    # 2.1.1 Формирование файла с доступными инструментами (securities) и их финансовыми данными (marketdata)
    # <Формирование файла с доступными securities в интересующих режимах торгов>
    print('Создаю файл с доступными инструментами (securities) и их финансовыми данными (marketdata)')

    print('Для этого сначала выружаю доступные инструменты (securities)')

    body = {'instrumentStatus': 'INSTRUMENT_STATUS_ALL'} if plusNotTraded else {'instrumentStatus': 'INSTRUMENT_STATUS_BASE'}
        # около 1500-2000 торгуемых облигаций

    securitieS = get_json_df(body,
                             headers,
                             pause,
                             'https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/Bonds')

    valueS = ['INSTRUMENT_VALUE_UNSPECIFIED', # неопределенное значение (обычно не используется)
              'INSTRUMENT_VALUE_LAST_PRICE', # последняя цена
              'INSTRUMENT_VALUE_LAST_PRICE_DEALER', # цена последней сделки дилера
              'INSTRUMENT_VALUE_CLOSE_PRICE', # цена закрытия
              'INSTRUMENT_VALUE_EVENING_SESSION_PRICE', # цена вечерней сессии
              'INSTRUMENT_VALUE_OPEN_INTEREST', # открытый интерес
              'INSTRUMENT_VALUE_THEOR_PRICE', # теоретическая цена
              'INSTRUMENT_VALUE_YIELD'] # YTM

    print('Теперь выружаю их финансовые данные (marketdata)')
    marketdata = []
    for batch_lenth in tqdm(range(0, len(securitieS), 1500)):
        body = {'instrumentId': list(securitieS['figi'][batch_lenth: batch_lenth + 1500]), 'values': valueS}

        marketdata_df_additional = get_json_df(
            body,
            headers,
            pause,
            'https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetMarketValues'
            )
    
        if not marketdata_df_additional.empty: marketdata.append(marketdata_df_additional)
        time.sleep(pause)

    marketdata_df = pandas.concat(marketdata, ignore_index=True) if marketdata else pandas.DataFrame()

    # display('marketdata_df 1:', marketdata_df) # для отладки

    print('Распарсиваю столбец values в marketdata_df')
    marketdata_df_withValues = marketdata_df[marketdata_df['values'].apply(lambda cellContent: cellContent != [])]
    # display('marketdata_df_withValues:', marketdata_df_withValues) # для отладки

    marketdata_values = []
    for marketdata_df_row in tqdm(marketdata_df_withValues.index):
        if marketdata_df_withValues['values'][marketdata_df_row]:
            marketdata_values_df_additional = valuesParcer(marketdata_df_withValues, marketdata_df_row)

        else: marketdata_values_df_additional = pandas.DataFrame(index=[marketdata_df_row])

        marketdata_values.append(marketdata_values_df_additional)

    marketdata_values_df = pandas.concat(marketdata_values)
    # display('marketdata_values_df:', marketdata_values_df) # для отладки

    marketdata_df = pandas.concat([marketdata_df, marketdata_values_df], axis=1)
    for marketdata_df_column in marketdata_df.columns: # убрать tz у всех datetime-столбцов с timezone
        if isinstance(marketdata_df[marketdata_df_column].dtype, pandas.DatetimeTZDtype):
            marketdata_df[marketdata_df_column] = marketdata_df[marketdata_df_column].dt.tz_convert(None)

    # display('marketdata_df 2:', marketdata_df) # для отладки

    # if returnDfs: return marketdata_df, securitieS

    securities_marketdata_df = cellsLeftMerger.cellsLeftMerger(marketdata_df,
                                                               securitieS,
                                                               ['classCode', 'ticker']) # следует мёрджить по classCode и ticker

    if returnDfs: return securities_marketdata_df
