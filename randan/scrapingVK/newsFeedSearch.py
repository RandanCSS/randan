#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and maximizes VK content extraction using the platform's official API method newsfeed.search
(RU) Модуль для упрощения выгрузки контента ВК методом его API newsfeed.search и максимизации размера этой выгрузки
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
        from datetime import datetime
        from IPython.display import display
        from randan.scrapingVK import scrapingVK_tools # модуль для упрощения скрапинга VK

        from randan.tools import calendarWithinYear, coLabAdaptor, df2file, os_wrapper, scrapingTools # модули для
            # (а) работы с календарём конкретного года
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (г) упрощения работы с OS
            # (д) упрощения скрапинга

        import json, os, pandas, re, shutil, time, traceback, requests, warnings
        break # выход из цикла while True

    except (ImportError, ModuleNotFoundError) as excptn_1:
        module = excptn_1.name.split('.')[0]
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

# 1. Вспомогательные функции..
# .. обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessor(complicatedNamePart,
                 countTarget,
                 df_additional,
                 df_in,
                 fields,
                 fileFormatChoice,
                 folder,
                 goS, # единственная из функций, принимающая этот аргумент
                 method,
                 momentCurrent,
                 params_extra,
                 q,
                 silentMode,
                 slash,
                 stage,
                 year,
                 yearsRange):

    df = pandas.concat([df_in, df_additional]) # порядок аргументов гарантирует, что в дальнейшем при keep='last' выживают именно новые записи

    columnsForCheck = []
    for column in df.columns: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        if column == 'id' or column.endswith('_id') or column.startswith('id_'): columnsForCheck.append(column)
    # print('Столбцы, по которым проверяю дублирующиеся записи:', columnsForCheck) # для отладки

    if columnsForCheck: df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True)
        # при дублировании записей из itemS из Temporal и от пользователя и новых записей, оставить новые

    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС

# Сохранение следа исполнения скрипта, прерванного пользователем или натолкнувшегося на ошибку
    if not goS:
        temporal_exportPath = folder + f"{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"
        if not silentMode: print(f"Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию '{temporal_exportPath}'")
        # print('temporal_exportPath:', temporal_exportPath) # для отладки
        if not os.path.exists(temporal_exportPath):
            os.makedirs(temporal_exportPath)
            print(f"Директория '{temporal_exportPath}' создана")
        # else: print(f"Директория '{temporal_exportPath}' существует")

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
        if not fields: fields = []
        with open(temporal_exportPath + slash + 'fields.json', 'w', encoding='utf-8') as file:
            json.dump(fields, file, ensure_ascii=False, indent=4)

        scrapingTools.containerExport(temporal_exportPath + slash + 'method.txt', method)

        with open(temporal_exportPath + slash + 'params_extra.json', 'w', encoding='utf-8') as file:
            json.dump(params_extra, file, ensure_ascii=False, indent=4)

        scrapingTools.containerExport(temporal_exportPath + slash + 'q.txt', q if q else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'stageTarget.txt', str(stage)) # stage и stageTarget принимает значения [0; 1]
        scrapingTools.containerExport(temporal_exportPath + slash + 'countTarget.txt', str(countTarget) if countTarget else '0')
        scrapingTools.containerExport(temporal_exportPath + slash + 'year.txt', str(year)) # год, на котором остановилось исполнение скрипта
        scrapingTools.containerExport(temporal_exportPath + slash + 'yearsRange.txt', str(yearsRange))

        df2file.df2fileShell(complicatedNamePart=f'{complicatedNamePart}_Temporal',
                             df_in=df,
                             fileFormatChoice=fileFormatChoice,
                             folder=temporal_exportPath,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла  
                             momentCurrent=momentCurrent)

        if not silentMode:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473''')

                sys.exit()
    return df

# .. метода search из API ВК, помогающая работе с ключами
def newsfeedSearchCore(api_keyS,
                       count,
                       end_time,
                       fields,
                       iteration,
                       keyOrder,
                       latitude,
                       longitude,
                       params_extra,
                       pause,
                       q,
                       silentMode,
                       start_from,
                       start_time,
                       version_api):

    # print('    start_from:', start_from) # для отладки
    df_additional = pandas.DataFrame()
    goS = True

    params_basic = {'access_token': api_keyS[keyOrder], # обязательный параметр
                    'count': count, # опциональный параметр
                    'end_time': end_time, # опциональный параметр
                    'extended': 1, # опциональный параметр
                    'fields': fields, # опциональный параметр
                    'latitude': latitude, # опциональный параметр
                    'longitude': longitude, # опциональный параметр
                    'q': q, # опциональный параметр
                    'start_from': start_from, # опциональный параметр
                    'start_time': start_time, # опциональный параметр
                    'v': version_api} # обязательный параметр

    PARAMS_PROTECTED = {'access_token', 'v'}
    if params_extra:
        for key, value in params_extra.items():
            if key in PARAMS_PROTECTED: continue
            if key not in params_basic or params_basic[key] is None: params_basic[key] = value

    params = {key: value for key, value in params_basic.items() if value not in (None, '', [])}

    for key, value in list(params.items()):
        if isinstance(value, (list, tuple)): # requests сериализует list как повторяющийся ключ, например: fields=a&fields=b , а wall.get ожидает fields=a,b
            params[key] = ','.join(map(str, value)) # склеить список в строку только для запроса; исходный fields остаётся нетронутым для dfColumnsProcessor

    goC = True
    tryer = 0
    while goC:
        response = requests.get('https://api.vk.ru/method/newsfeed.search', params=params).json()
        # print('response', response) # для отладки
        if 'response' in response.keys():
            response = response['response']
            # print('    response.keys() внутри newsfeedSearchCore', response.keys()) # для отладки

            df_additional = pandas.json_normalize(response['items'])

            if not silentMode: print('    Итерация №', iteration, ', number of items', len(response['items']), '                                        ', end='\r')
                # ^ для визуализации процесса

            iteration += 1
            if len(df_additional) > 0: df_additional = scrapingVK_tools.dfColumnsProcessor(df_additional, fields, response)            
            break # нет смысла в новых итерациях цикла while goC

        else:
            goC, goS, keyOrder, pause, response, tryer = scrapingVK_tools.errorProcessor(api_keyS, keyOrder, pause, response, tryer)
            params['access_token'] = api_keyS[keyOrder]

    return df_additional, goS, iteration, keyOrder, pause, response

# 2. Основная функция
def newsFeedSearch(access_token=None,
                   count=None,
                   end_time=None, # !!! можно сделать 2 варианта подачи: год или Unix
                   fields=None,
                   folder=None,
                   latitude=None,
                   longitude=None,
                   params=None,
                   q=None,
                   returnDfs=False,
                   silentMode=False,
                   start_time=None,
                   version_api='5.199'):
    '''
    Функция для выгрузки постов и их характеристик из ВК методом его API newsfeed.search . Причём количество постов максимизируется путём сегментирования выгрузки по годам

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/newsfeed.search , за исключением аргументов folder , params , returnDfs и silentMode
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params , который обычно подаётся в метод get пакета requests
    access_token : str
           count : int
        end_time : int -- формат Unix
          fields : list
          folder : str -- директория, в которой пользователь желает расположить выгружаемые данные и настройки выгрузки; по умолчанию, они расположатся в той же директории, в которой исполняется этот скрипт
        latitude : int
       longitude : int
          params : dict -- в случае наличия готового словаря с аргументами метода https://dev.vk.com/ru/method/newsfeed.search , чтобы не подавать эти аргументы по отдельности
               q : str
       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными
      silentMode : bool -- в случае True в функции отключаются CLI-диалоги; при returnDfs=False и silentMode=False функция завершает процесс через sys.exit()
      start_time : int -- формат Unix
     version_api : str
    '''

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API ВК
    countTarget = 999999 # на случай, если сигнал прерывания поступит до newsfeedSearchCore
    fileFormatChoice = 'xlsx' # базовый формат сохраняемых файлов; формат json добавляется опционально через наличие columnsToJSON
    folder_file = None
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    folder_subfolder_in_folderCurrent = None
    folderWithTemporal = None
    goS = True
    itemS = pandas.DataFrame()
    itemS_additional = pandas.DataFrame() # на случай, если сигнал прерывания поступит до dfsProcessor
    keyOrder = 0
    method = 'newsfeed.search'
    params_extra = {}
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    stageTarget = 0 # stageTarget принимает значения [0; 1] и относится к стадиям скрипта
    yearMinByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearMaxByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearsRange = ''

    momentCurrent = datetime.now() # запрос текущего момента
    print('\nТекущий момент:', momentCurrent.strftime('%Y%m%d_%H%M'), '-- он будет использован для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')

    if silentMode: decision = '' # для веток, в которых silentMode b пользователя " не спрашивают"

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    api_keyS = scrapingTools.credentialsProcessor(access_token, 'VK', folder, 'https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8', os.listdir(folder), silentMode)

# Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, прерванном пользователем или ошибкой
    # Поиск данных
    print('Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
    folderS_subfolderS_in_folderCurrent = os_wrapper.list_by_freshness( # список директорий и субдиректорий внутри folder , удовлетворяющих условиям и упорядоченных по "свежести"
        folder,
        filter_dirEntry=lambda e: (                                   # предикат по DirEntry
            e.name.endswith('Temporal') and 'VK' in e.name and        # дешёвые строковые проверки
            e.is_dir(follow_symlinks=False)                           # тип проверяем последним: это уже сискол (кешируется в DirEntry)
        ))

    for folder_subfolder_in_folderCurrent in folderS_subfolderS_in_folderCurrent:
        if len(os.listdir(folder_subfolder_in_folderCurrent)) < 9:
            if not silentMode: print(
f"Директория '{folder_subfolder_in_folderCurrent}' похожая на искомую Temporal , но в ней не 9-10 файлов => если это она, то либо повреждена, либо создалась при безрезультатном запуске; тогда Вам следует удалить её вручную") # допустимо 9 или 10 файлов

        else:
            countTargetTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'countTarget.txt', int)

            with open(f'{folder_subfolder_in_folderCurrent}{slash}fields.json', 'r', encoding='utf-8') as file:
                fieldsTemporal = json.load(file)

            methodTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'method.txt', str)

            with open(f'{folder_subfolder_in_folderCurrent}{slash}params_extra.json', 'r', encoding='utf-8') as file:
                params_extraTemporal = json.load(file)

            try: qTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'q.txt', str)
            except Exception as excptn:
                print('Exception в searchByText') # для отладки
                print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
                print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой
                qTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'q.txt', str)

            stageTargetTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'stageTarget.txt', int)
            yearTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'year.txt', int)
            yearsRangeTemporal = scrapingTools.containerImport(folder_subfolder_in_folderCurrent + slash + 'yearsRange.txt', str)

            if not silentMode:
                print(f"Нашёл директорию '{folder_subfolder_in_folderCurrent}'. В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:"
                      , '\n- скрипт остановился на методе', methodTemporal)

                if yearTemporal < int(momentCurrent.strftime('%Y')): print('- и на годе (при сегментировани по годам)', yearTemporal)
                print(f"- пользователь сформулировал запрос-фильтр как '{qTemporal}'" if qTemporal else '- пользователь НЕ сформулировал запрос-фильтр')
                print(f'- пользователь ограничил временнОй диапазон границами {yearsRangeTemporal}' if yearsRangeTemporal else '- пользователь НЕ ограничил временнОй диапазон')
                print(f"- пользователь подал дополнительные настройки посредством словаря params: '{params_extraTemporal}'" if params_extraTemporal else '')

                print(
f'''В дальнейшей работе могу опираться на эти промежуточные результаты. При этом случае ПРОТИВОРЕЧИЯ между найденными в директории '{folder_subfolder_in_folderCurrent}' и поданными Вами при текущем запуске аргументами будут применены ПЕРВЫЕ.
--- Если следует опираться на промежуточные результаты, нажмите Enter;  
--- Если эти промежуточные результаты уже не актуальны и хотите, чтобы я удалил их, введите 'R' и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter''')

                try: decision = input()
                except EOFError:
                    print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                    decision = ''
                    silentMode = True

            if not silentMode and decision.lower() == 'r':
                shutil.rmtree(folder_subfolder_in_folderCurrent, ignore_errors=True)

                print(
f"Директорию '{folder_subfolder_in_folderCurrent}' удалил, продолжаю поиск ещё одной директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку")

            elif not silentMode and decision and decision.lower() != 'r': print(
f"Директорию '{folder_subfolder_in_folderCurrent}' пропустил, продолжаю поиск ещё одной директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку")

            elif silentMode or not decision: # при silentMode Temporal-значения результаты принудительно берутся из свежей Temporal-директории и обретают статус постоянных и будут использоваться при исполнении скрипта
                # ^ они обретают статус постоянных и при пустом decision

                countTarget = countTargetTemporal
                fields = fieldsTemporal
                method = methodTemporal
                params_extra = params_extraTemporal
                q = qTemporal
                stageTarget = stageTargetTemporal
                yearsRange = yearsRangeTemporal
                if yearsRange: yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)

                temporalNameS = os.listdir(folder_subfolder_in_folderCurrent)
                folderWithTemporal, goC, itemS = scrapingTools.get_itemsTemporal(temporalNameS)
                if not goC: break # выход из for folder_subfolder_in_folderCurrent in folderS_subfolderS_in_folderCurrent

    if not silentMode and not folderWithTemporal: # если itemsTemporal, в т.ч. пустой, не существует и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        print('Не найдены (потому что их нет или они сознательно пропущены пользователем) подходящие данные, гипотетически сохранённые при прошлом запуске скрипта')

        # Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
        folder, folder_file, itemS = scrapingTools.call_for_user_items(f'в котором есть ранее выгруженные из ВК методом {method} данные')
            # ^ теперь определены объекты: folder и folder_file (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский)

    if not params and not access_token and not count and not end_time and not fields and not latitude and not longitude and not q and not start_time and not returnDfs:
        # print('Пользователь не подал аргументы') # для отладки
        experiencedMode = False

    else:
        experiencedMode = True
        KEYS_BASIC = {'access_token', 'count', 'end_time', 'fields', 'latitude', 'longitude', 'q', 'start_time', 'v'}
        if params:
            access_token = scrapingTools.argument_key_comparison(access_token, 'access_token', params)
            # print('access_token:', access_token) # для отладки

            count = scrapingTools.argument_key_comparison(count, 'count', params)
            # print('count:', count) # для отладки

            end_time = scrapingTools.argument_key_comparison(end_time, 'end_time', params)
            # print('end_time:', end_time) # для отладки

            fields = scrapingTools.argument_key_comparison(fields, 'fields', params)
            # print('fields:', fields) # для отладки

            latitude = scrapingTools.argument_key_comparison(latitude, 'latitude', params)
            # print('latitude:', latitude) # для отладки

            longitude = scrapingTools.argument_key_comparison(longitude, 'longitude', params)
            # print('longitude:', longitude) # для отладки

            q = scrapingTools.argument_key_comparison(q, 'q', params)
            # print('q:', q) # для отладки

            start_time = scrapingTools.argument_key_comparison(start_time, 'start_time', params)
            # if start_time is not None: # эта обработка вынесена наружу ветки if params
            #     if not isinstance(start_time, int): start_time = int(start_time)
            # print('start_time:', start_time) # для отладки

            if not params_extra: # если params_extra НЕ извлечены из Temporal, то попробовать извлечь их из params
                params_extra = {key: value for key, value in params.items() if key not in KEYS_BASIC}
                    # всё, что осталось -- пользовательские аргументы метода
    
                params_extra = {key: value for key, value in params_extra.items() if value is not None}

    if count:
        if not isinstance(count, int): count = int(count)
        count = min(abs(count), 200)
    
    else: count = 200
    # print('count:', count) # для отладки

    if end_time is not None:
        if not isinstance(end_time, int): end_time = int(end_time)
    # print('end_time:', end_time) # для отладки

    if latitude is not None:
        if not isinstance(latitude, int): latitude = int(latitude)
    # print('latitude:', latitude) # для отладки

    if longitude is not None:
        if not isinstance(longitude, int): longitude = int(longitude)
    # print('longitude:', longitude) # для отладки
    
    if q:
        if not isinstance(q, str): q = str(q)
    # print('q:', q) # для отладки

    if start_time is not None:
        if not isinstance(start_time, int): start_time = int(start_time)
    # print('start_time:', start_time) # для отладки

# На выходе аргументы + params_extra . Приоритет определён при первой подаче аргументов и словаря params, этот приоритет транслируется через в Temporal, следовательно, при извлечении из Temporal перепроверять приоритет не требуется

    if not silentMode:
        if not experiencedMode:
            print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.''')

        print(
f'''    Скрипт нацелен на выгрузку постов и их характеристик из ВК методом его API {method} . Причём количество постов максимизируется путём сегментирования выгрузки по годам. При этом следует учесть, что текущая версия API ВК, на которой и основан скрипт, 'не умеет' обращаться к контенту, опубликованному ранее 2020 года.
Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК''')

        if not experiencedMode:
            try: input('--- После прочтения этой инструкции нажмите Enter')
            except EOFError:
                print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                decision = ''
                silentMode = True

# 2.0.2 Пользовательские настройки запроса к API ВК
    if not silentMode: # если основная функция этого модуля явлется частью другой функции, запрос данных не требуется -- они будут переданы
    # <Диалог с пользователем>
        if not q: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать посты открытых аккаунтов по ТЕКСТОВОМУ ЗАПРОСУ-ФИЛЬТРУ
--- Введите текст запроса-фильтра, который ожидаете найти в постах, после чего нажмите Enter''')

            if folder_file: print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folder_file, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых записей и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла')

            try: q = input()
            except EOFError:
                print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                q = ''
                silentMode = True

            if q == '': q = None # для единообразия
            else: print('')

        # Ограничения временнОго диапазона
        if not start_time and not end_time and not yearsRange: # если пользователь не подал эти аргументы в рамках experiencedMode
            print(
f'''Если требуется конкретный временнОй диапазон, то можно использовать его не на этапе выгрузки данных, а на этапе предобработки датафрейма с выгруженными данными. Проблема в том, что без назначения временнОго диапазона метод {method} выдаёт ограниченное количество постов, причём наиболее приближенных к текущему моменту
--- Поэтому если всё же требуется назначить временнОй диапазон на этапе выгрузки данных, назначьте его годами (а не более мелкими единицами времени). Введите без кавычек либо один год (если требуется один год), либо диапазон: минимальный год диапазона, тире, максимальный год диапазона -- и нажмите Enter !!! Cледует учесть, что текущая версия API ВК 'не умеет' обращаться к контенту, опубликованному ранее 2020 года
--- Если НЕ требуется назначить временнОй диапазон на этапе выгрузки данных, нажмите Enter''')

            while True:
                try: yearsRange = input()
                except EOFError:
                    print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                    yearsRange = ''
                    silentMode = True

                if yearsRange:
                    yearsRange = re.sub(r'\s*', '', yearsRange)
                    yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)
                    break
    # </Диалог с пользователем>

    if start_time is not None:
        yearMinByUser = int(datetime.fromtimestamp(start_time).strftime('%Y')) # из experiencedMode
        # print('elif start_time is not None', yearMinByUser) # для отладки

    if end_time is not None:
        yearMaxByUser = int(datetime.fromtimestamp(end_time).strftime('%Y')) # из experiencedMode
        # print('elif end_time is not None:', yearMaxByUser) # для отладки

    if yearMinByUser and not yearMaxByUser: yearMaxByUser = int(momentCurrent.strftime('%Y')) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
    elif not yearMinByUser and yearMaxByUser: yearMinByUser = 1970 # нижняя граница Unix-эпохи в случае отсутствия пользовательской нижней временнОй границы при наличии верхней

    # print('yearMinByUser', yearMinByUser) # для отладки
    # print('yearMaxByUser', yearMaxByUser) # для отладки

    if not start_time and yearMinByUser: start_time = int(datetime(yearMinByUser, 1, 1).timestamp()) # int(time.mktime(datetime(yearMinByUser, 1, 1).timetuple()))
    if not end_time and yearMaxByUser: end_time = int(datetime(yearMaxByUser, 12, 31).timestamp())

    if yearTemporal: year = yearTemporal
    elif yearMaxByUser: year = yearMaxByUser
    else: year = int(momentCurrent.strftime('%Y')) # в случае отсутствия пользовательского временнОго диапазона
        # ^ с этого года возможно сегментирование по годам вглубь веков (пока выдача не пустая)


    if yearsRange: print('') # чтобы был отступ, если пользователь подал этот аргумент

    # Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    if q:
        q_safe = re.sub(r'[\\/:*?"<>|]', '_', q)
        complicatedNamePart += '_' + q_safe if len(q_safe) < 50 else '_' + q_safe[:50]

    complicatedNamePart += '_' + str(yearMinByUser) + '-' + str(yearMaxByUser) if yearMinByUser or yearMaxByUser else ''
    # print('complicatedNamePart', complicatedNamePart)

# 2.1 Первичный сбор контента методом newsFeed.search
    iteration = 1 # номер итерации применения текущего метода
    pause = 0.25
    stage = 0
    if not silentMode:
        print(f'В этой части функции используются следующие аргументы метода {method} API ВК: q, start_from, start_time, end_time, expand и другие аргументы, которые пользователь может подать через params')
        # print('experiencedMode:', experiencedMode) # для отладки
        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter') # обработка случая неинтерактивныой среды уже есть выше
        print('') # для отступа

    try: # обработать сигнал прерывания, поданный на любом этапе сбора данных
# 2.1.0 Первое обращение к API БЕЗ аргументов start_time, end_time (этап stage = 0)
        if stage < stageTarget: # если в Temporal ЕСТЬ файл stage.txt с указанием пропустить этап
            if not silentMode: print(f"Этап {stage} пропускаю согласно настройкам из файла stage.txt из директории '{folderWithTemporal}'\n")

        else:
            if not silentMode: print('Первое обращение к API -- прежде всего, чтобы узнать примерное число доступных релевантных постов')
            # print('    start_from:', start_from) # для отладки

            itemS_additional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                             count,
                                                                                             end_time,
                                                                                             fields,
                                                                                             iteration,
                                                                                             keyOrder,
                                                                                             latitude,
                                                                                             longitude,
                                                                                             params_extra,
                                                                                             pause,
                                                                                             q,
                                                                                             silentMode,
                                                                                             start_from=None,
                                                                                             start_time=start_time,
                                                                                             version_api=version_api)

        # <Обработка результата первого обращения к API>
            # print('goS 1:', goS) # для отладки
            if 'total_count' in response.keys(): # при ошибке в newsfeedSearchCore response является словарём ошибки; тогда эта ветка пропускается и исполняется dfsProcessor ниже, который сохраняет Temporal-снимок и завершает исполнение скрипта
                countTarget = response['total_count']
                if len(itemS_additional) == 0 or countTarget == 0: # ожидаемое или реальное количество постов равно нулю
                    if not silentMode:
                        print(
'  Искомых постов по Вашему запросу, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта. Что можно сделать для выгрузки постов? Поменяйте настройки запроса и запустите скрипт с начала')

                        print('Скрипт исполнен. Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')

                    if folderWithTemporal and os.path.exists(folderWithTemporal):
                        # print('folderWithTemporal:', folderWithTemporal) # для отладки
                        if not silentMode:
                            print(
'''Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, можно удалить её во избежание путаницы при следующих запусках скрипта
--- Если хотите их удалить, введите 'R' и нажмите Enter''')

                            try: decision = input()
                            except EOFError:
                                print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                                decision = ''
                                silentMode = True

                        if silentMode or decision.lower() == 'r': shutil.rmtree(folderWithTemporal, ignore_errors=True) # при silentMode использованная Temporal-директория удаляется

                    if returnDfs or silentMode: return itemS
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            if not silentMode: print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473''')

                            sys.exit()

            itemS = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                 countTarget=countTarget,
                                 df_additional=itemS_additional,
                                 df_in=itemS,
                                 fields=fields,
                                 fileFormatChoice=fileFormatChoice,
                                 folder=folder,
                                 goS=goS,
                                 method=method,
                                 momentCurrent=momentCurrent,
                                 params_extra=params_extra,
                                 q=q,
                                 silentMode=silentMode,
                                 slash=slash,
                                 stage=stage,
                                 year=year,
                                 yearsRange=yearsRange)

            if not goS and silentMode: return itemS # завершение исполнения скрипта вне dfsProcessor после сохранения в нём Temporal-снимка
        # </Обработка результата первого обращения к API>

            if not silentMode: print('  Проход по всем следующим страницам с выдачей          ')
            while 'next_from' in response.keys() and goS:
                start_from = response['next_from']
                # print('    start_from:', start_from) # для отладки
                # print('goS 2:', goS) # для отладки

                itemS_additional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                 count,
                                                                                                 end_time,
                                                                                                 fields,
                                                                                                 iteration,
                                                                                                 keyOrder,
                                                                                                 latitude,
                                                                                                 longitude,
                                                                                                 params_extra,
                                                                                                 pause,
                                                                                                 q,
                                                                                                 silentMode,
                                                                                                 start_from,
                                                                                                 start_time,
                                                                                                 version_api)

                # print("    response['next_from'] после newsfeedSearchCore :", response['next_from']) # для отладки

                itemS = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                     countTarget=countTarget,
                                     df_additional=itemS_additional,
                                     df_in=itemS,
                                     fields=fields,
                                     fileFormatChoice=fileFormatChoice,
                                     folder=folder,
                                     goS=goS,
                                     method=method,
                                     momentCurrent=momentCurrent,
                                     params_extra=params_extra,
                                     q=q,
                                     silentMode=silentMode,
                                     slash=slash,
                                     stage=stage,
                                     year=year,
                                     yearsRange=yearsRange)

                if not goS and silentMode: return itemS # завершение исполнения скрипта вне dfsProcessor после сохранения в нём Temporal-снимка
            print('  Искомых постов', countTarget, ', а выгруженных БЕЗ сегментирования по годам:', len(itemS))

# 2.1.1 Этап сегментирования по годам (stage = 1) # !!! можно сделать динамическое определение периода, который наиболее приближает размер допустимой выдачи (1000) к бенчмарку
        stage = 1
        if stage < stageTarget: # если в Temporal ЕСТЬ файл stage.txt с указанием пропустить этап
            if not silentMode: print(f"Этап {stage} пропускаю согласно настройкам из файла stage.txt из директории '{folderWithTemporal}'\n")

        elif len(itemS) == countTarget: # остановка алгоритма, если все искомые объекты найдены БЕЗ сегментирования по годам
            if not silentMode: print('  Все искомые посты выгружены')
            
        elif len(itemS) < countTarget:
            if not silentMode:
                print(
f'''Метод {method} выдаёт ограниченное количество постов, причём наиболее приближенных к текущему моменту. Скрипт умеет внутри каждого года сегментировать временнЫе периоды для максимизации количества выгружаемых постов; двигается вглубь веков, пока не достигнет заданной пользователем временнОй глубины или года с пустой выдачей
--- Если хотите для поиска дополнительных постов попробовать сегментирование по годам, просто нажмите Enter. Поиск может занять минуты и даже часы, но Вы СМОЖЕТЕ ПРЕРВАТЬ ИСПОЛНЕНИЕ СКРИПТА КНОПКОЙ и ВЫГРУЖЕННЫЕ ДАННЫЕ СОХРАНЯТСЯ
--- Если НЕ хотите, нажмите пробел и затем Enter''')

                try: decision = input()
                except EOFError:
                    print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                    decision = ''
                    silentMode = True

            if silentMode or not decision: # если silentMode , то максимизация количества выгружаемых постов выполняется принудительно
                while goS:
                    # print('Ищу текст запроса-фильтра в контенте за', year, 'год')
                    calendar = calendarWithinYear.calendarWithinYear(year)
                    calendarColumnS = calendar.columns

                    if year == int(momentCurrent.strftime('%Y')): calendarColumnS = calendarColumnS[:int(momentCurrent.strftime('%m'))]
                            # чтобы исключить проход по будущим месяцам текущего года

            # <Выгрузка постов за год year>
                    if yearMinByUser: # если пользователь ограничил временнОй диапазон
                        if year < yearMinByUser:
                            if not silentMode: print('Достигнут заданный пользователем временнОй горизонт:', yearMinByUser)
                            break # выход из while goS

                    itemS_additional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                     count,
                                                                                                     end_time=int(datetime(year, int(calendarColumnS[-1]), int(calendar[calendarColumnS[-1]].dropna().index[-1]), 23, 59, 59).timestamp()),
                                                                                                      # чтобы исключить проход по будущим месяцам текущего года
                                                                                                     
                                                                                                     fields=fields,
                                                                                                     iteration=iteration,
                                                                                                     keyOrder=keyOrder,
                                                                                                     latitude=latitude,
                                                                                                     longitude=longitude,
                                                                                                     params_extra=params_extra,
                                                                                                     pause=pause,
                                                                                                     q=q,
                                                                                                     silentMode=silentMode,
                                                                                                     start_from=None,
                                                                                                     start_time=int(datetime(year, 1, 1).timestamp()),
                                                                                                     version_api=version_api)

                    # print('itemS_additional:') # для отладки
                    # display(itemS_additional.sort_values('date')['date'].drop_duplicates()) # для отладки

                    itemS = dfsProcessor(complicatedNamePart,
                                         countTarget=countTarget,
                                         df_additional=itemS_additional,
                                         df_in=itemS,
                                         fields=fields,
                                         fileFormatChoice=fileFormatChoice,
                                         folder=folder,
                                         goS=goS,
                                         method=method,
                                         momentCurrent=momentCurrent,
                                         params_extra=params_extra,
                                         q=q,
                                         silentMode=silentMode,
                                         slash=slash,
                                         stage=stage,
                                         year=year,
                                         yearsRange=yearsRange)

                    if not goS and silentMode: return itemS # завершение исполнения скрипта вне dfsProcessor после сохранения в нём Temporal-снимка
                    # print('len(itemS):', len(itemS)) # для отладки

                    # print('goS 2:', goS) # для отладки
                    countTarget_withinYear = response['total_count']

                # <Выгрузка постов за год year без более глубокого сегментирования>
                    if len(itemS_additional) > 0 and countTarget_withinYear > 0 and countTarget_withinYear <= 1000: # ожидаемое (и реальное) количество постов больше нуля, но позволяет не прибегать к углублению сегментирования
                        if not silentMode: print('  Проход по всем следующим страницам с выдачей', '               ', end='\r')
                        while 'next_from' in response.keys() and goS:
                            start_from = response['next_from']
                            # print('    start_from:', start_from) # для отладки

                            itemS_additional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                             count,
                                                                                                             end_time=int(datetime(year, int(calendarColumnS[-1]), int(calendar[calendarColumnS[-1]].dropna().index[-1]), 23, 59, 59).timestamp()),
                                                                                                                # чтобы исключить проход по будущим месяцам текущего года
                                                                                                            
                                                                                                             fields=fields,
                                                                                                             iteration=iteration,
                                                                                                             keyOrder=keyOrder,
                                                                                                             latitude=latitude,
                                                                                                             longitude=longitude,
                                                                                                             params_extra=params_extra,
                                                                                                             pause=pause,
                                                                                                             q=q,
                                                                                                             silentMode=silentMode,
                                                                                                             start_from=start_from,
                                                                                                             start_time=int(datetime(year, 1, 1).timestamp()),
                                                                                                             version_api=version_api)

                            # print('itemS_additional:') # для отладки
                            # display(itemS_additional.sort_values('date')['date'].drop_duplicates()) # для отладки

                            itemS = dfsProcessor(complicatedNamePart,
                                                 countTarget=countTarget,
                                                 df_additional=itemS_additional,
                                                 df_in=itemS,
                                                 fields=fields,
                                                 fileFormatChoice=fileFormatChoice,
                                                 folder=folder,
                                                 goS=goS,
                                                 method=method,
                                                 momentCurrent=momentCurrent,
                                                 params_extra=params_extra,
                                                 q=q,
                                                 silentMode=silentMode,
                                                 slash=slash,
                                                 stage=stage,
                                                 year=year,
                                                 yearsRange=yearsRange)

                            if not goS and silentMode: return itemS # завершение исполнения скрипта вне dfsProcessor после сохранения в нём Temporal-снимка
                            # print('len(itemS):', len(itemS)) # для отладки
                            time.sleep(pause)
                    if not silentMode: print('  Искомых постов', countTarget, ', а выгруженных после добавления контента за', year, 'год:', len(itemS), '                    ')
                # </Выгрузка постов за год year без более глубокого сегментирования>

                # <Выгрузка постов за год year с сегментированием по месяцам>
                    elif countTarget_withinYear > 1000: # ожидаемое количество постов позволяет не прибегать к углублению сегментирования
                        for month in calendarColumnS:
                            if not goS: break
                            if not silentMode:
                                print('Ищу текст запроса-фильтра в контенте за',  month, 'месяц', year, 'года', '               ') # , end='\r'
                                print('  Заход на первую страницу выдачи', '               ', end='\r')

                            itemS_additional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                             count,
                                                                                                             end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1]), 23, 59, 59).timestamp()),
                                                                                                             fields=fields,
                                                                                                             iteration=iteration,
                                                                                                             keyOrder=keyOrder,
                                                                                                             latitude=latitude,
                                                                                                             longitude=longitude,
                                                                                                             params_extra=params_extra,
                                                                                                             pause=pause,
                                                                                                             q=q,
                                                                                                             silentMode=silentMode,
                                                                                                             start_from=None,
                                                                                                             start_time=int(datetime(year, int(month), 1).timestamp()),
                                                                                                             version_api=version_api)

                            # print('itemS_additional:') # для отладки
                            # display(itemS_additional.sort_values('date')['date'].drop_duplicates()) # для отладки

                            itemS = dfsProcessor(complicatedNamePart,
                                                 countTarget=countTarget,
                                                 df_additional=itemS_additional,
                                                 df_in=itemS,
                                                 fields=fields,
                                                 fileFormatChoice=fileFormatChoice,
                                                 folder=folder,
                                                 goS=goS,
                                                 method=method,
                                                 momentCurrent=momentCurrent,
                                                 params_extra=params_extra,
                                                 q=q,
                                                 silentMode=silentMode,
                                                 slash=slash,
                                                 stage=stage,
                                                 year=year,
                                                 yearsRange=yearsRange)

                            if not goS and silentMode: return itemS # завершение исполнения скрипта вне dfsProcessor после сохранения в нём Temporal-снимка
                            # print('len(itemS):', len(itemS)) # для отладки

                            if not silentMode: print('  Проход по всем следующим страницам с выдачей', '               ', end='\r')
                            while 'next_from' in response.keys() and goS:
                                start_from = response['next_from']
                                # print('    start_from:', start_from) # для отладки

                                itemS_additional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                                 count,
                                                                                                                 end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1]), 23, 59, 59).timestamp()),
                                                                                                                 fields=fields,
                                                                                                                 iteration=iteration,
                                                                                                                 keyOrder=keyOrder,
                                                                                                                 latitude=latitude,
                                                                                                                 longitude=longitude,
                                                                                                                 params_extra=params_extra,
                                                                                                                 pause=pause,
                                                                                                                 q=q,
                                                                                                                 silentMode=silentMode,
                                                                                                                 start_from=start_from,
                                                                                                                 start_time=int(datetime(year, int(month), 1).timestamp()),
                                                                                                                 version_api=version_api)

                                # print('itemS_additional:') # для отладки
                                # display(itemS_additional.sort_values('date')['date'].drop_duplicates()) # для отладки

                                itemS = dfsProcessor(complicatedNamePart,
                                                     countTarget=countTarget,
                                                     df_additional=itemS_additional,
                                                     df_in=itemS,
                                                     fields=fields,
                                                     fileFormatChoice=fileFormatChoice,
                                                     folder=folder,
                                                     goS=goS,
                                                     method=method,
                                                     momentCurrent=momentCurrent,
                                                     params_extra=params_extra,
                                                     q=q,
                                                     silentMode=silentMode,
                                                     slash=slash,
                                                     stage=stage,
                                                     year=year,
                                                     yearsRange=yearsRange)

                                if not goS and silentMode: return itemS # завершение исполнения скрипта вне dfsProcessor после сохранения в нём Temporal-снимка
                                # print('len(itemS_additional):', len(itemS_additional)) # для отладки
                                time.sleep(pause)
                    if not silentMode: print('  Искомых постов', countTarget, ', а выгруженных после добавления контента за', year, 'год:', len(itemS), '                    ')
                # </Выгрузка постов за год year с сегментированием по месяцам>

                # <Выгрузка постов за год year пуста>
                    elif len(itemS_additional) == 0 or countTarget_withinYear == 0: # ожидаемое или реальное количество постов равно нулю
                        if not silentMode:
                            print(
f'''  Дополнительных постов по Вашему запросу для года {year}, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта
--- Если НЕ хотите для поиска дополнительных постов попробовать двигаться к следующему месяцу вглубь веков, просто нажмите Enter
--- Если хотите, нажмите пробел и затем Enter''')

                            try: decision = input()
                            except EOFError:
                                print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                                decision = ''
                                silentMode = True

                        if silentMode or not decision: # если silentMode , то завершение сегментирования по годам выполняется принудительно
                            # print('Завершил сегментирования по годам')
                            if not silentMode: print('Искомых постов', countTarget, ', а выгруженных:', len(itemS), '          ')
                            break # выход из while goS
                # </Выгрузка постов за год year пуста>
            # </Выгрузка постов за год year>
                    year -= 1
        # pandas.set_option('display.max_columns', None)
        if not silentMode:
            display(itemS.head())
            print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0])

# 2.1.2 Экспорт выгрузки метода newsFeed.search и финальное завершение скрипта
        df2file.df2fileShell(complicatedNamePart,
                             df_in=itemS,
                             fileFormatChoice=fileFormatChoice,
                             folder=folder,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                             momentCurrent=momentCurrent)

        if not silentMode: 
            print('Скрипт исполнен. Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
            if folderWithTemporal:
                if os.path.exists(folderWithTemporal):
                    # print('folderWithTemporal:', folderWithTemporal) # для отладки
                    print(
'''Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, можно удалить её во избежание путаницы при следующих запусках скрипта
--- Если хотите их удалить, введите 'R' и нажмите Enter''')

                try: decision = input()
                except EOFError:
                    print('Похоже, исполняюсь неинтерактивныой среде (например, CI или cron), поэтому включаю silentMode')
                    decision = ''
                    silentMode = True

                if decision.lower() == 'r': shutil.rmtree(folderWithTemporal, ignore_errors=True)
        if fields: print(
f'''
Чтобы распаковать JSON из любого столбца, содержащего этот формат, в отдельный датафрейм, используйте такой код:
import pandas
column = 'Имя_столбца'
JSONS = []
for cellContent in Исходный_датафрейм[column].dropna():
    JSONS.extend(cellContent)
    Новый_датафрейм = pandas.json_normalize(JSONS).drop_duplicates('id').reset_index(drop=True)
''')

        if returnDfs or silentMode: return itemS
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if not silentMode: print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473''')

                sys.exit()

    except (Exception, KeyboardInterrupt) as excptn: # обработать прерывания по воле пользователя или ошибок 
        print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
        print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой

        itemS = dfsProcessor(complicatedNamePart,
                             countTarget=countTarget,
                             df_additional=itemS_additional,
                             df_in=itemS, # даже если itemS пуст, есть смысл сделать Temporal-снимок ради сохранения пользовательских аргументов
                             fields=fields,
                             fileFormatChoice=fileFormatChoice,
                             folder=folder,
                             goS=False,
                             method=method,
                             momentCurrent=momentCurrent,
                             params_extra=params_extra,
                             q=q,
                             silentMode=silentMode,
                             slash=slash,
                             stage=stage,
                             year=year,
                             yearsRange=yearsRange)

        if returnDfs or silentMode: return itemS
#         else:
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 if not silentMode: print(
# '''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
# Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473''')

#                 sys.exit()
