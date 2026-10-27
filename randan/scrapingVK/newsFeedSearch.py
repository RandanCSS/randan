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

        from randan.tools import calendarWithinYear, coLabAdaptor, df2file, files2df, scrapingTools, varPreprocessor # авторские модули для
            # (а) работы с календарём конкретного года
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (г) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (д) упрощения скрапинга
            # (е) предобработки переменных номинального, порядкового, интервального и более высокого типа шкалы

        import json, numpy, os, pandas, re, shutil, time, requests, warnings
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
                 df_additional,
                 dfFinal, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода search
                 dfIn,
                 fields,
                 fileFormatChoice,
                 folder,
                 goS, # единственная из функций, принимающая этот аргумент
                 method,
                 momentCurrent,
                 params_extra,
                 q,
                 slash,
                 stage,
                 countTarget,
                 year,
                 yearsRange):

    df = pandas.concat([dfIn, df_additional])
    columnsForCheck = []
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id' in column: columnsForCheck.append(column)
    # print('Столбцы, по которым проверяю дублирующиеся записи:', columnsForCheck) # для отладки

    if columnsForCheck: df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True)
        # при дублировании объектов из itemS из Temporal и от пользователя и новых объектов, оставить новые

    if not goS and complicatedNamePart: # при silentMode complicatedNamePart is None
        temporal_exportPath = folder + f"{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"
        print(f"Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию '{temporal_exportPath}'")
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

        if stage > 1: stage -= 1 # чтобы не пропускать неполностью пройденный stage
        scrapingTools.containerExport(temporal_exportPath + slash + 'stageTarget.txt', str(stage)) # stage и stageTarget принимает значения [0; 3]

        scrapingTools.containerExport(temporal_exportPath + slash + 'countTarget.txt', str(countTarget) if countTarget else '0')
        scrapingTools.containerExport(temporal_exportPath + slash + 'year.txt', str(year)) # год, на котором остановилось исполнение скрипта
        scrapingTools.containerExport(temporal_exportPath + slash + 'yearsRange.txt', yearsRange if yearsRange else '')

        df2file.df2fileShell(complicatedNamePart=f'{complicatedNamePart}_Temporal',
                             df_in=df,
                             fileFormatChoice=fileFormatChoice,
                             folder=temporal_exportPath,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла  
                             momentCurrent=momentCurrent)

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
                       start_from,
                       start_time,
                       version_api):

    # print('    start_from', start_from) # для отладки
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

    tryer = 0
    while goC:
        response = requests.get('https://api.vk.ru/method/wall.get', params=params).json()
        # print('response', response) # для отладки
        if 'response' in response.keys():
            response = response['response']
            # print('    response.keys() внутри newsfeedSearchCore', response.keys()) # для отладки

            df_additional = pandas.json_normalize(response['items'])

            print('    Итерация №', iteration, ', number of items', len(response['items']), '                                        ', end='\r')
                # для визуализации процесса

            iteration += 1
            if len(df_additional) > 0: df_additional = scrapingVK_tools.dfColumnsProcessor(df_additional, fields, response)            
            break # нет смысла в новых итерациях цикла while goC
        else: goC, goS, keyOrder, pause, response, tryer = scrapingVK_tools.errorProcessor(api_keyS, keyOrder, pause, response, tryer)
    return df_additional, goS, iteration, keyOrder, pause, response

# 2. Основная функция
def newsFeedSearch(access_token=None,
                   count=None,
                   end_time=None, # !!! можно сделать 2 варианта подачи: год или Unix
                   fields=None,
                   latitude=None,
                   longitude=None,
                   params=None,
                   q=None,
                   returnDfs=False,
                   start_time=None,
                   version_api='5.199'):
    '''
    Функция для выгрузки характеристик контента ВК методом его API newsfeed.search . Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/newsfeed.search , за исключением аргументов params и returnDfs
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params , который обычно подаётся в метод get пакета requests
    access_token : str
           count : int
        end_time : int -- формат Unix
          fields : list
        latitude : int
       longitude : int
          params : dict -- в случае наличия готового словаря с аргументами метода https://dev.vk.com/ru/method/newsfeed.search , чтобы не подавать эти аргументы по отдельности
               q : str
       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными
      silentMode : bool -- в случае True в функции отключаются CLI-диалоги и экспорт результата исполнения функции для долговременного хранения 
      start_time : int -- формат Unix
     version_api : str
    '''

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API ВК
    complicatedNamePart = None # для silentMode
    fileFormatChoice = 'xlsx' # базовый формат сохраняемых файлов; формат json добавляется опционально через наличие columnsToJSON
    folder_file = None
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    folder_in_folderCurrent = None
    folderWithTemporal = None
    goS = True
    itemS = pandas.DataFrame()
    itemS_additional = pandas.DataFrame() # на случай, если сигнал прерывания поступит до dfsProcessor
    itemsMonthlyAdditional = None # чтобы обработать сигнал прерывания, поданный на любом этапе сбора данных
    itemsYearlyAdditional = None # чтобы обработать сигнал прерывания, поданный на любом этапе сбора данных   
    keyOrder = 0
    params_extra = {}
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    stageTarget = 0 # stageTarget принимает значения [0; 3] и относится к стадиям скрипта
    yearMinByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearMaxByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearsRange = None

    momentCurrent = datetime.now() # запрос текущего момента
    print('\nТекущий момент:', momentCurrent.strftime('%Y%m%d_%H%M'), '-- он будет использован для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')
    year = int(momentCurrent.strftime('%Y')) # в случае отсутствия пользовательского временнОго диапазона
        # с этого года возможно сегментирование по годам вглубь веков (пока выдача не пустая)

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    nameS_in_folderCurrent = os.listdir()
    # print('nameS_in_folderCurrent:', nameS_in_folderCurrent) # для отладки
    api_keyS = scrapingTools.credentialsProcessor(access_token, 'VK', folder, 'https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8', nameS_in_folderCurrent, silentMode)


# Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
    # Поиск данных
    if not silentMode: # если основная функция этого модуля явлется частью другой функции, поиск данных не требуется -- они будут переданы
        print('Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
        for name_in_folderCurrent in nameS_in_folderCurrent:
            if name_in_folderCurrent.endswith('Temporal') and 'VK' in name_in_folderCurrent:
                folder_in_folderCurrent = folder + name_in_folderCurrent # имя файла достраивается до абсолютного пути
                if len(os.listdir(folder_in_folderCurrent)) < 9: print(
f"В директории '{folder_in_folderCurrent}' есть директория, похожая на искомую Temporal , но в ней не 9-10 файлов => если это она, то либо повреждена, либо создалась при безрезультатном запуске; тогда Вам следует удалить её вручную") # допустимо 6 или 7 файлов
                    countTargetTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'countTarget.txt', int)

                    with open(f'{folder_in_folderCurrent}{slash}fields.json', 'r', encoding='utf-8') as file:
                        fieldsTemporal = json.load(file)

                    methodTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'method.txt', str)

                    with open(f'{folder_in_folderCurrent}{slash}params_extra.json', 'r', encoding='utf-8') as file:
                        params_extraTemporal = json.load(file)

                    try: qTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'q.txt', str)
                    except Exception as excptn:
                        print('Exception в searchByText') # для отладки
                        print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
                        print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой
                        qTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'q.txt', str)

                    stageTargetTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'stageTarget.txt', int)
                    yearTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'year.txt', int)

                    yearsRangeTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'yearsRange.txt', str)
                    if yearsRangeTemporal == '': yearsRangeTemporal = None # для единообразия

                    print(f"Нашёл директорию '{folder_in_folderCurrent}'. В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:"
                          , '\n- скрипт остановился на методе', methodTemporal)

                    if yearTemporal < int(momentCurrent.strftime('%Y')): print('- и на годе (при сегментировани по годам)', yearTemporal)
                    print(f"- пользователь сформулировал запрос-фильтр как '{qTemporal}'" if qTemporal else '- пользователь НЕ сформулировал запрос-фильтр')
                    print(f'- пользователь ограничил временнОй диапазон границами {yearsRangeTemporal}' if yearsRangeTemporal else '- пользователь НЕ ограничил временнОй диапазон')
                    print(f"- пользователь подал дополнительные настройки посредством словаря params: '{params_extraTemporal}'" if params_extraTemporal else '')

                    print(
f'''В дальнейшей работе могу опираться на эти промежуточные результаты. При этом случае ПРОТИВОРЕЧИЯ между найденными в директории '{folder_in_folderCurrent}'и поданными Вами при текущем запуске аргументами будут применены ПЕРВЫЕ.
--- Если следует опираться на промежуточные результаты, нажмите Enter;  
--- Если эти промежуточные результаты уже не актуальны и хотите, чтобы я удалил их, введите 'R' и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter''')

                    decision = input()
                    if decision.lower() == 'r':
                        shutil.rmtree(folder_in_folderCurrent, ignore_errors=True)

                        print(
f"Директорию '{folder_in_folderCurrent}' удалил, продолжаю поиск ещё одной директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку")

                    elif decision: print(
f"Директорию '{folder_in_folderCurrent}' пропустил, продолжаю поиск ещё одной директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку")

                    elif not decision: # Temporal-значения обретают статус постоянных и будут использоваться при исполнении скрипта
                        countTarget = countTargetTemporal
                        fields = fieldsTemporal
                        method = methodTemporal
                        params_extra = params_extraTemporal
                        q = qTemporal
                        stageTarget = stageTargetTemporal
                        year = yearTemporal
                        yearsRange = yearsRangeTemporal

                        temporalNameS = os.listdir(folder_in_folderCurrent)
                        folderWithTemporal, itemS = scrapingTools.get_itemsTemporal(temporalNameS)

        if not folderWithTemporal: # если itemsTemporal, в т.ч. пустой, не существует и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
            print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта')

# Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    # if not silentMode продолжается
        print(
f'''
Возможно, Вы располагаете файлом, в котором есть ранее выгруженные из ВК методом {method} данные, и который хотели бы дополнить?
Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом формата XLSX, укажите полный путь, включая название файла, и нажмите Enter.
Затем при необходимости сможете добавить к нему другие располагаемые файлы''')

        while True:
            folder_file = input()
            if len(folder_file) == 0:
                folder_file = None # для унификации
                break

            else:
                itemS, error, folder = files2df.files2df(folder_file)
                if error:
                    if 'No such file or directory' in error: print('Путь:', folder_file, '-- не существует; попробуйте, пожалуйста, ещё раз..')
                    else:
                        print('error из files2df:', error)
                        break
                else: break
            # display('itemS:', itemS)
# Теперь определены объекты: folder и folder_file (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский)

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
            if start_time is not None:
                if not isinstance(start_time, int): start_time = int(start_time)
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
            if not isinstance(latitude, float): latitude = float(latitude)
        # print('latitude:', latitude) # для отладки

        if longitude is not None:
            if not isinstance(longitude, float): longitude = float(longitude)
        # print('longitude:', longitude) # для отладки
        
        if q:
            if not isinstance(q, str): q = str(q)
        # print('q:', q) # для отладки

        if start_time is not None:
            if not isinstance(start_time, int): start_time = int(start_time)
        # print('start_time:', start_time) # для отладки

# На выходе аргументы + params_extra . Приоритет определён при первой подаче аргументов и словаря params, этот приоритет транслируется через в Temporal, следовательно, при извлечении из Temporal перепроверять приоритет не требуется

    method = 'newsfeed.search'
    if not experiencedMode:
        print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.'''
              )
    print(
f'''    Скрипт нацелен на выгрузку характеристик контента ВК методом его API {method} . Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам. При этом следует учесть, что текущая версия API ВК, на которой и основан скрипт, 'не умеет' обращаться к контенту, опубликованному ранее 2020 года.
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК'''
          )
    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0.2 Пользовательские настройки запроса к API ВК
    if not silentMode: # если основная функция этого модуля явлется частью другой функции, поиск данных не требуется -- они будут переданы
        if not q: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать контент в постах открытых аккаунтов по ТЕКСТОВОМУ ЗАПРОСУ-ФИЛЬТРУ
--- Введите текст запроса-фильтра, который ожидаете найти в постах, после чего нажмите Enter''')

            if folderFile: print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folderFile, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла')

            q = input()
            if q == '': q = None # для единообразия
            else: print('')

        # Ограничения временнОго диапазона
        if not start_time and not end_time and not yearsRange: # если пользователь не подал эти аргументы в рамках experiencedMode
            print(
f'''Если требуется конкретный временнОй диапазон, то можно использовать его не на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными. Проблема в том, что без назначения временнОго диапазона метод {method} выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту
--- Поэтому если всё же требуется назначить временнОй диапазон на этапе выгрузки данных, назначьте его годами (а не более мелкими единицами времени). Для назначения диапазона введите без кавычек минимальный год диапазона, тире, максимальный год диапазона (минимум и максимум могут совпадать в такой записи: 'год-тот же год') и нажмите Enter !!! Cледует учесть, что текущая версия API ВК 'не умеет' обращаться к контенту, опубликованному ранее 2020 года
--- Если НЕ требуется назначить временнОй диапазон на этапе выгрузки данных, нажмите Enter''')

            while True:
                yearsRange = input()
                if len(yearsRange) != 0:
                    yearsRange = re.sub(r'\s*', '', yearsRange)
                    if '-' in yearsRange:
                        yearsRange = yearsRange.split('-')
                        if len(yearsRange) == 2:
                            if (len(yearsRange[0]) == 4) & (len(yearsRange[1]) == 4):
                                yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)
                                year = yearMaxByUser
                                break
                            else: print('--- Вы ввели год[ы] НЕ из четырёх цифр. Попробуйте ещё раз..')
                        else: print('--- Вы ввели тире, но при этом ввели НЕ два года. Попробуйте ещё раз..')
                    else: print('--- Вы НЕ ввели тире. Попробуйте ещё раз..')
                else:
                    yearsRange = None # для унификации
                    break
        if start_time is not None:
            yearMinByUser = int(datetime.fromtimestamp(start_time).strftime('%Y')) # из experiencedMode
            # print('elif start_time is not None', yearMinByUser) # для отладки

        if end_time is not None:
            yearMaxByUser = int(datetime.fromtimestamp(end_time).strftime('%Y')) # из experiencedMode
            # print('elif end_time is not None:', yearMaxByUser) # для отладки
            year = yearMaxByUser

        if yearMinByUser and not yearMaxByUser: yearMaxByUser = int(momentCurrent.strftime('%Y')) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
        elif not yearMinByUser and yearMaxByUser: yearMinByUser = 1970 # в случае отсутствия пользовательской нижней временнОй границы при наличии верхней

        # print('yearMinByUser', yearMinByUser) # для отладки
        # print('yearMaxByUser', yearMaxByUser) # для отладки

        if not start_time and yearMinByUser: start_time = int(datetime(yearMinByUser, 1, 1).timestamp()) # int(time.mktime(datetime(yearMinByUser, 1, 1).timetuple()))
        if not end_time and yearMaxByUser: end_time = int(datetime(yearMaxByUser, 12, 31).timestamp())

        if yearsRange: print('') # чтобы был отступ, если пользователь подал этот аргумент

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    if q: complicatedNamePart += '_' + q if len(q) < 50 else '_' + q[:50]
    complicatedNamePart += '_' + str(yearMinByUser) + '-' + str(yearMaxByUser) if yearMinByUser or yearMaxByUser else ''
    # print('complicatedNamePart', complicatedNamePart)

    # Завершение if not silentMode

# 2.1 Первичный сбор контента методом search
# 2.1.0 Первое обращение к API БЕЗ аргументов start_time, end_time (этап stage = 0)
    iteration = 1 # номер итерации применения текущего метода
    pause = 0.25
    stage = 0
    if not silentMode: print(
f'''В скрипте используются следующие аргументы метода {method} API ВК: q, start_from, start_time, end_time, expand .
Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/{method} ,
-- можете подать их в скобки функции newsFeedSearch перед её запуском или скопировать код исполняемого сейчас скрипта и сделать это внутри кода внутри метода {method} в разделе 2''')

        # print('experiencedMode:', experiencedMode) # для отладки
        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа

    try: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
            print('Первое обращение к API -- прежде всего, чтобы узнать примерное число доступных релевантных объектов')
            # print('    start_from', start_from) # для отладки

            itemsAdditional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
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
                                                                                            start_from=None,
                                                                                            start_time=start_time,
                                                                                            version_api=version_api)

            # print('goS 645:', goS) # для отладки
            if goS: # проверка, что функция  завершилась успехом; обработка вилки с countTarget == 0
                countTarget = response['total_count']
                if countTarget == 0:
                    print(
'''  Искомых объектов на серверах ВК по Вашему запросу, ноль, поэтому нет смысла в продолжении исполнения скрипта. Что делать? Поменяйте настройки запроса и запустите скрипт с начала'''
                          )
                    warnings.filterwarnings('ignore')
                    print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'''
                          )
                    sys.exit()
            else: countTarget = None # проверка, что функция newsfeedSearchCore завершилась успехом

            itemS = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                 df_additional=itemsAdditional,
                                 dfFinal=itemS,
                                 dfIn=itemS,
                                 fields=fields,
                                 fileFormatChoice=fileFormatChoice,
                                 folder=folder,
                                 goS=goS,
                                 method=method,
                                 momentCurrent=momentCurrent,
                                 params_extra=params_extra,
                                 q=q,
                                 slash=slash,
                                 stage=stage,
                                 countTarget=countTarget,
                                 year=year,
                                 yearsRange=yearsRange)

            print('  Проход по всем следующим страницам с выдачей          ')
            while ('next_from' in response.keys()) & goS:
                start_from = response['next_from']
                # print('    start_from', start_from) # для отладки
                # print('goS 700:', goS) # для отладки

                itemsAdditional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
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
                                                                                                start_from,
                                                                                                start_time,
                                                                                                version_api)

                # print('''    response['next_from'] после newsfeedSearchCore''', response['next_from']) # для отладки

                itemS = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                     df_additional=itemsAdditional,
                                     dfFinal=itemS,
                                     dfIn=itemS,
                                     fields=fields,
                                     fileFormatChoice=fileFormatChoice,
                                     folder=folder,
                                     goS=goS,
                                     method=method,
                                     momentCurrent=momentCurrent,
                                     params_extra=params_extra,
                                     q=q,
                                     slash=slash,
                                     stage=stage,
                                     countTarget=countTarget,
                                     year=year,
                                     yearsRange=yearsRange)

            print('  Искомых объектов', countTarget, ', а найденных БЕЗ сегментирования по годам и месяцам:', len(itemS))

# 2.1.1 Этап сегментирования по годам и месяцам (stage = 1) # !!! можно сделать динамическое определение периода, который наиболее приближает размер допустимой выдачи (1000) к бенчмарку
        stage = 1
        if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
            if len(itemS) < countTarget:
            # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ сегментирования по годам и месяцам
                print(
f'''Метод {method} выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту. Скрипт умеет внутри каждого года помесячно выгружать контент, двигаясь вглубь веков, пока не достигнет заданной пользователем левой границы временнОго диапазона или года с пустой выдачей
--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам и месяцам, просто нажмите Enter. Поиск может занять минуты и даже часы, но Вы СМОЖЕТЕ ПРЕРВАТЬ ИСПОЛНЕНИЕ СКРИПТА КНОПКОЙ и ВЫГРУЖЕННЫЕ ДАННЫЕ СОХРАНЯТСЯ
--- Если НЕ хотите, нажмите пробел и затем Enter'''
                      )
                if len(input()) == 0:
                    while goS:
                        # print('Ищу текст запроса-фильтра в контенте за', year, 'год')
                        calendar = calendarWithinYear.calendarWithinYear(year)
                        itemsYearlyAdditional = pandas.DataFrame()
                        calendarColumnS = calendar.columns
                        if year == int(momentCurrent.strftime('%Y')): calendarColumnS = calendarColumnS[:int(momentCurrent.strftime('%m'))]
                                # чтобы исключить проход по будущим месяцам текущего года
                        for month in calendarColumnS:
                            print('Ищу текст запроса-фильтра в контенте за',  month, 'месяц', year, 'года', '               ') # , end='\r'
                            print('  Заход на первую страницу выдачи', '               ', end='\r')

                            itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                                   count,
                                                                                                                   end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1])).timestamp()),
                                                                                                                   fields=fields,
                                                                                                                   iteration=iteration,
                                                                                                                   keyOrder=keyOrder,
                                                                                                                   latitude=latitude,
                                                                                                                   longitude=longitude,
                                                                                                                   params_extra=params_extra,
                                                                                                                   pause=pause,
                                                                                                                   q=q,
                                                                                                                   start_from=None,
                                                                                                                   start_time=int(datetime(year, int(month), 1).timestamp()),
                                                                                                                   version_api=version_api)

                            # print('itemsMonthlyAdditional:') # для отладки
                            # display(itemsMonthlyAdditional.sort_values('date')['date'].drop_duplicates()) # для отладки

                            itemsYearlyAdditional = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                                                 df_additional=itemsMonthlyAdditional,
                                                                 dfFinal=itemS,
                                                                 dfIn=itemsYearlyAdditional,
                                                                 fields=fields,
                                                                 fileFormatChoice=fileFormatChoice,
                                                                 folder=folder,
                                                                 goS=goS,
                                                                 method=method,
                                                                 momentCurrent=momentCurrent,
                                                                 params_extra=params_extra,
                                                                 q=q,
                                                                 slash=slash,
                                                                 stage=stage,
                                                                 countTarget=countTarget,
                                                                 year=year,
                                                                 yearsRange=yearsRange)

                            # print('len(itemsYearlyAdditional):', len(itemsYearlyAdditional)) # для отладки

                            print('  Проход по всем следующим страницам с выдачей', '               ', end='\r')

                            while ('next_from' in response.keys()) & goS:
                                start_from = response['next_from']
                                # print('    start_from', start_from) # для отладки

                                itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = newsfeedSearchCore(api_keyS,
                                                                                                                       count,
                                                                                                                       end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1])).timestamp()),
                                                                                                                       params_extra=params_extra,
                                                                                                                       fields=fields,
                                                                                                                       iteration=iteration,
                                                                                                                       keyOrder=keyOrder,
                                                                                                                       latitude=latitude,
                                                                                                                       longitude=longitude,
                                                                                                                       params_extra=params_extra,
                                                                                                                       pause=pause,
                                                                                                                       q=q,
                                                                                                                       start_from=start_from,
                                                                                                                       start_time=int(datetime(year, int(month), 1).timestamp()),
                                                                                                                       version_api=version_api)

                                # print('itemsMonthlyAdditional:') # для отладки
                                # display(itemsMonthlyAdditional.sort_values('date')['date'].drop_duplicates()) # для отладки

                                itemsYearlyAdditional = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                                                     df_additional=itemsMonthlyAdditional,
                                                                     dfFinal=itemS,
                                                                     dfIn=itemsYearlyAdditional,
                                                                     fields=fields,
                                                                     fileFormatChoice=fileFormatChoice,
                                                                     folder=folder,
                                                                     goS=goS,
                                                                     method=method,
                                                                     momentCurrent=momentCurrent,
                                                                     params_extra=params_extra,
                                                                     q=q,
                                                                     slash=slash,
                                                                     stage=stage,
                                                                     countTarget=countTarget,
                                                                     year=year,
                                                                     yearsRange=yearsRange)

                                # print('len(itemsYearlyAdditional):', len(itemsYearlyAdditional)) # для отладки
                                time.sleep(pause)
                        # print('itemsYearlyAdditional:') # для отладки
                        # display(itemsYearlyAdditional.sort_values('date')['date'].drop_duplicates()) # для отладки

                        itemS = dfsProcessor(complicatedNamePart=complicatedNamePart,
                                             df_additional=itemsYearlyAdditional,
                                             dfFinal=itemS,
                                             dfIn=itemS,
                                             fields=fields,
                                             fileFormatChoice=fileFormatChoice,
                                             folder=folder,
                                             goS=goS,
                                             method=method,
                                             momentCurrent=momentCurrent,
                                             params_extra=params_extra,
                                             q=q,
                                             slash=slash,
                                             stage=stage,
                                             countTarget=countTarget,
                                             year=year,
                                             yearsRange=yearsRange)

                        # print('len(itemS):', len(itemS)) # для отладки
                        # display(itemS.head()) # для отладки
                        # print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0]) # для отладки

                        if len(itemsYearlyAdditional) == 0:
                            print(f'\nВыдача для года {year} -- пуста'
                                  , '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать двигаться к следующему месяцу вглубь веков, просто нажмите Enter'
                                  , '\n--- Если хотите, нажмите пробел и затем Enter')
                            if len(input()) == 0:
                                # print(f'\nЗавершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser}')
                                break

                        elif yearMinByUser: # если пользователь ограничил временнОй диапазон
                            if year <= yearMinByUser:
                                print(f'Завершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser}')
                                break

                        print('  Искомых объектов', countTarget, ', а найденных после добавления контента', year, 'года:', len(itemS), '                    ')
                        year -= 1
                    print('Искомых объектов', countTarget, ', а найденных:', len(itemS), '          ')

            # pandas.set_option('display.max_columns', None)
            display(itemS.head())
            print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0])

        elif stage < stageTarget:
            print(
f'''
Этап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"'''
                  )

# 2.1.2 Экспорт выгрузки метода search и финальное завершение скрипта
        df2file.df2fileShell(complicatedNamePart=complicatedNamePart,
                             currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             dfIn=itemS,
                             fileFormatChoice=fileFormatChoice,
                             folder=folder,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method) # чтобы избавиться от лишней точки в имени файла

                    print('Скрипт исполнен. Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
                    if folderWithTemporal:
                        if os.path.exists(folderWithTemporal):
                            # print('folderWithTemporal:', folderWithTemporal) # для отладки
                            print(
'''Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, можно удалить её во избежание путаницы при следующих запусках скрипта
--- Если хотите их удалить, введите 'R' и нажмите Enter''')

                        decision = input()
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

                if returnDfs: return itemS
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473''')

                        sys.exit()

    except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        # display(itemS)
        if len(itemS) > 0:
            if itemsYearlyAdditional is not None:
                df_additional = itemsYearlyAdditional
                dfFinal = itemS
                dfIn = itemS

            elif itemsMonthlyAdditional is not None:
                df_additional = itemsMonthlyAdditional
                dfFinal = itemS
                dfIn = itemsYearlyAdditional

            elif itemsAdditional is not None:
                df_additional = itemsAdditional
                dfFinal = itemS
                dfIn = itemS

            dfsProcessor(complicatedNamePart=complicatedNamePart,
                         df_additional=df_additional,
                         dfFinal=itemS,
                         dfIn=itemS,
                         fields=fields,
                         fileFormatChoice=fileFormatChoice,
                         folder=folder,
                         goS=False,
                         method=method,
                         momentCurrent=momentCurrent,
                         params_extra=params_extra,
                         q=q,
                         slash=slash,
                         stage=stage,
                         countTarget=countTarget,
                         year=year,
                         yearsRange=yearsRange)

            if returnDfs: return itemS

# Введённый аргумент end_time приравнять к текущему моменту, чтобы не было прохода по лишним месяцам
# Сделать выбор временнОго диапазона динамическим
