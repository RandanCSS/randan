#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and maximizes VK content extraction using the platform's official API method wall.get
(RU) Модуль для упрощения выгрузки контента ВК методом его API wall.get и максимизации размера этой выгрузки
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

        from randan.tools import coLabAdaptor, df2file, files2df, scrapingTools # модули для
            # (а) адаптации текущего скрипта к файловой системе CoLab
            # (б) сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (в) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (г) упрощения скрапинга

        import json, os, pandas, requests, shutil, time, warnings
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
# .. обработки выдачи, помогающая работе с ключами
def dfsProcessor(complicatedNamePart,
                 df_additional,
                 df_in,
                 domain,
                 fields,
                 fileFormatChoice,
                 filter,
                 folder,
                 goS, # единственная из функций, принимающая этот аргумент
                 method,
                 momentCurrent,
                 offset):

    if df_additional is None: df_additional = pandas.DataFrame()
    # if df_in is None: df_in = pandas.DataFrame()
    df = pandas.concat([df_in, df_additional])
    columnsForCheck = []
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    for column in df.columns: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        if column == 'id' or column.endswith('_id') or column.startswith('id_'): columnsForCheck.append(column)

    # print('Столбцы, по которым проверяю дублирующиеся записи:', columnsForCheck) # для отладки

    if columnsForCheck: df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True)
        # при дублировании записей из itemS из Temporal и от пользователя и новых записей, оставить новые

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
    if not goS and complicatedNamePart: # при silentMode complicatedNamePart is None
        temporal_exportPath = folder + f"{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"
        print(f"Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию '{temporal_exportPath}'")
        # print('temporal_exportPath:', temporal_exportPath) # для отладки
        if not os.path.exists(temporal_exportPath):
            os.makedirs(temporal_exportPath)
            print(f"Директория '{temporal_exportPath}' создана")
        # else: print(ff"Директория '{temporal_exportPath}' существует")

        scrapingTools.containerExport(temporal_exportPath + slash + 'domain.txt', domain if domain else '')

        if not fields: fields = []
        with open(temporal_exportPath + slash + 'fields.json', 'w', encoding='utf-8') as file:
            json.dump(fields, file, ensure_ascii=False, indent=4)

        scrapingTools.containerExport(temporal_exportPath + slash + 'filter.txt', filter if filter else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'method.txt', method)
        scrapingTools.containerExport(temporal_exportPath + slash + 'offset.txt', offset)

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

def wallGetCore(api_keyS,
                count,
                domain,
                fields,
                filter,
                iteration,
                keyOrder,
                offset,
                params_extra,
                pause,
                version_api):

    df_additional = pandas.DataFrame()
    goS = True

    params_basic = {'access_token': api_keyS[keyOrder], # обязательный параметр
                      'count': count, # опциональный параметр
                      'domain': domain, # обязательный параметр, но без него не будет результата
                      'extended': 1, # опциональный параметр
                      'fields': fields, # опциональный параметр
                      'filter': filter, # опциональный параметр
                      'offset': offset, # опциональный параметр
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
        response = requests.get('https://api.vk.ru/method/wall.get', params=params).json()
        # print('response', response) # для отладки
        if 'response' in response.keys():
            response = response['response']
            # print('    response.keys() в wallGetCore', response.keys()) # для отладки

            df_additional = pandas.json_normalize(response['items'])

            print('    Итерация №', iteration, ', number of items', len(response['items']), '                                        ', end='\r')
                # для визуализации процесса

            iteration += 1
            if len(df_additional) > 0: df_additional = scrapingVK_tools.dfColumnsProcessor(df_additional, fields, response)            
            break # нет смысла в новых итерациях цикла while goC
        else: goC, goS, keyOrder, pause, response, tryer = scrapingVK_tools.errorProcessor(api_keyS, keyOrder, pause, response, tryer)
    return df_additional, goS, iteration, keyOrder, pause

# 2. Основная функция
def wallGet(access_token=None,
            count=None,
            domain=None,
            fields=None,
            filter=None,
            folder=None,
            offset=None,
            params=None,
            returnDfs=False,
            version_api='5.199'):
    '''
    Функция для выгрузки характеристик контента ВК методом его API wall.get . Причём количество объектов выгрузки максимизируется посредством offset

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/wall.get , за исключением аргументов params и returnDfs
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params ,
    который обычно подаётся в метод get пакета requests
    access_token : str
           count : int
          domain : str
          fields : list
          filter : str
          folder : str
          offset : int
          params : dict -- в случае наличия готового словаря с аргументами метода https://dev.vk.com/ru/method/wall.get , чтобы не подавать эти аргументы по отдельности

       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными, если исполнение скрипта не прервано пользователем или из-за ошибки
      silentMode : bool -- в случае True в функции отключаются CLI-диалоги и экспорт результата исполнения функции для долговременного хранения 
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
    keyOrder = 0
    # name_in_folderCurrent = None
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС

    momentCurrent = datetime.now() # запрос текущего момента
    if not silentMode: 
        print('\nТекущий момент:', momentCurrent.strftime('%Y%m%d_%H%M'), '-- он будет использован для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    nameS_in_folderCurrent = os.listdir(folder)
    # print('nameS_in_folderCurrent:', nameS_in_folderCurrent) # для отладки
    api_keyS = scrapingTools.credentialsProcessor(access_token, 'VK', folder, 'https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8', nameS_in_folderCurrent, silentMode)

# Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
    # Поиск данных
    if not silentMode: # если основная функция этого модуля явлется частью другой функции, поиск данных не требуется -- они будут переданы
        print('Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
        for name_in_folderCurrent in nameS_in_folderCurrent:
            if name_in_folderCurrent.endswith('Temporal') and 'VK' in name_in_folderCurrent:
                folder_in_folderCurrent = folder + name_in_folderCurrent # имя файла достраивается до абсолютного пути
                if len(os.listdir(folder_in_folderCurrent)) < 5:  print(
f"В директории '{folder_in_folderCurrent}' есть директория, похожая на искомую Temporal , но в ней не 5-6 файлов => если это она, то либо повреждена, либо создалась при безрезультатном запуске; тогда Вам следует удалить её вручную") # допустимо 6 или 7 файлов

                else:
                    domainTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'domain.txt', str)

                    with open(f'{folder_in_folderCurrent}{slash}fields.json', 'r', encoding='utf-8') as file:
                        fieldsTemporal = json.load(file)

                    filterTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'filter.txt', str)
                    offsetTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'offset.txt', int)

                    print(f"Нашёл директорию '{folder_in_folderCurrent}'. В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:"
                          , '\n- скрипт остановился на offset', offsetTemporal)

                    print(f"- пользователь определил страницу: '{domainTemporal}'" if domainTemporal else '- пользователь НЕ определил страницу')
                    print(f"- пользователь определил поля: '{fieldsTemporal}'" if fieldsTemporal else '- пользователь НЕ определил поля')
                    print(f"- пользователь определил фильтр: '{filterTemporal}'" if filterTemporal else '- пользователь НЕ определил фильтр')

                    print(
'''--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter
--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите 'R' и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter''')

                    decision = input()
                    if decision.lower() == 'r':
                        shutil.rmtree(folder_in_folderCurrent, ignore_errors=True)

                        print(
f"Директорию '{folder_in_folderCurrent}' удалил, продолжаю поиск ещё одной директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку")

                    elif decision: print(
f"Директорию '{folder_in_folderCurrent}' пропустил, продолжаю поиск ещё одной директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку")

                    elif not decision: # Temporal-значения обретают статус постоянных и будут использоваться при исполнении скрипта
                        domain = domainTemporal
                        fields = fieldsTemporal
                        filter = filterTemporal
                        offset = offsetTemporal

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



    params_extra = {}
    if not params and not access_token and not count and not domain and not fields and not filter and not offset and not returnDfs:
        if not silentMode: print('Пользователь не подал аргументы') # для отладки
        experiencedMode = False

    else:
        experiencedMode = True
        KEYS_BASIC = {'access_token', 'count', 'domain', 'fields', 'filter', 'offset', 'v'}
        if params:
            if access_token: access_token = scrapingTools.argument_key_comparison(access_token, 'access_token', params)
            elif 'access_token' in params.keys(): access_token = params['access_token']
            # print('access_token:', access_token) # для отладки

            if count: count = scrapingTools.argument_key_comparison(count, 'count', params)
            elif 'count' in params.keys(): count = params['count']
            # print('count:', count) # для отладки

            if domain: domain = scrapingTools.argument_key_comparison(domain, 'domain', params)
            elif 'domain' in params.keys(): domain = params['domain']
            # print('domain:', domain) # для отладки

            if fields: fields = scrapingTools.argument_key_comparison(fields, 'fields', params)
            elif 'fields' in params.keys(): fields = params['fields']
            # print('fields:', fields) # для отладки

            if filter: filter = scrapingTools.argument_key_comparison(filter, 'filter', params)
            elif 'filter' in params.keys(): filter = params['filter']
            # print('filter:', filter) # для отладки

            if offset is not None: offset = scrapingTools.argument_key_comparison(offset, 'offset', params)
            elif 'offset' in params.keys(): offset = params['offset']
            # print('offset:', offset) # для отладки

            if version_api is not None: version_api = scrapingTools.argument_key_comparison(version_api, 'v', params)
            elif 'v' in params.keys(): version_api = params['v']
            # print('v:', v) # для отладки

            params_extra = {key: value for key, value in params.items() if key not in KEYS_BASIC}
                # всё, что осталось -- пользовательские аргументы метода

            params_extra = {key: value for key, value in params_extra.items() if value is not None}

        if count:
            if not isinstance(count, int): count = int(count)
            count = min(abs(count), 100)
        
        else: count = 100
        # print('count:', count) # для отладки

        if domain:
            if not isinstance(domain, str): domain = str(domain)
        # print('domain:', domain) # для отладки

        if offset is not None:
            if not isinstance(offset, int): offset = int(offset)
        # print('offset:', offset) # для отладки

    method = 'wall.get'
    if not silentMode:
        if not experiencedMode:
            print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.''')

        print(
f'''    Скрипт нацелен на выгрузку характеристик контента ВК методом его API {method} . Причём количество объектов выгрузки максимизируется посредством offset .
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК''')

        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0.2 Пользовательские настройки запроса к API ВК
    # if not silentMode продолжается
        if not domain: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать посты открытых страниц
--- Введите название интересующей страницы (персональной и группы), после чего нажмите Enter''')

            if folder_file: print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folder_file, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести название той же страницы, что и при формировании указанного Вами файла')

            domain = input()

        if not domain: domain = '80054288' # моя страница
        else: print('')

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    if domain: complicatedNamePart += '_' + domain if len(domain) < 50 else '_' + domain[:50]

    # Завершение if not silentMode

# 2.1 Первичный сбор контента методом get
# 2.1.0 Первое обращение к API
    iteration = 1 # номер итерации применения текущего метода
    pause = 0.25
    if not silentMode: print(
f'В скрипте используются следующие аргументы метода {method} API ВК: count, domain, fields, filter, offset .',
'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
f'Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/{method} ,',
f'-- можете подать их в скобки функции wallGet перед её запуском или скопировать код исполняемого сейчас скрипта и сделать это внутри кода внутри метода {method} в разделе 2')

        # print('experiencedMode:', experiencedMode) # для отладки
        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа

    if offset is None: offset = 0 # если offset ни подан пользователем, ни сохранён при прошлом запуске
    try: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        while True:
            itemS_additional, goS, iteration, keyOrder, pause = wallGetCore(api_keyS,
                                                                            count,
                                                                            domain,
                                                                            fields,
                                                                            filter,
                                                                            iteration,
                                                                            keyOrder,
                                                                            offset,
                                                                            params_extra,
                                                                            pause,
                                                                            version_api)

            # после самого первого исполнения wallGetCore iteration == 2
            # print('goS:', goS) # для отладки
            if not goS or goS and len(itemS_additional) > 0: # сигнал остановиться; число выгруженных постов не важно ИЛИ сигнала остановиться нет, а выгруженные посты есть
                # в первом случае dfsProcessor завешит скрипт и создаст Temporal-снимок, во втором -- объединит свежевыгруженные посты с ранее выгруженными
                # Вариантов сигнала в этом и аналогичном скриптах три: len(itemS_additional) > 0 , not goS , 'next_from' is not in response.keys()

                itemS = dfsProcessor(complicatedNamePart,
                                     itemS_additional,
                                     itemS,
                                     domain,
                                     fields,
                                     fileFormatChoice,
                                     filter,
                                     folder,
                                     goS, # единственная из функций, принимающая этот аргумент
                                     method,
                                     momentCurrent,
                                     offset)

                offset += len(itemS_additional)
                time.sleep(pause)

            if goS and len(itemS_additional) == 0: # сигнала остановиться нет, но и выгруженных постов нет
                print(
f'''{'Дополнительных' if iteration > 2 else 'Искомых'} постов на странице из Вашего запроса, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта{'' if iteration > 2 else '. Что можно сделать для выгрузки постов? Поменяйте настройки запроса и запустите скрипт с начала'}''')
                if iteration >= 2:
# 2.1.2 Экспорт выгрузки метода get и финальное завершение скрипта
                    df2file.df2fileShell(complicatedNamePart,
                                         df_in=itemS,
                                         fileFormatChoice=fileFormatChoice,
                                         folder=folder,
                                         method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                                         momentCurrent=momentCurrent)

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
        # # display(itemS)
        # if itemS is None or itemS.empty: itemS = pandas.DataFrame()
        dfsProcessor(complicatedNamePart,
                     itemS_additional,
                     itemS,
                     domain,
                     fields,
                     fileFormatChoice,
                     filter,
                     folder,
                     False, # единственная из функций, принимающая аргумент goS
                     method,
                     momentCurrent,
                     offset)

        # if returnDfs: return itemS
