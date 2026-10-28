#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and maximizes VK content extraction using the platform's official API method wall.getComments
(RU) Модуль для упрощения выгрузки контента ВК методом его API wall.getComments и максимизации размера этой выгрузки
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

        from randan.tools import calendarWithinYear, coLabAdaptor, df2file, files2df, scrapingTools # модули для
            # (а) работы с календарём конкретного года
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (г) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (д) упрощения скрапинга

        import json, os, pandas, shutil, requests, time, warnings
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
                 coLabFolder,
                 df_additional,
                 dfFinal, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода getComments
                 dfIn,
                 fields,
                 fileFormatChoice,
                 goS, # единственная из функций, принимающая этот аргумент
                 method,
                 momentCurrent,
                 offset,
                 owner_id,
                 params_extra,
                 post_id,
                 slash):

    df = pandas.concat([dfIn, df_additional])
    columnsForCheck = []
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id' in column: columnsForCheck.append(column)

    # print('Столбцы, по которым проверяю дублирующиеся записи:', columnsForCheck) # для отладки

    df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True)
        # при дублировании записей из itemS из Temporal и от пользователя и новых записей, оставить новые

    if not goS and complicatedNamePart: # при silentMode complicatedNamePart is None
        temporal_exportPath = folder + f"{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"
        print(f"Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию '{temporal_exportPath}'")
        # print('temporal_exportPath:', temporal_exportPath) # для отладки
        if not os.path.exists(temporal_exportPath):
            os.makedirs(temporal_exportPath)
            print(f"Директория '{temporal_exportPath}' создана")
        # else: print(ff"Директория '{temporal_exportPath}' существует")

        if not fields: fields = []
        with open(temporal_exportPath + slash + 'fields.json', 'w', encoding='utf-8') as file:
            json.dump(fields, file, ensure_ascii=False, indent=4)

        scrapingTools.containerExport(temporal_exportPath + slash + 'filter.txt', filter if filter else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'method.txt', method)
        scrapingTools.containerExport(temporal_exportPath + slash + 'offset.txt', offset)
        scrapingTools.containerExport(temporal_exportPath + slash + 'owner_id.txt', owner_id if owner_id else '')

        with open(temporal_exportPath + slash + 'params_extra.json', 'w', encoding='utf-8') as file:
            json.dump(params_extra, file, ensure_ascii=False, indent=4)

        scrapingTools.containerExport(temporal_exportPath + slash + 'post_id.txt', post_id if post_id else '')

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

# .. метода getComments из API ВК, помогающая работе с ключами
def wallGetCommentsCore(api_keyS,
                        count,
                        fields,
                        iteration,
                        keyOrder,
                        offset,
                        owner_id,
                        params_extra,
                        pause,
                        post_id,
                        version_api):

    df_additional = pandas.DataFrame()
    goS = True

    params_basic = {'access_token': api_keyS[keyOrder], # обязательный параметр
                    'count': count, # опциональный параметр
                    'extended': 1, # опциональный параметр
                    'fields': fields, # опциональный параметр
                    'need_likes': 1, # опциональный параметр
                    'offset': offset, # опциональный параметр
                    'owner_id': owner_id, # опциональный параметр, но без него не будет результата
                    'post_id': post_id, # опциональный параметр, но без него не будет результата
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
            # print('    response.keys() внутри wallgetCommentsCore', response.keys()) # для отладки

            df_additional = pandas.json_normalize(response['items'])

            print('    Итерация №', iteration, ', number of items', len(response['items']), '                                        ', end='\r')
                # для визуализации процесса

            iteration += 1
            if len(df_additional) > 0: df_additional = scrapingVK_tools.dfColumnsProcessor(df_additional, fields, response)            
            break # нет смысла в новых итерациях цикла while goC
        else: goC, goS, keyOrder, pause, response, tryer = scrapingVK_tools.errorProcessor(api_keyS, keyOrder, pause, response, tryer)
    return df_additional, goS, iteration, keyOrder, pause, response

# 2. Основная функция
def wallGetComments(access_token=None,
                    count=None,
                    fields=None,
                    offset=None,
                    owner_id=None,
                    params=None,
                    post_id=None,
                    returnDfs=False,
                    silentMode=False,
                    version_api='5.199'):
    '''
    Функция для выгрузки комментариев к постам ВК методом его API wall.getComments . Причём количество выгружаемых постов максимизируется посредством offset

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/wall.getComments , за исключением аргументов params , returnDfs и silentMode
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params , который обычно подаётся в метод get пакета requests
    access_token : str
           count : int
          fields : list
          offset : int
        owner_id : str
          params : dict -- в случае наличия готового словаря с аргументами метода https://dev.vk.com/ru/method/wall.getComments , чтобы не подавать эти аргументы по отдельности
         post_id : str
       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными
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
    method = 'wall.getComments'
    params_extra = {}
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС

    momentCurrent = datetime.now() # запрос текущего момента
    if not silentMode:
        print('\nТекущий момент:', momentCurrent.strftime('%Y%m%d_%H%M'), '-- он будет использован для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    nameS_in_folderCurrent = os.listdir()
    # print('nameS_in_folderCurrent:', nameS_in_folderCurrent) # для отладки
    api_keyS = scrapingTools.credentialsProcessor(access_token, 'VK', folder, 'https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8', nameS_in_folderCurrent, silentMode)

# Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, прерванном пользователем или ошибкой
    # Поиск данных
    if not silentMode: # если основная функция этого модуля явлется частью другой функции, поиск данных не требуется -- они будут переданы
        print('Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
        for name_in_folderCurrent in nameS_in_folderCurrent:
            if name_in_folderCurrent.endswith('Temporal') and 'VK' in name_in_folderCurrent:
                folder_in_folderCurrent = folder + name_in_folderCurrent # имя файла достраивается до абсолютного пути
                if len(os.listdir(folder_in_folderCurrent)) < 6: print(
f"В директории '{folder_in_folderCurrent}' есть директория, похожая на искомую Temporal , но в ней не 6-7 файлов => если это она, то либо повреждена, либо создалась при безрезультатном запуске; тогда Вам следует удалить её вручную") # допустимо 6 или 7 файлов

                else:
                    with open(f'{folder_in_folderCurrent}{slash}fields.json', 'r', encoding='utf-8') as file:
                        fieldsTemporal = json.load(file)

                    offsetTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'offset.txt', int)
                    owner_idTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'owner_id.txt', str)

                    with open(f'{folder_in_folderCurrent}{slash}params_extra.json', 'r', encoding='utf-8') as file:
                        params_extraTemporal = json.load(file)

                    post_idTemporal = scrapingTools.containerImport(folder_in_folderCurrent + slash + 'post_id.txt', str)
                    
                    print(f"Нашёл директорию '{folder_in_folderCurrent}'. В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:"
                          , '\n- скрипт остановился на offset', offsetTemporal)

                    print(f"- пользователь определил поля: '{fieldsTemporal}'" if fieldsTemporal else '- пользователь НЕ определил поля')
                    print(f"- пользователь определил ID страницы: '{owner_id}'" if owner_idTemporal else '- пользователь НЕ определил ID страницы')
                    print(f"- пользователь определил ID поста: '{post_id}'" if post_idTemporal else '- пользователь НЕ определил ID поста')
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
                        fields = fieldsTemporal
                        offset = offsetTemporal
                        owner_id = owner_idTemporal
                        params_extra = params_extraTemporal
                        post_id = post_idTemporal

                        temporalNameS = os.listdir(folder_in_folderCurrent)
                        folderWithTemporal, itemS = scrapingTools.get_itemsTemporal(temporalNameS)

        if not folderWithTemporal: # если itemsTemporal, в т.ч. пустой, не существует и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
            print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта')

# Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    # if not silentMode продолжается
        folder, folder_file, itemS = scrapingTools.call_for_user_items(f'в котором есть ранее выгруженные из ВК методом {method} данные')
            # теперь определены объекты: folder и folder_file (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский)

    if not params and not access_token and not count and not fields and not offset and not owner_id and not post_id and not returnDfs:
        # print('Пользователь не подал аргументы') # для отладки
        experiencedMode = False
        
    else:
        # print('Пользователь подал аргументы') # для отладки
        experiencedMode = True
        if params:
            access_token = scrapingTools.argument_key_comparison(access_token, 'access_token', params)
            # print('access_token:', access_token) # для отладки

            count = scrapingTools.argument_key_comparison(count, 'count', params)
            # print('count:', count) # для отладки

            fields = scrapingTools.argument_key_comparison(fields, 'fields', params)
            # print('fields:', fields) # для отладки

            offset = scrapingTools.argument_key_comparison(offset, 'offset', params)
            # print('offset:', offset) # для отладки

            owner_id = scrapingTools.argument_key_comparison(owner_id, 'owner_id', params)
            # print('owner_id:', owner_id) # для отладки

            post_id = scrapingTools.argument_key_comparison(post_id, 'post_id', params)
            # print('post_id:', post_id) # для отладки

            if not params_extra: # если params_extra НЕ извлечены из Temporal, то попробовать извлечь их из params
                params_extra = {key: value for key, value in params.items() if key not in KEYS_BASIC}
                    # всё, что осталось -- пользовательские аргументы метода
    
                params_extra = {key: value for key, value in params_extra.items() if value is not None}

        if count:
            if type(count) != int: count = int(count)

        else: count = 100

        # print('count:', count) # для отладки

        if offset:
            if type(offset) != int: offset = int(offset)
        # print('offset:', offset) # для отладки
            
        if owner_id:
            if type(owner_id) != str: owner_id = str(owner_id)

        else: owner_id = '80054288' # моя страница

        # print('owner_id:', owner_id) # для отладки

        if post_id:
            if type(post_id) != str: post_id = str(post_id)

        else: post_id = '1' # первый пост на моей странице

        # print('post_id:', post_id) # для отладки

# На выходе аргументы + params_extra . Приоритет определён при первой подаче аргументов и словаря params, этот приоритет транслируется через в Temporal, следовательно, при извлечении из Temporal перепроверять приоритет не требуется

    if not silentMode:
        if not experiencedMode: print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.''')

        print(
f'''    Скрипт нацелен на выгрузку комментариев к постам ВК методом его API {method} . Причём количество выгружаемых постов максимизируется посредством offset .
Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК''')

        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0.2 Пользовательские настройки запроса к API ВК
    if not silentMode: # если основная функция этого модуля явлется частью другой функции, поиск данных не требуется -- они будут переданы
        if not owner_id or not post_id: # если пользователь не подал эти аргументы в рамках experiencedMode
            if not owner_id:
                print(
'''Скрипт умеет искать комментарии к постам открытых страниц
--- Введите ID интересующей страницы (персональной и группы), после чего нажмите Enter''')

                owner_id = input()
                if owner_id == '': owner_id = None # для единообразия
                else: print('')

            if not post_id:
                print(
'''Скрипт умеет искать комментарии к постам открытых страниц
--- Введите ID интересующего поста, после чего нажмите Enter''')

                post_id = input()
                if post_id == '': post_id = None # для единообразия
                else: print('')

                if folder_file: print(
'ВАЖНО! В результате исполнения его скрипта данные из указанного Вами файла', folder_file, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести те же ID, что и при формировании указанного Вами файла')

# Сложная часть имени будущих директорий и файлов
        complicatedNamePart = '_VK'
        if owner_id: complicatedNamePart += '_' + owner_id if len(owner_id) < 50 else '_' + owner_id[:50]
        if post_id: complicatedNamePart += '_' + post_id if len(post_id) < 50 else '_' + post_id[:50]
    
    # Завершение if not silentMode

# 2.1 Первичный сбор контента методом wall.getComments
# 2.1.0 Первое обращение к API
    iteration = 1 # номер итерации применения текущего метода
    pause = 0.25
    if not silentMode: print(f'В этой части функции используются следующие аргументы метода {method} API ВК: count, fields, offset, owner_id, post_id и другие аргументы, которые пользователь может подать через params')
        # print('experiencedMode:', experiencedMode) # для отладки
        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа

    if offset is None: offset = 0 # если offset ни подан пользователем, ни сохранён при прошлом запуске
    try: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        while True:
            itemsAdditional, goS, iteration, keyOrder, pause, response = wallGetCommentsCore(api_keyS,
                                                                                             count,
                                                                                             fields,
                                                                                             iteration,
                                                                                             keyOrder,
                                                                                             offset,
                                                                                             owner_id,
                                                                                             params_extra,
                                                                                             pause,
                                                                                             post_id,
                                                                                             version_api)

            # После самого первого исполнения wallGetCore iteration == 2
            # print('goS:', goS) # для отладки
            if not goS or goS and len(itemS_additional) > 0: # сигнал остановиться и число выгруженных постов не важно; ИЛИ сигнала остановиться нет, а выгруженные посты есть
                # в первом случае dfsProcessor завешит скрипт и создаст Temporal-снимок, во втором -- объединит свежевыгруженные посты с ранее выгруженными
                # Вариантов сигнала в этом и аналогичном скриптах три: len(itemS_additional) > 0 , not goS , 'next_from' is not in response.keys()

                itemS = dfsProcessor(complicatedNamePart,
                                     coLabFolder,
                                     itemsAdditional,
                                     itemS, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода getComments
                                     itemS,
                                     fields,
                                     fileFormatChoice,
                                     goS, # единственная из функций, принимающая этот аргумент
                                     method,
                                     momentCurrent,
                                     offset,
                                     owner_id,
                                     params_extra,
                                     post_id,
                                     slash)
                offset += count
                time.sleep(pause)

            if goS and len(itemS_additional) == 0: # сигнала остановиться нет, но и выгруженных постов нет
                if not silentMode: print(
f'''{'Дополнительных' if iteration > 2 else 'Искомых'} комментариев к посту на странице из Вашего запроса, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта{'' if iteration > 2 else '. Что можно сделать для выгрузки комментариев? Поменяйте настройки запроса и запустите скрипт с начала'}''')
                if iteration >= 2:
# 2.1.2 Экспорт выгрузки метода wall.getComments и финальное завершение скрипта
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
                            if not silentMode: print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473''')

                            sys.exit()

    except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        # display(itemS)
        if not itemS: itemS = pandas.DataFrame()
        dfsProcessor(complicatedNamePart,
                     coLabFolder,
                     df_additional,
                     itemS, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода getComments
                     itemS,
                     fields,
                     fileFormatChoice,
                     goS, # единственная из функций, принимающая этот аргумент
                     method,
                     momentCurrent,
                     offset,
                     owner_id,
                     params_extra,
                     post_id,
                     slash)

        if returnDfs: return itemS
