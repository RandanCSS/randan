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

        import json, os, pandas, shutil, requests, time, warnings
        break # выход из цикла while True

    except ModuleNotFoundError as excptn_1:
        module = excptn_1.name.split('.')[0]
        if attempt > MAX_ATTEMPTS:
            print(
f'Пакет {module} НЕ удалось импортировать за {MAX_ATTEMPTS} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт'
            )

            raise

        print(
f'Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас. Попытка № {attempt} из {MAX_ATTEMPTS}'
        )

        try: check_call([sys.executable, '-m', 'pip', 'install', module, '--quiet', '--disable-pip-version-check'])
        except CalledProcessError as excptn_2:
            print(f"Не удалось установить {module}. {type(excptn_2).__name__}: {str(excptn_2).split('Stacktrace:')[0].strip()}")
            raise

        attempt += 1

# 1. Вспомогательные функции для..
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
    if df_in is None: df_in = pandas.DataFrame()
    df = pandas.concat([df_in, df_additional])
    columnsForCheck = []

    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if not folder: folder = ''
    # if folder is None) | (folder == ''): folder = ''
    else: folder += slash

    for column in df.columns: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        if column == 'id' or column.endswith('_id') or column.startswith('id_'): columnsForCheck.append(column)

    # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck) # для отладки

    if columnsForCheck: df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True)
        # при дублировании записей из itemS из Temporal и от пользователя и новых записей, оставить новые

    if not goS:
        print(
f'''Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"'''
        )

        temporal_exportPath = folder + f"{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"
        if not os.path.exists(temporal_exportPath):
            os.makedirs(temporal_exportPath)
            print(f'''Директория "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal" создана''')
        # else:
            # print(f'''Директория "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal" существует''')

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
        scrapingTools.containerExport(temporal_exportPath + slash + 'domain.txt', domain if domain else '')

        if not fields: fields = []
        with open(temporal_exportPath + slash + 'fields.json', 'w', encoding='utf-8') as file:
            json.dump(fields, file, ensure_ascii=False, indent=4)

        scrapingTools.containerExport(temporal_exportPath + slash + 'filter.txt', filter if filter else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'method.txt', method)
        scrapingTools.containerExport(temporal_exportPath + slash + 'offset.txt', offset)

        df2file.df2fileShell(complicatedNamePart=f'{complicatedNamePart}_Temporal',
                             currentMoment=momentCurrent.strftime('%Y%m%d'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             df_in=df,
                             fileFormatChoice=fileFormatChoice,
                             folder=temporal_exportPath,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method)
                                 # чтобы избавиться от лишней точки в имени файла  

        print('Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')

    return df

def wallGetCore(api_keyS,
                count,
                domain,
                fields,
                filter,
                iteration,
                keyOrder,
                offset,
                pause):

    df_additional = pandas.DataFrame()
    goS = True

    params_initial = {'access_token': api_keyS[keyOrder], # обязательный параметр
                      'count': count, # опциональный параметр
                      'domain': domain, # обязательный параметр, но без него не будет результата
                      'extended': 1, # опциональный параметр
                      'fields': fields, # опциональный параметр
                      'filter': filter, # опциональный параметр
                      'offset': offset, # опциональный параметр
                      'v': '5.199'} # обязательный параметр

    params = {key: value for key, value in params_initial.items() if value not in (None, '', [])}
    goC = True
    tryer = 0
    while goC:
        try: # чтобы обработать сигнал прерывания, поданный на любом этапе сбора данных
            response = requests.get('https://api.vk.ru/method/wall.get', params=params).json()
            # print('response', response) # для отладки
            if 'response' in response.keys():
                response = response['response']
                # print('    response.keys() внутри wallGetCore', response.keys()) # для отладки

                df_additional = pandas.json_normalize(response['items'])
                break # нет смысла в новых итерациях цикла while goC

            else: goC, goS, keyOrder, pause, response, tryer = scrapingVK_tools.errorProcessor(api_keyS, keyOrder, pause, response, tryer)

        except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
            response = {'items': [], 'total_count': 0} # принудительная выдача для response
            goS = False # нет смысла продолжать исполнение скрипта
            # print('goS wallGetCore:', goS) # для отладки

            break # и, следовательно, нет смысла в новых итерациях цикла while goC

    if goS:
        # Для визуализации процесса
        print('    Итерация №', iteration, ', number of items', len(response['items']), '                                        ', end='\r')

        iteration += 1
        if len(df_additional) > 0: df_additional = scrapingVK_tools.dfColumnsProcessor(df_additional, fields, response)

    return df_additional, goS, iteration, keyOrder, pause

# 2. Основная функция
def wallGet(access_token=None,
            count=None,
            domain=None,
            fields=None,
            filter=None,
            offset=None,
            params=None,
            returnDfs=False):

    method = 'wall.get'

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
          offset : int
          params : dict -- в случае наличия готового словаря с аргументами метода https://dev.vk.com/ru/method/wall.get ,
          чтобы не подавать эти аргументы по отдельности

       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными
    '''
    if not params and not access_token and not count and not domain and not fields and not filter and not offset and not returnDfs:
        print('Пользователь не подал аргументы') # для отладки
        experiencedMode = False

    else:
        experiencedMode = True
        if params:
            if access_token: access_token = scrapingTools.argument_key_comparison(access_token, 'access_token', params)
            # print('access_token:', access_token) # для отладки

            if count: count = scrapingTools.argument_key_comparison(count, 'count', params)
            # print('count:', count) # для отладки

            if domain: domain = scrapingTools.argument_key_comparison(domain, 'domain', params)
            # print('domain:', domain) # для отладки

            if fields: fields = scrapingTools.argument_key_comparison(fields, 'fields', params)
            # print('fields:', fields) # для отладки

            if filter: filter = scrapingTools.argument_key_comparison(filter, 'filter', params)
            # print('filter:', filter) # для отладки

            if offset is not None: offset = scrapingTools.argument_key_comparison(offset, 'offset', params)
            # print('offset:', offset) # для отладки

        if count:
            if not isinstance(count, int): count = min(abs(int(count)), 100) # VK API wall.get принимает count не более 100. Если пользователь подаст больше, offset += count начнёт перепрыгивать посты

        else: count = 100
        # print('count:', count) # для отладки

        if domain:
            if not isinstance(domain, str): domain = str(domain)
        # print('domain:', domain) # для отладки

        if offset is not None:
            if not isinstance(offset, int): offset = int(offset)
        # print('offset:', offset) # для отладки
            
    if not experiencedMode:
        print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.'''
        )

    print(
f'''    Скрипт нацелен на выгрузку характеристик контента ВК методом его API {method} . Причём количество объектов выгрузки максимизируется посредством offset .
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК'''
    )

    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API ВК

    # Блок, поскольку folder многократно используется внутри функции в формулах
    coLabFolder = coLabAdaptor.coLabAdaptor() # либо '/content/drive/MyDrive/Colab Notebooks' , либо None
    folder = coLabFolder
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if not folder: folder = ''
    # if folder is None) | (folder == ''): folder = ''
    else: folder += slash

    fileFormatChoice = '.xlsx' # базовый формат сохраняемых файлов; формат .json добавляется опционально через наличие columnsToJSON
    folderFile = None
    goS = True
    itemS = pandas.DataFrame()
    itemS_additional = None # чтобы обработать сигнал прерывания, поданный на любом этапе сбора данных
    keyOrder = 0
    temporalName = None

    momentCurrent = datetime.now() # запрос текущего момента
    print('\nТекущий момент:', momentCurrent.strftime('%Y%m%d_%H%M'), '-- он будет использован для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    nameS_in_folderCurrent = os.listdir()
    api_keyS = scrapingTools.credentialsProcessor(access_token, 'VK', 'https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8', nameS_in_folderCurrent)
# Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
    # Поиск данных
    print('Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
    for name_in_folderCurrent in nameS_in_folderCurrent:
        message =\
f"В текущей директории '{name_in_folderCurrent}' есть директория, похожая на искомую Temporal , но в ней не 6-7 файлов => если это она, то либо повреждена, либо создалась при безрезультатном запуске; тогда Вам следует удалить её вручную"

        if name_in_folderCurrent.endswith('Temporal') and 'VK' in name_in_folderCurrent:
            if len(os.listdir(name_in_folderCurrent)) < 5: print(message) # допустимо 6 или 7 файлов
            else:
                domain = scrapingTools.containerImport(name_in_folderCurrent + slash + 'domain.txt', str)

                with open(f'{name_in_folderCurrent}{slash}fields.json', 'r', encoding='utf-8') as file:
                    fields = json.load(file)

                filter = scrapingTools.containerImport(name_in_folderCurrent + slash + 'filter.txt', str)
                offset = scrapingTools.containerImport(name_in_folderCurrent + slash + 'offset.txt', int)

                print(f"Нашёл директорию '{name_in_folderCurrent}'. В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:"
                      , '\n- скрипт остановился на offset', offset)
                print('- пользователь НЕ определил страницу' if not domain else f"- пользователь определил страницу: '{domain}'")
                print('- пользователь НЕ определил поля' if not fields else f"- пользователь определил поля: '{fields}'")
                print('- пользователь НЕ определил фильтр' if not filter else f"- пользователь определил фильтр: '{filter}'")

                folderWithTemporal = name_in_folderCurrent
                print(
'''--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter
--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите 'R' и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter'''
                )

                decision = input()
                if len(decision) == 0:
                    temporalNameS = os.listdir(name_in_folderCurrent)
                    itemS_json = None
                    itemS_xlsx = None
                    for temporalName in temporalNameS:
                        if temporalName.endswith('.json') and 'VK' in temporalName: itemS_json = pandas.read_json(f'{name_in_folderCurrent}{slash}{temporalName}')
                        if temporalName.endswith('.xlsx') and 'VK' in temporalName: itemS_xlsx = pandas.read_excel(f'{name_in_folderCurrent}{slash}{temporalName}', index_col=0)
                        if itemS_json is not None and itemS_xlsx is not None:
                            itemS = itemS_json.merge(itemS_xlsx, how='outer', on='id')
# Данные, сохранённые при прошлом запуске скрипта, загружены
                            break # выход из for temporalName in temporalNameS

                    if itemS_xlsx is not None and itemS_json is None: itemS = itemS_xlsx.copy() # for temporalName in temporalNameS завершился, но файл itemS_json не импортировался
                    elif itemS_xlsx is None and itemS_json is None:
                        temporalName = None # флаг, что данные, сохранённые при прошлом запуске скрипта, не найдены
                        print(message)

                elif decision == 'R': shutil.rmtree(folderWithTemporal, ignore_errors=True)
            
# Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if not temporalName: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку

        name_in_folderCurrent = 'No folder'
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку')
        print(
'''
Возможно, Вы располагаете файлом, в котором есть ранее выгруженные из ВК методом wall.get данные, и который хотели бы дополнить?
Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом формата XLSX, укажите полный путь, включая название файла, и нажмите Enter.
Затем при необходимости сможете добавить к нему другие располагаемые файлы'''
        )

        while True:
            folderFile = input()
            if len(folderFile) == 0:
                folderFile = None # для унификации
                break

            else:
                itemS, error, folder = files2df.files2df(folderFile)
                if error:
                    if 'No such file or directory' in error:
                        print('Путь:', folderFile, '-- не существует; попробуйте, пожалуйста, ещё раз..')

                else: break
            # display('itemS:', itemS)
# Теперь определены объекты: folder и folderFile (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский), slash

# 2.0.2 Пользовательские настройки запроса к API ВК
        if not domain: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать посты открытых страниц
--- Введите название интересующей страницы (персональной и группы), после чего нажмите Enter'''
            )

            if folderFile:
                print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folderFile, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести название той же страницы, что и при формировании указанного Вами файла'
                )

            domain = input()
            if not domain: domain = '80054288' # моя страница
            else: print('')

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    if domain: complicatedNamePart += '_' + domain if len(domain) < 50 else '_' + domain[:50]

# 2.1 Первичный сбор контента методом get
# 2.1.0 Первое обращение к API
    itemS_additional = pandas.DataFrame() # на случай, если сигнал прерывания поступит до dfsProcessor
    iteration = 1 # номер итерации применения текущего метода
    method = 'wall.get'
    pause = 0.25

    print(
f'В скрипте используются следующие аргументы метода {method} API ВК: count, domain, fields, filter, offset .',
'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
f'Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/{method} ,',
f'-- можете подать их в скобки функции wallGet перед её запуском или скопировать код исполняемого сейчас скрипта и сделать это внутри кода внутри метода {method} в разделе 2'
    )

    # print('experiencedMode:', experiencedMode) # для отладки
    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
    print('') # для отступа

    if not offset: offset = 0 # если offset ни подан пользователем, ни сохранён при прошлом запуске
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
                                                                           pause)

            # print('goS:', goS) # для отладки
            if len(itemS_additional) == 0: #ничего нового не выгружено независимо от сигнала остановиться
                print('Ни один дополнительный пост НЕ выгружен')
                break

            elif goS and len(itemS_additional) > 0: # нет сигнала остановиться и новые записи выгружены
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

            elif not goS and len(itemS_additional) > 0: # есть сигнал остановиться, при этом новые записи выгружены
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
                break

# 2.1.2 Экспорт выгрузки метода get и финальное завершение скрипта
        df2file.df2fileShell(complicatedNamePart=complicatedNamePart,
                             currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             df_in=itemS,
                             fileFormatChoice=fileFormatChoice,
                             folder=folder,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method)
                                # чтобы избавиться от лишней точки в имени файла

        print('Скрипт исполнен. Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
        if folderWithTemporal:
            if os.path.exists(folderWithTemporal):
                # print('folderWithTemporal:', folderWithTemporal) # для отладки
                print(
'''Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, можно удалить её во избежание путаницы при следующих запусках скрипта
--- Если хотите их удалить, введите 'R' и нажмите Enter'''
                )

                decision = input()
                if decision == 'R': shutil.rmtree(folderWithTemporal, ignore_errors=True)

        if fields: print(
f'''
Чтобы распаковать JSON из любого столбца, содержащего этот формат, в отдельный датафрейм, используйте такой код:
import pandas
column = 'Имя_столбца'
JSONS = []
for cellContent in Исходный_датафрейм[column].dropna():
    JSONS.extend(cellContent)
Новый_датафрейм = pandas.json_normalize(JSONS).drop_duplicates('id').reset_index(drop=True)
'''
        )

        if returnDfs: return itemS

    except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        # display(itemS)
        if itemS is None or itemS.empty: itemS = pandas.DataFrame()

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

        if returnDfs: return itemS
