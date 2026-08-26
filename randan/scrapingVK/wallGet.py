#!/usr/bin/env python
# coding: utf-8

"""
(EN) A module that simplifies and maximizes VK content extraction using the platform's official API method wall.get
(RU) Модуль для упрощения выгрузки контента ВК методом его API wall.get и максимизации размера этой выгрузки
"""

# 0. Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# 0.0 В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
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
        break # выход из цикла for attempt in range(3)

    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", "").replace("'", "") #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print(
f"""Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 3
"""
              )
        check_call([sys.executable, "-m", "pip", "install", module])
        if  attempt == 3:
            print(
f"""Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
"""
                  )

# 1. Вспомогательные функции для..
# .. метода get из API ВК, помогающая работе с ключами
# .. обработки выдачи, помогающая работе с ключами
def dfsProcessor(complicatedNamePart,
                 coLabFolder,
                 dfAdd,
                 dfFinal, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода get
                 dfIn,
                 domain,
                 fields,
                 fileFormatChoice,
                 filter,
                 goS, # единственная из функций, принимающая этот аргумент
                 method,
                 momentCurrent,
                 offset,
                 slash):
    df = pandas.concat([dfIn, dfAdd])
    columnsForCheck = []
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id' in column: columnsForCheck.append(column)

    # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck) # для отладки

    df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True)
        # при дублировании записей из itemS из Temporal и от пользователя и новых записей, оставить новые

    if goS == False:
        print(
f'Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal"'
              )
        if not os.path.exists(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal'):
            os.makedirs(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal')
            print(f'Директория "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal" создана')
        # else:
            # print(f'Директория "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal" существует')

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
        if not domain: domain = ''
        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}domain.txt', 'w+') # открыть на запись
        file.write(domain if domain else '')
        file.close()

        if not fields: fields = []
        with open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}fields.txt', 'w', encoding='utf-8') as file:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)

        if not filter: filter = ''
        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}filter.txt', 'w+') # открыть на запись
        file.write(filter if filter else '')
        file.close()

        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}method.txt', 'w+') # открыть на запись
        file.write(method)
        file.close()

        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}offset.txt', 'w+')
        file.write(str(offset)) # год, на котором остановилось исполнение скрипта
        file.close()

        df2file.df2fileShell(complicatedNamePart=f'{complicatedNamePart}_Temporal',
                             dfIn=df,
                             fileFormatChoice=fileFormatChoice,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method,
                                 # чтобы избавиться от лишней точки в имени файла

                             coLabFolder=coLabFolder,
                             currentMoment=momentCurrent.strftime("%Y%m%d"))
                                 # .strftime -- чтобы варьировать для итоговой директории и директории Temporal

        warnings.filterwarnings("ignore")
        print(
'Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть',
'Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'
              )
        sys.exit()

    return df

def wallGetCore(API_keyS,
                count,
                domain,
                fields,
                filter,
                iteration,
                keyOrder,
                offset,
                pause):
    dfAdd = pandas.DataFrame()
    goS = True

    params = {'access_token': API_keyS[keyOrder], # обязательный параметр
              'count': count, # опциональный параметр
              'domain': domain, # обязательный параметр, но без него не будет результата
              'extended': 1, # опциональный параметр
              'fields': fields, # опциональный параметр
              'filter': filter, # опциональный параметр
              'offset': offset, # опциональный параметр
              'v': '5.199'} # обязательный параметр

    goC = True
    tryer = 0
    while goC:
        try: # чтобы обработать сигнал прерывания, поданный на любом этапе сбора данных
            response = requests.get('https://api.vk.ru/method/wall.get', params=params)
            response = response.json() # отобразить выдачу метода get в виде JSON
            # print('response', response) # для отладки
            if 'response' in response.keys():
                response = response['response']
                # print('    response.keys() внутри wallGetCore', response.keys()) # для отладки

                dfAdd = pandas.json_normalize(response['items'])
                break # нет смысла в новых итерациях цикла while goC

            else: goC, goS, keyOrder, pause, response, tryer = scrapingVK_tools.errorProcessor(API_keyS, keyOrder, pause, response, tryer)

        except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
            response = {'items': [], 'total_count': 0} # принудительная выдача для response
            goS = False # нет смысла продолжать исполнение скрипта
            # print('goS wallGetCore:', goS) # для отладки

            break # и, следовательно, нет смысла в новых итерациях цикла while goC

    if goS:
        # Для визуализации процесса
        print('    Итерация №', iteration, ', number of items', len(response['items']), '                                        ', end='\r')

        iteration += 1
        if len(dfAdd) > 0: dfAdd = scrapingVK_tools.dfColumnsProcessor(dfAdd, fields, response)

    return dfAdd, goS, iteration, keyOrder, pause, response

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

    f"""
    Функция для выгрузки характеристик контента ВК методом его API {method} . Причём количество объектов выгрузки максимизируется посредством offset

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/{method} , за исключением аргументов params и returnDfs
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params ,
    который обычно подаётся в метод get пакета requests
    access_token : str
           count : int
          domain : str
          fields : list
          filter : str
          offset : int
          params : dict -- в случае наличия готового словаря с аргументами метода https://dev.vk.com/ru/method/{method} ,
          чтобы не подавать эти аргументы по отдельности

       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными
    """
    if not params and not access_token and not count and not domain and not fields and not filter and not offset and not returnDfs:
        # print('Пользователь не подал аргументы') # для отладки

        expiriencedMode = False
        count = 100

    else:
        expiriencedMode = True
        if params:
            access_token = scrapingTools.argument_key_comparison(access_token, 'access_token', params)
            # print('access_token:', access_token) # для отладки

            count = scrapingTools.argument_key_comparison(count, 'count', params)
            if count:
                if type(count) != int: count = int(count)
            # print('count:', count) # для отладки

            domain = scrapingTools.argument_key_comparison(domain, 'domain', params)
            # print('domain:', domain) # для отладки

            fields = scrapingTools.argument_key_comparison(fields, 'fields', params)
            # print('fields:', fields) # для отладки

            filter = scrapingTools.argument_key_comparison(filter, 'filter', params)
            # print('filter:', filter) # для отладки

            offset = scrapingTools.argument_key_comparison(offset, 'offset', params)
            if offset:
                if type(offset) != int: offset = int(offset)
            # print('offset:', offset) # для отладки

    if expiriencedMode == False:
        print(
"""    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом."""
              )
    print(
f"""    Скрипт нацелен на выгрузку характеристик контента ВК методом его API {method} . Причём количество объектов выгрузки максимизируется посредством offset .
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК"""
          )
    if not expiriencedMode: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API ВК

    # Блок, поскольку folder многократно используется внутри функции в формулах
    coLabFolder = coLabAdaptor.coLabAdaptor()
    folder = coLabFolder
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if (folder == None) | (folder == ''): folder = ''
    else: folder += slash

    fileFormatChoice = '.xlsx' # базовый формат сохраняемых файлов; формат .json добавляется опционально через наличие columnsToJSON
    folderFile = None
    goS = True
    itemS = pandas.DataFrame()
    itemsAdditional = None # чтобы обработать сигнал прерывания, поданный на любом этапе сбора данных
    keyOrder = 0
    temporalName = None

    momentCurrent = datetime.now() # запрос текущего момента
    print('\nТекущий момент:', momentCurrent.strftime("%Y%m%d_%H%M"), '-- он будет использована для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    rootNameS = os.listdir()
    # Поиск ключей
    if access_token == None:
        print('Проверяю наличие файла credentialsVK.txt с ключ[ом ами], гипотетически сохранённым[и] при первом запуске скрипта')
        if 'credentialsVK.txt' in rootNameS:
            file = open('credentialsVK.txt')
            API_keyS = file.read()
            print('Нашёл файл credentialsVK.txt; далее буду использовать ключ[и] из него:', API_keyS)
        else:
            print(
"""--- НЕ нашёл файл credentialsVK.txt . Введите в окно Ваш API key для авторизации в API ВК 
(примерная инструкция, как создать API key, доступна по ссылке https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8 ). Для подстраховки от ограничения действия API key желательно создать несколько ключей (три -- отлично) и ввести их без кавычек через запятую с пробелом
--- После ввода нажмите Enter"""
                  )
            while True:
                API_keyS = input()
                if len(API_keyS) != 0:
                    print(f"-- далее буд{'у' if len(API_keyS) > 1 else 'е'}т использован{'ы' if len(API_keyS) > 1 else ''} эт{'и' if len(API_keyS) > 1 else 'и'} ключ{'и' if len(API_keyS) > 1 else ''}")

                    from randan.tools.textPreprocessor import multispaceCleaner # авторский модуль для предобработки нестандартизированного текста
                    API_keyS = multispaceCleaner(API_keyS)
                    while API_keyS[-1] == ',': API_keyS = API_keyS[:-1] # избавиться от запятых в конце текста

                    file = open("credentialsVK.txt", "w+") # открыть на запись
                    file.write(API_keyS)
                    file.close()
                    break
                else:
                    print('--- Вы ничего НЕ ввели. Попробуйте ещё раз..')
        API_keyS = API_keyS.replace(' ', '') # контроль пробелов
        API_keyS = API_keyS.replace(',', ', ') # контроль пробелов
        API_keyS = API_keyS.split(', ')
    else: API_keyS = [access_token]
    print('Количество ключей:', len(API_keyS), '\n')

# 2.0.2 Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
    # Поиск данных
    print('Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
    for rootName in rootNameS:
        if 'Temporal' in rootName:
            if len(os.listdir(rootName)) == 4:
                file = open(f'{rootName}{slash}domain.txt')
                domain = file.read()
                file.close()

                with open(f'{rootName}{slash}fields.json', 'r', encoding='utf-8') as file:
                    fields = json.load(file)

                file = open(f'{rootName}{slash}filter.txt')
                filter = file.read()
                file.close()

                file = open(f'{rootName}{slash}offset.txt')
                offset = file.read()
                file.close()
                offset = int(offset)

                print(f'Нашёл директорию "{rootName}". В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:'
                      , '\n- скрипт остановился на offset', offset)
                print('- пользователь НЕ определил страницу' if not domain else f"- пользователь определил страницу: '{domain}'")
                print('- пользователь НЕ определил поля' if not fields else f"- пользователь определил поля: '{fields}'")
                print('- пользователь НЕ определил фильтр' if not filter else f"- пользователь определил фильтр: '{filter}'")
                print(
"""--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter
--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите "R" и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter"""
                      )
                decision = input()
                if len(decision) == 0:
                    temporalNameS = os.listdir(rootName)
                    for temporalName in temporalNameS:
                        if '.xlsx' in temporalName: break
                    itemS = pandas.read_excel(f'{rootName}{slash}{temporalName}', index_col=0)

                    for temporalName in temporalNameS:
                        if '.json' in temporalName:
                            itemS = itemS.merge(pandas.read_json(f'{rootName}{slash}{temporalName}'), on='id', how='outer')
                            break # выход из for temporalName in temporalNameS

# Данные, сохранённые при прошлом запуске скрипта, загружены
                    break # выход из for rootName in rootNameS
                elif decision == 'R': shutil.rmtree(rootName, ignore_errors=True)
            else: shutil.rmtree(rootName, ignore_errors=True) # в директории Temporal не 7 файлов => либо она повреждена, либо создалась при безрезультатном запуске

# 2.0.3 Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if temporalName == None: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        rootName = 'No folder'
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку')
        print(
"""
Возможно, Вы располагаете файлом, в котором есть ранее выгруженные из ВК методом wall.get данные, и который хотели бы дополнить?
Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом формата XLSX, укажите полный путь, включая название файла, и нажмите Enter.
Затем при необходимости сможете добавить к нему другие располагаемые файлы"""
              )
        while True:
            folderFile = input()
            if len(folderFile) == 0:
                folderFile = None # для унификации
                break
            else:
                itemS, error, folder = files2df.files2df(folderFile)
                if error != None:
                    if 'No such file or directory' in error:
                        print('Путь:', folderFile, '-- не существует; попробуйте, пожалуйста, ещё раз..')
                else: break
            # display(itemS)
# Теперь определены объекты: folder и folderFile (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский), slash

# 2.0.4 Пользовательские настройки запроса к API ВК
        if not domain: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
"""Скрипт умеет искать посты открытых страниц
--- Введите название интересующей страницы (персональной и группы), после чего нажмите Enter"""
                  )

            if folderFile:
                print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folderFile, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести название той же страницы, что и при формировании указанного Вами файла'
                      )
            domain = input()
            if domain == '': domain = None # для единообразия
            else: print('')

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    if domain: complicatedNamePart += "_" + domain if len(domain) < 50 else "_" + domain[:50]

# 2.1 Первичный сбор контента методом get
# 2.1.0 Первое обращение к API
    try: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        method = 'wall.get'
        iteration = 0 # номер итерации применения текущего метода
        pause = 0.25

        print(
f'В скрипте используются следующие аргументы метода {method} API ВК: count, domain, fields, filter, offset .',
'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
f'Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/{method} ,',
f'-- можете подать их в скобки функции wallGet перед её запуском или скопировать код исполняемого сейчас скрипта и сделать это внутри кода внутри метода {method} в разделе 2'
              )

        # print('expiriencedMode:', expiriencedMode) # для отладки
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа

        iteration = 1
        if not offset: offset = 0 # если offset ни подан пользователем, ни сохранён при прошлом запуске
        while True:
            itemsAdditional, goS, iteration, keyOrder, pause, response = wallGetCore(API_keyS,
                                                                                     count,
                                                                                     domain,
                                                                                     fields,
                                                                                     filter,
                                                                                     iteration,
                                                                                     keyOrder,
                                                                                     offset,
                                                                                     pause)

            # print('goS:', goS) # для отладки
            if goS & (len(itemsAdditional) == 0):
                print('Все посты страницы выгружены')
                break

            else:
                itemS = dfsProcessor(complicatedNamePart,
                                     coLabFolder,
                                     itemsAdditional,
                                     itemS, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода get
                                     itemS,
                                     domain,
                                     fields,
                                     fileFormatChoice,
                                     filter,
                                     goS, # единственная из функций, принимающая этот аргумент
                                     method,
                                     momentCurrent,
                                     offset,
                                     slash)

                offset += count
                time.sleep(pause)

# 2.1.2 Экспорт выгрузки метода get и финальное завершение скрипта
        df2file.df2fileShell(
                             complicatedNamePart=complicatedNamePart,
                             dfIn=itemS,
                             fileFormatChoice=fileFormatChoice,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method,
                                # чтобы избавиться от лишней точки в имени файла

                             coLabFolder=coLabFolder,
                             currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             )

        print('Скрипт исполнен. Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
        if os.path.exists(rootName):
            print('rootName:', rootName)
            print(
    'Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта'
                  )
            shutil.rmtree(rootName, ignore_errors=True)
        if fields != None: print(
f"""
Чтобы распаковать JSON из любого столбца, содержащего этот формат, в отдельный датафрейм, используйте такой код:
import pandas
column = 'Имя_столбца'
JSONS = []
for cellContent in Исходный_датафрейм[column].dropna():
    JSONS.extend(cellContent)
Новый_датафрейм = pandas.json_normalize(JSONS).drop_duplicates('id').reset_index(drop=True)

Чтобы сохранить результат распаковки в ту же директорию, в которую уже сохранены основные данные, используйте такой код:
from randan.tools.df2file import df2fileShell
df2fileShell('{complicatedNamePart}', Новый_датафрейм, '{fileFormatChoice}', column, {'{coLabFolder}' if coLabFolder else None}, '{momentCurrent.strftime("%Y%m%d_%H%M")}')
"""
                                 )
        if returnDfs: return itemS

    except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        # display(itemS)
        if not itemS: itemS = pandas.DataFrame()
        dfsProcessor(complicatedNamePart,
                     coLabFolder,
                     dfAdd,
                     itemS, # на обработке какой бы ни было выгрузки не возникла бы непреодолимая ошибка, сохранить следует выгрузку метода get
                     itemS,
                     domain,
                     fields,
                     fileFormatChoice,
                     filter,
                     goS, # единственная из функций, принимающая этот аргумент
                     method,
                     momentCurrent,
                     offset,
                     slash)

        if returnDfs: return itemS
