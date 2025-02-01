#!/usr/bin/env python
# coding: utf-8


# In[ ]:


# # 0 Активировать требуемые для работы скрипта модули и пакеты + пререквизиты


# In[ ]:


# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from datetime import date, datetime
        from randan.tools import calendarWithinYear # авторский модуль для работы с календарём конкретного года
        from randan.tools import df2file # авторский модуль для сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
        from randan.tools import files2df # авторский модуль для оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
        from tqdm import tqdm
        import os, pandas, re, shutil, time, requests, warnings
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print('Пакет', module, 'НЕ прединсталлируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталлирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print('Пакет', module
                  , 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,'
                  , 'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
            break
tqdm.pandas() # для визуализации прогресса функций, применяемых к датафреймам


# In[ ]:


# # 1 Авторские функции


# In[ ]:


# 1.0 для метода search из API ВК, помогающая работе с ключами
def bigSearch(
              params
              , API_keyS
              , goS
              , iteration
              , keyOrder
              , pause
              , q
              , latitude
              , longitude
              , fields
              , start_from
              , start_time
              , end_time
              ):
    dfAdd = pandas.DataFrame()
    while True:
        if params == None:
            params = {
                'access_token': API_keyS[keyOrder] # обязательный параметр
                , 'v': '5.199' # обязательный параметр
                , 'q': q # опциональный параметр
                , 'count': 100 # опциональный параметр
                , 'start_time': start_time # опциональный параметр
                , 'end_time': end_time # опциональный параметр
                , 'latitude': latitude # опциональный параметр
                , 'longitude': longitude # опциональный параметр
                , 'extended': 1 # опциональный параметр
                , 'fields': fields # опциональный параметр
                , 'start_from': start_from # опциональный параметр
                }            
        response = requests.get('https://api.vk.ru/method/newsfeed.search', params=params)
        response = response.json() # отобразить выдачу метода get в виде JSON
        # print('response', response) # для отладки
        if 'response' in response.keys():
            response = response['response']
            dfAdd = pandas.json_normalize(response['items'])
            break
        elif 'error' in response.keys():
            if 'Too many requests per second' in response['error']['error_msg']:
                # print('  keyOrder до замены', '                    ') # для отладки
                keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
                print(f'\nПохоже, ключ попал под ограничение вследствие слишком высокой частоты обращения скрипта к API; пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту')
                # print('  keyOrder после замены', keyOrder, '                    ') # для отладки
                pause += 0.25

            elif 'Unknown application: could not get application' in response['error']['error_msg']:
                # print('  keyOrder до замены', '                    ') # для отладки                
                keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
                print('\nПохоже, Ваше ВК-приложение попало под ограничение; пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту')
                # print('  keyOrder после замены', keyOrder, '                    ') # для отладки
                pause += 0.25

            elif 'User authorization failed' in response['error']['error_msg']:
                print('\nПохоже, аккаунт попал под ограничение. Оно может быть снято с аккаунта сразу или спустя какое-то время.'
                      , 'Подождите или подготовьте новый ключ в другом аккаунте. И запустите скрипт с начала')
                response = {'items': [], 'total_count': 0} # принудительная выдача для response
                goS = False # нет смысла продолжать исполнение скрипта
                break # и, следовательно, нет смысла в новых итерациях цикла   
            
            else:
                print('  Похоже, проблема НЕ в слишком высокой частоте обращения скрипта к API((')
                print('  ', response['error']['error_msg'])
                goS = False # нет смысла продолжать исполнение скрипта
                break # и, следовательно, нет смысла в новых итерациях цикла                

    # Для визуализации процесса
    print('    Итерация №', iteration, ', number of items', len(response['items']), '                    ', end='\r')
    iteration += 1

    # Сменить формат представления дат, класс данных столбцов с id, создать столбец с кликабельными ссылками на контент
        # Здесь, а не в конце, поскольку нужна совместимость с itemS из Temporal и от пользователя
    if len(dfAdd) > 0:
        dfAdd['date'] = dfAdd['date'].apply(lambda content: datetime.fromtimestamp(content).strftime('%Y.%m.%d'))
        dfAdd['URL'] = dfAdd['from_id'].astype(str)
        dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-') == False].index, 'URL'] = 'id' + dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-') == False].index, 'URL']
        dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-')].index, 'URL'] = dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-')].index, 'URL'].str.replace('-', 'public')
        dfAdd['URL'] = 'https://vk.com' + '/' + dfAdd['URL'] + '?w=' + dfAdd['inner_type'].str.split('_').str[0] + dfAdd['owner_id'].astype(str) + '_' + dfAdd['id'].astype(str)

    return dfAdd, goS, iteration, keyOrder, pause, response

# 1.1 для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessing(complicatedNamePart, fileFormatChoice, dfAdd, dfFinal, dfIn, goS, method, q, slash, stage, targetCount, today, year, yearsRange):
    df = pandas.concat([dfIn, dfAdd])        
    columnsForCheck = []
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id' in column:
                columnsForCheck.append(column)
    # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck)
    df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True) # при дублировании объектов из itemS из Temporal и от пользователя и новых объектов, оставить новые 

    if goS == False:
        print('Поскольку исполнение скрипта натолкнулось на ошибку,'
              , f'сохраняю выгруженный контент и текущий этап поиска в директорию "{today}{complicatedNamePart}_Temporal"')
        if not os.path.exists(f'{today}{complicatedNamePart}_Temporal'):
                os.makedirs(f'{today}{complicatedNamePart}_Temporal')
                print(f'Директория "{today}{complicatedNamePart}_Temporal" создана')
        # else:
            # print(f'Директория "{today}{complicatedNamePart}_Temporal" существует')
        saveSettings(complicatedNamePart, fileFormatChoice, itemS, method, q, slash, stage, targetCount, today, year, yearsRange)
        print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit"'
              , '\nТак и должно быть'
              , '\nМодуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
        sys.exit()
    return df

# 1.2 для сохранения следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
def saveSettings(complicatedNamePart, fileFormatChoice, itemS, method, q, slash, stageTarget, targetCount, today, year, yearsRange):
    file = open(f'{today}{complicatedNamePart}_Temporal{slash}method.txt', 'w+') # открыть на запись
    file.write(method)
    file.close()
    
    file = open(f'{today}{complicatedNamePart}_Temporal{slash}q.txt', 'w+')
    file.write(q)
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}stageTarget.txt', 'w+')
    file.write(str(stageTarget)) # stageTarget принимает значения [0; 3]
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}targetCount.txt', 'w+')
    file.write(str(targetCount))
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}year.txt', 'w+')
    file.write(str(year)) # год, на котором остановилось исполнение скрипта
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}yearsRange.txt', 'w+')
    file.write(yearsRange if yearsRange != None else '') # пользовательский временнОй диапазон
    file.close()

    if '.' in method: df2file.df2fileShell(f'{complicatedNamePart}_Temporal', itemS, fileFormatChoice, method.split('.')[0] + method.split('.')[1].capitalize(), today)
        # чтобы избавиться от лишней точки в имени файла
    else: df2file.df2fileShell(f'{complicatedNamePart}_Temporal', itemS, fileFormatChoice, method, today)
    if os.path.exists(rootName):
        print('Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы,'
              , 'УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта')
        shutil.rmtree(rootName, ignore_errors=True)


# In[ ]:


# # 2 Авторская функция исполнения скрипта


# In[ ]:


def newsFeedSearch(
                   params=None
                   , access_token=None
                   , q=None
                   , start_time=None
                   , end_time=None
                   , latitude=None
                   , longitude=None
                   , fields=None
                   ):
    """
    Функция для выгрузки характеристик контента ВК методом его API newsfeed.search. Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам
    
    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/newsfeed.search
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params , который обычно подаётся в метод get пакета requests
    access_token : str
               q : str
      start_time : int
        end_time : int
        latitude : int
       longitude : int
          fields : list
          params : dict
    """
    if (access_token == None) & (q == None) & (start_time == None) & (end_time == None) & (latitude == None) & (longitude == None) & (fields == None) & (params == None):
        # print('Пользователь не подал аргументы')
        expiriencedMode = False
    else:
        expiriencedMode = True
        if params != None:
            access_token = params['access_tokenq'] if 'access_token' in params.keys() else None
            q = params['q'] if 'q' in params.keys() else None
            start_time = params['start_time'] if 'start_time' in params.keys() else None
            end_time = params['end_time'] if 'end_time' in params.keys() else None

    if expiriencedMode == False:
        print('    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными).'
              , 'Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию:'
              , 'https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ.'
              , 'Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК.'
              , 'Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком'
              , '(одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК:'
              , 'посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта,'
              , 'то Вам потребуется дополнительная авторизация.'
              , '\n    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.')
    print('    Скрипт нацелен на выгрузку характеристик контента ВК методом его API newsfeed.search. Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам.'
          , '\n    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.'
          , '\n    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel.'
          , 'Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента,'
          , 'не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК')
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API ВК
    fileFormatChoice = '.xlsx' # базовый формат сохраняемых файлов; формат .json добавляется опционально через наличие columnsToJSON
    folder = None
    folderFile = None
    goS = True
    itemS = pandas.DataFrame()
    keyOrder = 0
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    stageTarget = 0 # stageTarget принимает значения [0; 3] и относится к стадиям скрипта
    temporalName = None
    yearsRange = None

    today = date.today().strftime("%Y%m%d") # запрос сегодняшней даты в формате yyyymmdd
    print('\nТекущяя дата:', today, '-- она будет использована для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')
    # print('Сегодня год:', today[:4])
    # print('Сегодня месяц:', today[4:6])
    # print('Сегодня день:', today[6:])
    year = int(today[:4]) # в случае отсутствия пользовательского временнОго диапазона
        # с этого года возможно сегментирование по годам вглубь веков (пока выдача не пустая)
    yearMinByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearMaxByUser = None # в случае отсутствия пользовательского временнОго диапазона

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
            print('--- НЕ нашёл файл credentialsVK.txt . Введите в окно Ваш API key для авторизации в API ВК'
                  , '(примерная инструкция, как создать API key, доступна по ссылке https://docs.google.com/document/d/1dRqPGzLgr1wLp-_N6iuuZCmzCqrjYg1PuH7G7yomYdw ).'
                  , 'Для подстраховки от ограничения действия API key желательно создать несколько ключей (три -- отлично) и ввести их без кавычек через запятую с пробелом'
                  , '\n--- После ввода нажмите Enter')
            while True:
                API_keyS = input()
                if len(API_keyS) != 0:
                    print('-- далее буд[е у]т использован[ы] эт[от и] ключ[и]')

                    from randan.tools.textPreprocessing import multispaceCleaner # авторский модуль для предобработки нестандартизированного текста
                    API_keyS = multispaceCleaner(API_keyS)
                    while API_keyS[-1] == ',': API_keyS = API_keyS[:-1] # избавиться от запятых в конце текста

                    file = open("credentialsVK.txt", "w+") # открыть на запись
                    file.write(API_keyS)
                    file.close()
                    break
                else:
                    print('--- Вы ничего НЕ ввели. Попробуйте ещё раз..')
        API_keyS = API_keyS.split(', ')
    else: API_keyS = [access_token]
    print('Количество ключей:', len(API_keyS), '\n')

# 2.0.2 Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
    # Поиск данных
    print('Проверяю наличие директории Temporal с данными и их мета-данными,'
          ,'гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку')
    for rootName in rootNameS:
        if 'Temporal' in rootName:
            file = open(f'{rootName}{slash}targetCount.txt')
            targetCount = file.read()
            file.close()
            targetCount = int(targetCount)
    
            file = open(f'{rootName}{slash}method.txt')
            method = file.read()
            file.close()
    
            file = open(f'{rootName}{slash}year.txt')
            year = file.read()
            file.close()
            year = int(year)
    
            file = open(f'{rootName}{slash}q.txt') # , encoding='utf-8'
            q = file.read()
            file.close()
    
            file = open(f'{rootName}{slash}yearsRange.txt')
            yearsRange = file.read()
            file.close()
    
            file = open(f'{rootName}{slash}stageTarget.txt')
            stageTarget = file.read()
            file.close()
            stageTarget = int(stageTarget)
    
            print(f'Нашёл директорию "{rootName}". В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:'
                  # , '\n- было выявлено целевое число объектов (targetCount)', targetCount
                  , '\n- скрипт остановился на методе', method)
            if year < int(today[:4]): print('- и на годе (при сегментировани по годам)', year)
            print('- пользователь НЕ сформулировал запрос-фильтр' if q == '' else  f'- пользователь сформулировал запрос-фильтр как "{q}"')
            print('- пользователь НЕ ограничил временнОй диапазон' if yearsRange == None else  f'- пользователь ограничил временнОй диапазон границами {yearsRange}')
            print('--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter'
                  , '\n--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите "R" и нажмите Enter'
                  , '\n--- Если хотите найти другие промежуточные результаты, введите любой символ, кроме "R", и нажмите Enter')
            decision = input()
            if len(decision) == 0:
                temporalNameS = os.listdir(rootName)
                for temporalName in temporalNameS:
                    if '.xlsx' in temporalName: break
                itemS = pandas.read_excel(f'{rootName}{slash}{temporalName}', index_col=0)
                
                for temporalName in temporalNameS:
                    if '.json' in temporalName: break
                itemS = itemS.merge(pandas.read_json(f'{rootName}{slash}{temporalName}'), on='id', how='outer')
                
                if yearsRange != None:
                    yearsRange = yearsRange.split('-')
                    yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)
# Данные, сохранённые при прошлом запуске скрипта, загружены; их метаданные (q, yearsRange, stageTarget) будут использоваться при исполнении скрипта
                break
            elif decision == 'R': shutil.rmtree(rootName, ignore_errors=True)

# 2.0.3 Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if temporalName == None: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку')
        print('\nВозможно, Вы располагаете файлом, в котором есть ранее выгруженные из ВК методом newsfeed.search данные, и который хотели бы дополнить?'
              , 'Или планируете первичный сбор контента?'
              , '\n--- Если планируете первичный сбор, нажмите Enter'
              , '\n--- Если располагаете файлом формата XLSX, укажите полный путь, включая название файла, и нажмите Enter. Затем при необходимости сможете добавить к нему другие располагаемые файлы')
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
        if q == None: # если пользователь не подал этот аргумент в рамках experiencedMode
            print('Скрипт умеет искать контент в постах открытых аккаунтов по текстовому запросу-фильтру'
                  , '\n--- Введите текст запроса-фильтра, который ожидаете найти в постах, после чего нажмите Enter')
            if folderFile != None: print('ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла'
                , folderFile
                , 'будут дополнены актуальными данными из выдачи скрипта'
                , '(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),'
                , 'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла')
            q = input()
    
        # Ограничения временнОго диапазона
        if (start_time == None) & (end_time == None) & (yearsRange == None): # если пользователь не подал эти аргументы в рамках experiencedMode
            while True:
                print('\nЕсли требуется конкретный временнОй диапазон, то можно использовать его не на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными.'
                      , 'Проблема в том, что без назначения временнОго диапазона метод newsfeed.search выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту.'
                      , '\n--- Поэтому если всё же требуется назначить временнОй диапазон на этапе выгрузки данных, назначьте его годами (а не более мелкими единицами времени).'
                      , 'Для назначения диапазона введите без кавычек минимальный год диапазона, тире, максимальный год диапазона (минимум и максимум могут совпадать в такой записи: "год-тот же год") и нажмите Enter'
                      , '\n--- Если НЕ требуется назначить временнОй диапазон на этапе выгрузки данных, нажмите Enter')
                yearsRange = input()
                if len(yearsRange) != 0:
                    yearsRange = re.sub(r' *', '', yearsRange)
                    if '-' in yearsRange:
                        yearsRange = yearsRange.split('-')
                        if len(yearsRange) == 2:
                            yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)
                            year = yearMaxByUser
                            break
                        else: print('--- Вы ввели тире, но при этом ввели НЕ два года. Попробуйте ещё раз..')
                    else: print('--- Вы НЕ ввели тире. Попробуйте ещё раз..')
                else:
                    yearsRange = None # для унификации
                    break
        if start_time != None:
            yearMinByUser = int(datetime.fromtimestamp(start_time).strftime('%Y')) # из experiencedMode
            # print('elif start_time != None:', yearMinByUser) # для отладки
        
        if end_time != None:
            yearMaxByUser = int(datetime.fromtimestamp(end_time).strftime('%Y')) # из experiencedMode
            # print('elif end_time != None:', yearMaxByUser) # для отладки
            year = yearMaxByUser
        
        if (yearMinByUser != None) & (yearMaxByUser == None): yearMaxByUser = int(today[:4]) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
        elif (yearMinByUser == None) & (yearMaxByUser != None): yearMaxByUser = 1970 # в случае отсутствия пользовательской нижней временнОй границы при наличии верхней
            
        # print('yearMinByUser', yearMinByUser) # для отладки
        # print('yearMaxByUser', yearMaxByUser) # для отладки

        if (start_time == None) & (yearMinByUser != None): start_time = int(datetime(yearMinByUser, 1, 1).timestamp()) # int(time.mktime(datetime(yearMinByUser, 1, 1).timetuple()))
        if (end_time == None) & (yearMaxByUser != None): end_time = int(datetime(yearMaxByUser, 12, 31).timestamp())  

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    complicatedNamePart += "" if q == None else "_" + q[:50]
    complicatedNamePart += "" if ((yearMinByUser == None) & (yearMaxByUser == None)) else "_" + str(yearMinByUser) + '-' + str(yearMaxByUser)
    # print('complicatedNamePart', complicatedNamePart)

# 2.1 Первичный сбор контента методом search
# 2.1.0 Первое обращение к API БЕЗ аргументов start_time, end_time (этап stage = 0)
    stage = 0
    method = 'newsfeed.search'
    iteration = 0 # номер итерации применения текущего метода
    pause = 0.25

    print(f'В скрипте используются следующие аргументы метода {method} API ВК:'
          , 'q, start_from, start_time, end_time, expand.'
          , 'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
          , f'Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/newsfeed.search ,'
          , f'-- можете подать их в скобки функции newsFeedSearch перед её запуском или скопировать код исполняемого сейчас скрипта и сделать это внутри кода внутри метода {method} в чанке 1.1')
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
    
    if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
        print('\nПервое обращение к API -- прежде всего, чтобы узнать примерное число доступных релевантных объектов')      
        itemsAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                               params
                                                                               , API_keyS
                                                                               , goS
                                                                               , iteration
                                                                               , keyOrder
                                                                               , pause
                                                                               , q
                                                                               , latitude
                                                                               , longitude
                                                                               , fields
                                                                               , start_from=None
                                                                               , start_time=start_time
                                                                               , end_time=end_time
                                                                              )
        targetCount = response['total_count']
        # if len(itemS) < targetCount: # на случай достаточности
        itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, itemsAdditional, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
        print('  Проход по всем следующим страницам с выдачей          ')
        while 'next_from' in response.keys():
            start_from = response['next_from']
            # print('    start_from', start_from) # для отладки
            itemsAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                                   params
                                                                                   , API_keyS
                                                                                   , goS
                                                                                   , iteration
                                                                                   , keyOrder
                                                                                   , pause
                                                                                   , q
                                                                                   , latitude
                                                                                   , longitude
                                                                                   , fields
                                                                                   , start_from
                                                                                   , start_time=start_time
                                                                                   , end_time=end_time
                                                                                  )
            itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, itemsAdditional, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
        print('  Искомых объектов', targetCount, ', а найденных БЕЗ сегментирования по годам и месяцам:', len(itemS))

# 2.1.1 Этап сегментирования по годам и месяцам (stage = 1)
    stage = 1
    if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
        if len(itemS) < targetCount:
        # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ сегментирования по годам и месяцам
            print('Увы, без назначения временнОго диапазона метод newsfeed.search выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту.'
              , 'Поэтому внутри каждого года, начиная с текущего, помесячно выгружаю контент, после чего меняю год -- вглубь веков, пока не достигну заданной пользователем левой границы временнОго диапазона,'
              , 'или года с пустой выдачей')
            print('--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам, просто нажмите Enter, но учтите, что поиск может занять минуты и даже часы'
                  , '\n--- Если НЕ хотите, введите любой символ и нажмите Enter')
            if len(input()) == 0:
                while True:
                    # print('Ищу текст запроса-фильтра в контенте за', year, 'год')
                    calendar = calendarWithinYear.calendarWithinYear(year)
                    itemsYearlyAdditional = pandas.DataFrame()
                    calendarColumnS = calendar.columns
                    if year == int(today[:4]): calendarColumnS = calendarColumnS[:int(today[4:6])] # чтобы исключить проход по будущим месяцам текущего года
                    for month in calendarColumnS:
                        print('Ищу текст запроса-фильтра в контенте за',  month, 'месяц', year, 'года', '               ') # , end='\r'
                        print('  Заход на первую страницу выдачи', '               ', end='\r')
                        itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                                                      params
                                                                                                      , API_keyS
                                                                                                      , goS
                                                                                                      , iteration
                                                                                                      , keyOrder
                                                                                                      , pause
                                                                                                      , q
                                                                                                      , latitude
                                                                                                      , longitude
                                                                                                      , fields
                                                                                                      , start_from=None
                                                                                                      , start_time=int(datetime(year, int(month), 1).timestamp())
                                                                                                      , end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1])).timestamp())
                                                                                                      )
                        itemsYearlyAdditional = dfsProcessing(complicatedNamePart
                                                              , fileFormatChoice
                                                              , itemsMonthlyAdditional
                                                              , itemS
                                                              , itemsYearlyAdditional
                                                              , goS
                                                              , method
                                                              , q
                                                              , slash
                                                              , stage
                                                              , targetCount
                                                              , today
                                                              , year
                                                              , yearsRange)
                        print('  Проход по всем следующим страницам с выдачей', '               ', end='\r')
                        while 'next_from' in response.keys():
                            start_from = response['next_from']
                            # print('    start_from', start_from) # для отладки
                            itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                                                          params
                                                                                                          , API_keyS
                                                                                                          , goS
                                                                                                          , iteration
                                                                                                          , keyOrder
                                                                                                          , pause
                                                                                                          , q
                                                                                                          , latitude
                                                                                                          , longitude
                                                                                                          , fields
                                                                                                          , start_from
                                                                                                          , start_time=start_time
                                                                                                          , end_time=end_time
                                                                                                          )
                            itemsYearlyAdditional = dfsProcessing(complicatedNamePart
                                                                  , fileFormatChoice
                                                                  , itemsMonthlyAdditional
                                                                  , itemS
                                                                  , itemsYearlyAdditional
                                                                  , goS
                                                                  , method
                                                                  , q
                                                                  , slash
                                                                  , stage
                                                                  , targetCount
                                                                  , today
                                                                  , year
                                                                  , yearsRange)
                            time.sleep(pause)
                    itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, itemsYearlyAdditional, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
                    # display(itemS.head())
                    # print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0])
        
                    if len(itemsYearlyAdditional) == 0:
                        print(f'\nВыдача для года {year} -- пуста'
                              , '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать двигаться к следующему месяцу вглубь веков, просто нажмите Enter'
                              , '\n--- Если хотите, введите любой символ и нажмите Enter')
                        if len(input()) == 0:
                            # print(f'\nЗавершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser}')
                            break
            
                    elif yearMinByUser != None: # если пользователь ограничил временнОй диапазон
                        if year <= yearMinByUser:
                            print(f'Завершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser}')
                            break
        
                    print('  Искомых объектов', targetCount, ', а найденных после добавления контента', year, 'года:', len(itemS), '                    ')
                    year -= 1
                print('Искомых объектов', targetCount, ', а найденных:', len(itemS), '          ')
    
        # pandas.set_option('display.max_columns', None)
        display(itemS.head())
        print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0])
    
    elif stage < stageTarget:
        print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{today}{complicatedNamePart}_Temporal"')

# 2.1.2 Экспорт выгрузки метода search и финальное завершение скрипта
    df2file.df2fileShell(complicatedNamePart, itemS, '.xlsx', method.split('.')[0] + method.split('.')[1].capitalize(), today) # чтобы избавиться от лишней точки в имени файла

    print('Скрипт исполнен')
    if os.path.exists(rootName):
        print('Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы,'
              , 'УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта')
        shutil.rmtree(rootName, ignore_errors=True)
    
    warnings.filterwarnings("ignore")
    print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'
          , '\nМодуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
    sys.exit()

# warnings.filterwarnings("ignore")
# print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть)
# input()
# sys.exit()
