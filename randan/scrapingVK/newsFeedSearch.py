#!/usr/bin/env python
# coding: utf-8


# In[ ]:


# # 0 Активировать требуемые для работы скрипта модули и пакеты + пререквизиты


# In[ ]:


# 0.0 В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from datetime import date, datetime
        from randan.tools import calendarWithinYear # авторский модуль для работы с календарём конкретного года
        from randan.tools import coLabAdaptor # авторский модуль для адаптации текущего скрипта к файловой системе CoLab
        from randan.tools import df2file # авторский модуль для сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
        from randan.tools import files2df # авторский модуль для оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
        from randan.tools import varPreprocessor # авторский модуль для предобработки переменных номинального, порядкового, интервального и более высокого типа шкалы
        import numpy, os, pandas, re, shutil, time, requests, warnings
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print('Пакет', module, 'НЕ прединсталлируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталлирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print('Пакет', module, 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,',
                  'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
            break


# In[ ]:


# # 1 Авторские функции


# In[ ]:


# 1.0 для метода search из API ВК, помогающая работе с ключами
def bigSearch(
              API_keyS,
              count,
              end_time,
              fields,
              iteration,
              keyOrder,
              latitude,
              longitude,
              pause,
              q,
              start_from,
              start_time
              ):
    # print('    start_from', start_from) # для отладки
    dfAdd = pandas.DataFrame()
    goS = True
    params = {
              'access_token': API_keyS[keyOrder], # обязательный параметр
              'count': count, # опциональный параметр
              'end_time': end_time, # опциональный параметр
              'extended': 1, # опциональный параметр
              'fields': fields, # опциональный параметр
              'latitude': latitude, # опциональный параметр
              'longitude': longitude, # опциональный параметр
              'q': q, # опциональный параметр
              'start_from': start_from, # опциональный параметр
              'start_time': start_time, # опциональный параметр
              'v': '5.199' # обязательный параметр
              }
    while True:
        response = requests.get('https://api.vk.ru/method/newsfeed.search', params=params)
        response = response.json() # отобразить выдачу метода get в виде JSON
        # print('response', response) # для отладки
        if 'response' in response.keys():
            response = response['response']
            # print('    response.keys() внутри bigSearch', response.keys()) # для отладки
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

            elif 'Internal server error: Unknown error, try later' in response['error']['error_msg']:
                print('\nПохоже, ошибка на сервере ВК; подождите и запустите скрипт с начала')
                response = {'items': [], 'total_count': 0} # принудительная выдача для response
                goS = False # нет смысла продолжать исполнение скрипта
                break # и, следовательно, нет смысла в новых итерациях цикла

            elif 'User authorization failed' in response['error']['error_msg']:
                print(
'''
Похоже, аккаунт попал под ограничение. Оно может быть снято с аккаунта сразу или спустя какое-то время. Подождите или подготовьте новый ключ в другом аккаунте. И запустите скрипт с начала'''
                      )
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
        # display(dfAdd) # для отладки
        # print('dfAdd.columns', dfAdd.columns) # для отладки
        dfAdd['date'] = dfAdd['date'].apply(lambda content: datetime.fromtimestamp(content).strftime('%Y.%m.%d'))
        dfAdd['URL'] = dfAdd['from_id'].astype(str)
        dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-') == False].index, 'URL'] = 'id' + dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-') == False].index, 'URL']
        dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-')].index, 'URL'] = dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-')].index, 'URL'].str.replace('-', 'public')
        dfAdd['URL'] = 'https://vk.com' + '/' + dfAdd['URL'] + '?w=' + dfAdd['inner_type'].str.split('_').str[0] + dfAdd['owner_id'].astype(str) + '_' + dfAdd['id'].astype(str)

        if fields != None:
            for fieldsColumn in ['groups', 'profiles']:
                if fieldsColumn in response.keys(): dfAdd = fieldsProcessor(dfIn=dfAdd, fieldsColumn=fieldsColumn, response=response)

    return dfAdd, goS, iteration, keyOrder, pause, response

# 1.1 для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessor(
                  complicatedNamePart,
                  coLabFolder,
                  dfAdd,
                  dfFinal, # на обработке какой бы ни было выгрузки не возникла бы непреодолима ошибка, сохранить следует выгрузку метода search
                  dfIn,
                  fileFormatChoice,
                  goS, # единственная из функций, принимающая этот аргумент
                  method,
                  momentCurrent,
                  q,
                  slash,
                  stage,
                  targetCount,
                  year,
                  yearsRange
                  ):
    df = pandas.concat([dfIn, dfAdd])
    columnsForCheck = []
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся  строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id' in column:
                columnsForCheck.append(column)
    # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck)
    df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True) # при дублировании объектов из itemS из Temporal и от пользователя и новых объектов, оставить новые

    if goS == False:
        print(
f'Поскольку исполнение скрипта натолкнулось на ошибку, сохраняю выгруженный контент и текущий этап поиска в директорию "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal"'
              )
        if not os.path.exists(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal'):
                os.makedirs(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal')
                print(f'Директория "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal" создана')
        # else:
            # print(f'Директория "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal" существует')

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}method.txt', 'w+') # открыть на запись
        file.write(method)
        file.close()

        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}q.txt', 'w+') # открыть на запись
        file.write(q if q != None else '')
        file.close()
    
        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}stageTarget.txt', 'w+')
        file.write(str(stageTarget)) # stageTarget принимает значения [0; 3]
        file.close()
    
        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}targetCount.txt', 'w+')
        file.write(str(targetCount))
        file.close()
    
        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}year.txt', 'w+')
        file.write(str(year)) # год, на котором остановилось исполнение скрипта
        file.close()

        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}yearsRange.txt', 'w+')
        file.write(yearsRange if yearsRange != None else '') # пользовательский временнОй диапазон
        file.close()
    
        df2file.df2fileShell(
                             complicatedNamePart=f'{complicatedNamePart}_Temporal',
                             dfIn=df,
                             fileFormatChoice=fileFormatChoice,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                             coLabFolder=coLabFolder,
                             currentMoment=momentCurrent.strftime("%Y%m%d") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             )
        warnings.filterwarnings("ignore")
        print(
'Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть',
'Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'
              )
        sys.exit()

    return df


# 1.2 для обработки выдачи аргумента fields
def fieldsProcessor(dfIn, fieldsColumn, response):
    df = dfIn.copy()
    idColumnS = []
    for column in df.columns:
    # for column in df.columns[1:]: # для отладки
        if 'id' in column:
            # print('column:', column) # для отладки
            idColumnS.append(column)
    columnsToJSON = varPreprocessor.jsonChecker(df)
    idColumnS.extend(columnsToJSON)
    
    fieldsDf = pandas.json_normalize(response[fieldsColumn])
    idS = pandas.json_normalize(response[fieldsColumn])['id'].to_list()
    if len(idS) > 0:
        idsCopy = pandas.json_normalize(response[fieldsColumn])['id'].to_list()
        idsCopyStr = '' # список в текстовый объект, чтобы ниже подать его внутрь столбца idColumnsConcatinated, созданного конкатенацией idColumnS
        for idCopy in idsCopy: idsCopyStr += str(idCopy) + ', '
        idsCopyStr = idsCopyStr[:-2]
        # print('idsCopyStr':, idsCopyStr) # для отладки
    
    def fieldsIdsChecker(cellContent): # функция, приминяемая ниже посредством apply , чтобы ускорить процесс (по сравнению с циклом по ячейкам)
        idsCopy = cellContent.split('idsCopy')[1].split(', ')
        cellContent = cellContent.split('idsCopy')[0]
        idsToItemS = []
        for idCopy in idsCopy:
            if idCopy in cellContent:
                idsToItemS.append(int(idCopy))
        if idsToItemS != []:
            # print('idsToItemS не пустой список:', idsToItemS) # для отладки
            try: return fieldsDf[fieldsDf['id'].isin(idsToItemS)].to_dict('records')
            except:
                print('!!! Ошибка:', sys.exc_info()[1])
                print('dict:', fieldsDf[fieldsDf['id'].isin(idsToItemS)].to_dict('records')) # для отладки
                return ''
        else:
            # print('idsToItemS пустой список:', idsToItemS) # для отладки
            return ''

    df['idColumnsConcatinated'] = ''
    for idColumn in idColumnS:
        df['idColumnsConcatinated'] += ' ' + df[idColumn].astype(str)

    df['idColumnsConcatinated'] += 'idsCopy' + idsCopyStr
    df[fieldsColumn] = df['idColumnsConcatinated'].apply(fieldsIdsChecker)
    df[fieldsColumn] = df[fieldsColumn].replace('N/A', numpy.NaN)
    return df

# # Код, чтобы распарсить любой из двух столбцов датафрейма itemS с выдачей аргумента fields
# fieldsColumn = 'groups'
# # fieldsColumn = 'profiles'
# fieldsJSON = []
# for cell in itemS[fieldsColumn].dropna():
#     fieldsJSON.extend(cell)
# pandas.json_normalize(fieldsJSON).drop_duplicates('id').reset_index(drop=True)


# In[ ]:


# # 2 Авторская функция исполнения скрипта


# In[ ]:


def newsFeedSearch(
                   params=None,
                   access_token=None,
                   count=None,
                   end_time=None,
                   fields=None,
                   latitude=None,
                   longitude=None,
                   q=None,
                   start_time=None,
                   returnDfs=False
                   ):
    """
    Функция для выгрузки характеристик контента ВК методом его API newsfeed.search. Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://dev.vk.com/ru/method/newsfeed.search
    Причём они могут быть поданы и в качестве самостоятельных аргументов функции, и в качестве словаря params , который обычно подаётся в метод get пакета requests
          params : dict
    access_token : str
           count : int
        end_time : int
          fields : list
        latitude : int
       longitude : int
               q : str
      start_time : int
       returnDfs : bool -- в случае True функция возвращает итоговый датафрейм с постами и их метаданными
    """
    if (params == None) & (access_token == None) & (count == None) & (end_time == None) & (fields == None) & (latitude == None) & (longitude == None) & (q == None) & (start_time == None) & (returnDfs == False):
        # print('Пользователь не подал аргументы') # для отладки
        expiriencedMode = False
        count = 200
    else:
        expiriencedMode = True
        if params != None:
            access_token = params['access_token'] if 'access_token' in params.keys() else None
            q = params['q'] if 'q' in params.keys() else None

            if 'start_time' in params.keys():
                start_time = params['start_time']
                if type(start_time) == str: start_time = int(start_time)
            else: start_time = None

            if 'count' in params.keys():
                count = params['count']
                if type(count) == str: count = int(count)
            else: count = None

            if 'end_time' in params.keys():
                end_time = params['end_time']
                if type(end_time) == str: end_time = int(end_time)
            else: end_time = None

            fields = params['fields'] if 'fields' in params.keys() else None

            if 'latitude' in params.keys():
                latitude = params['latitude']
                if type(latitude) == str: latitude = int(latitude)
            else: latitude = None

            if 'longitude' in params.keys():
                longitude = params['longitude']
                if type(longitude) == str: longitude = int(longitude)
            else: longitude = None

    if expiriencedMode == False:
        print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1IiIWweiLP1GDl_f4yyhJO2F4K_RceTc3OSqMYotCXVg ). Для получения API key следует создать приложение и из него скопировать сервисный ключ. Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК. Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком (одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК: посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта, то Вам потребуется дополнительная авторизация.
    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.'''
              )
    print(
'''    Скрипт нацелен на выгрузку характеристик контента ВК методом его API newsfeed.search. Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам. При этом следует учесть, что текущая версия API ВК, на которой и основан скрипт, "не умеет" обращаться к контенту, опубликованному ранее 2020 года
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента, не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК'''
          )
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API ВК
    coLabFolder = coLabAdaptor.coLabAdaptor()
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

    momentCurrent = datetime.now() # запрос текущего момента
    print('\nТекущий момент:', momentCurrent.strftime("%Y%m%d_%H%M"), '-- он будет использована для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')
    year = int(momentCurrent.strftime("%Y")) # в случае отсутствия пользовательского временнОго диапазона
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
            print(
'''--- НЕ нашёл файл credentialsVK.txt . Введите в окно Ваш API key для авторизации в API ВК 
(примерная инструкция, как создать API key, доступна по ссылке https://docs.google.com/document/d/1dRqPGzLgr1wLp-_N6iuuZCmzCqrjYg1PuH7G7yomYdw ). Для подстраховки от ограничения действия API key желательно создать несколько ключей (три -- отлично) и ввести их без кавычек через запятую с пробелом
--- После ввода нажмите Enter'''
                  )
            while True:
                API_keyS = input()
                if len(API_keyS) != 0:
                    print('-- далее буд[е у]т использован[ы] эт[от и] ключ[и]')

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
            if len(os.listdir(rootName)) == 7:
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
    
                file = open(f'{rootName}{slash}q.txt', encoding='utf-8') # 
                q = file.read()
                file.close()
                if q == '': q = None # для единообразия
    
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
                if year < int(momentCurrent.strftime("%Y")): print('- и на годе (при сегментировани по годам)', year)
                print('- пользователь НЕ сформулировал запрос-фильтр' if q == None else  f'- пользователь сформулировал запрос-фильтр как "{q}"')
                print('- пользователь НЕ ограничил временнОй диапазон' if yearsRange == None else  f'- пользователь ограничил временнОй диапазон границами {yearsRange}')
                print(
'''--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter
--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите "R" и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter'''
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
                            break
    
                    if yearsRange != None:
                        yearsRange = yearsRange.split('-')
                        yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)
# Данные, сохранённые при прошлом запуске скрипта, загружены; их метаданные (q, yearsRange, stageTarget) будут использоваться при исполнении скрипта
                    break
                elif decision == 'R': shutil.rmtree(rootName, ignore_errors=True)

# 2.0.3 Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if temporalName == None: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        rootName = 'No folder'
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку')
        print(
'''
Возможно, Вы располагаете файлом, в котором есть ранее выгруженные из ВК методом newsfeed.search данные, и который хотели бы дополнить? Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом формата XLSX, укажите полный путь, включая название файла, и нажмите Enter. Затем при необходимости сможете добавить к нему другие располагаемые файлы'''
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
        if q == None: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать контент в постах открытых аккаунтов по текстовому запросу-фильтру
--- Введите текст запроса-фильтра, который ожидаете найти в постах, после чего нажмите Enter'''
                  )
            if folderFile != None:
                print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folderFile, 'будут дополнены актуальными данными из выдачи скрипта',
'(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла'
                      )
            q = input()
            if q == '': q = None # для единообразия
            else: print('')

        # Ограничения временнОго диапазона
        if (start_time == None) & (end_time == None) & (yearsRange == None): # если пользователь не подал эти аргументы в рамках experiencedMode
            print(
'''Если требуется конкретный временнОй диапазон, то можно использовать его не на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными. Проблема в том, что без назначения временнОго диапазона метод newsfeed.search выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту
--- Поэтому если всё же требуется назначить временнОй диапазон на этапе выгрузки данных, назначьте его годами (а не более мелкими единицами времени). Для назначения диапазона введите без кавычек минимальный год диапазона, тире, максимальный год диапазона (минимум и максимум могут совпадать в такой записи: "год-тот же год") и нажмите Enter !!! Cледует учесть, что текущая версия API ВК "не умеет" обращаться к контенту, опубликованному ранее 2020 года
--- Если НЕ требуется назначить временнОй диапазон на этапе выгрузки данных, нажмите Enter'''
                  )
            while True:
                yearsRange = input()
                if len(yearsRange) != 0:
                    yearsRange = re.sub(r' *', '', yearsRange)
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
        if start_time != None:
            yearMinByUser = int(datetime.fromtimestamp(start_time).strftime('%Y')) # из experiencedMode
            # print('elif start_time != None:', yearMinByUser) # для отладки

        if end_time != None:
            yearMaxByUser = int(datetime.fromtimestamp(end_time).strftime('%Y')) # из experiencedMode
            # print('elif end_time != None:', yearMaxByUser) # для отладки
            year = yearMaxByUser

        if (yearMinByUser != None) & (yearMaxByUser == None): yearMaxByUser = int(momentCurrent.strftime("%Y")) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
        elif (yearMinByUser == None) & (yearMaxByUser != None): yearMaxByUser = 1970 # в случае отсутствия пользовательской нижней временнОй границы при наличии верхней

        # print('yearMinByUser', yearMinByUser) # для отладки
        # print('yearMaxByUser', yearMaxByUser) # для отладки

        if (start_time == None) & (yearMinByUser != None): start_time = int(datetime(yearMinByUser, 1, 1).timestamp()) # int(time.mktime(datetime(yearMinByUser, 1, 1).timetuple()))
        if (end_time == None) & (yearMaxByUser != None): end_time = int(datetime(yearMaxByUser, 12, 31).timestamp())

        if yearsRange != None: print('') # чтобы был отступ, если пользователь подал этот аргумент

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_VK'
    if q != None: complicatedNamePart += "_" + q if len(q) < 50 else "_" + q[:50]
    complicatedNamePart += "" if ((yearMinByUser == None) & (yearMaxByUser == None)) else "_" + str(yearMinByUser) + '-' + str(yearMaxByUser)
    # print('complicatedNamePart', complicatedNamePart)

# 2.1 Первичный сбор контента методом search
# 2.1.0 Первое обращение к API БЕЗ аргументов start_time, end_time (этап stage = 0)
    stage = 0
    method = 'newsfeed.search'
    iteration = 0 # номер итерации применения текущего метода
    pause = 0.25
    print(
f'В скрипте используются следующие аргументы метода {method} API ВК: q, start_from, start_time, end_time, expand.',
'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
f'Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/newsfeed.search ,',
f'-- можете подать их в скобки функции newsFeedSearch перед её запуском или скопировать код исполняемого сейчас скрипта и сделать это внутри кода внутри метода {method} в разделе 2'
          )
    # print('expiriencedMode:', expiriencedMode) # для отладки
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
    print('') # для отступа

    if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
        print('Первое обращение к API -- прежде всего, чтобы узнать примерное число доступных релевантных объектов')
        # print('    start_from', start_from) # для отладки
        itemsAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                               API_keyS=API_keyS,
                                                                               count=count,
                                                                               end_time=end_time,
                                                                               fields=fields,
                                                                               iteration=iteration,
                                                                               keyOrder=keyOrder,
                                                                               latitude=latitude,
                                                                               longitude=longitude,
                                                                               pause=pause,
                                                                               q=q,
                                                                               start_from=None,
                                                                               start_time=start_time
                                                                               )
        targetCount = response['total_count']
        if targetCount == 0:
            print(
'  Искомых объектов на серверах ВК по Вашему запросу, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта. Что делать? Поменяйте настройки запроса и запустите скрипт с начала'
                  )
            warnings.filterwarnings("ignore")
            print(
'Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть',
'Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'
                  )
            sys.exit()

        # if len(itemS) < targetCount: # на случай достаточности
        itemS = dfsProcessor(
                             complicatedNamePart=complicatedNamePart,
                             coLabFolder=coLabFolder,
                             fileFormatChoice=fileFormatChoice,
                             goS=goS,
                             dfAdd=itemsAdditional,
                             dfFinal=itemS,
                             dfIn=itemS,
                             method=method,
                             momentCurrent=momentCurrent,
                             q=q,
                             slash=slash,
                             stage=stage,
                             targetCount=targetCount,
                             year=year,
                             yearsRange=yearsRange
                             )
        print('  Проход по всем следующим страницам с выдачей          ')
        while 'next_from' in response.keys():
            start_from = response['next_from']
            # print('    start_from', start_from) # для отладки
            itemsAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                                   API_keyS=API_keyS,
                                                                                   count=count,
                                                                                   end_time=end_time,
                                                                                   fields=fields,
                                                                                   iteration=iteration,
                                                                                   keyOrder=keyOrder,
                                                                                   latitude=latitude,
                                                                                   longitude=longitude,
                                                                                   pause=pause,
                                                                                   q=q,
                                                                                   start_from=start_from,
                                                                                   start_time=start_time
                                                                                   )

            # print('''    response['next_from'] после bigSearch''', response['next_from']) # для отладки

            itemS = dfsProcessor(
                                 complicatedNamePart=complicatedNamePart,
                                 coLabFolder=coLabFolder,
                                 fileFormatChoice=fileFormatChoice,
                                 goS=goS,
                                 dfAdd=itemsAdditional,
                                 dfFinal=itemS,
                                 dfIn=itemS,
                                 method=method,
                                 momentCurrent=momentCurrent,
                                 q=q,
                                 slash=slash,
                                 stage=stage,
                                 targetCount=targetCount,
                                 year=year,
                                 yearsRange=yearsRange
                                 )
        print('  Искомых объектов', targetCount, ', а найденных БЕЗ сегментирования по годам и месяцам:', len(itemS))

# 2.1.1 Этап сегментирования по годам и месяцам (stage = 1)
    stage = 1
    if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
        if len(itemS) < targetCount:
        # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ сегментирования по годам и месяцам
            print(
'''Увы, без назначения временнОго диапазона метод newsfeed.search выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту. Поэтому внутри каждого года, начиная с текущего, помесячно выгружаю контент, после чего меняю год -- вглубь веков, пока не достигну заданной пользователем левой границы временнОго диапазона или года с пустой выдачей'''
                  )
            print(
'''--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам и месяцам, просто нажмите Enter, но учтите, что поиск может занять минуты и даже часы
--- Если НЕ хотите, нажмите пробел и затем Enter'''
                  )
            if len(input()) == 0:
                while True:
                    # print('Ищу текст запроса-фильтра в контенте за', year, 'год')
                    calendar = calendarWithinYear.calendarWithinYear(year)
                    itemsYearlyAdditional = pandas.DataFrame()
                    calendarColumnS = calendar.columns
                    if year == int(momentCurrent.strftime("%Y")): calendarColumnS = calendarColumnS[:int(momentCurrent.strftime("%m"))]
                            # чтобы исключить проход по будущим месяцам текущего года
                    for month in calendarColumnS:
                        print('Ищу текст запроса-фильтра в контенте за',  month, 'месяц', year, 'года', '               ') # , end='\r'
                        print('  Заход на первую страницу выдачи', '               ', end='\r')
                        itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                                                      API_keyS=API_keyS,
                                                                                                      count=count,
                                                                                                      end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1])).timestamp()),
                                                                                                      fields=fields,
                                                                                                      iteration=iteration,
                                                                                                      keyOrder=keyOrder,
                                                                                                      latitude=latitude,
                                                                                                      longitude=longitude,
                                                                                                      pause=pause,
                                                                                                      q=q,
                                                                                                      start_from=None,
                                                                                                      start_time=int(datetime(year, int(month), 1).timestamp())
                                                                                                      )
                        itemsYearlyAdditional = dfsProcessor(
                                                             complicatedNamePart=complicatedNamePart,
                                                             coLabFolder=coLabFolder,
                                                             fileFormatChoice=fileFormatChoice,
                                                             goS=goS,
                                                             dfAdd=itemsMonthlyAdditional,
                                                             dfFinal=itemS,
                                                             dfIn=itemS,
                                                             method=method,
                                                             momentCurrent=momentCurrent,
                                                             q=q,
                                                             slash=slash,
                                                             stage=stage,
                                                             targetCount=targetCount,
                                                             year=year,
                                                             yearsRange=yearsRange
                                                             )
                        print('  Проход по всем следующим страницам с выдачей', '               ', end='\r')
                        while 'next_from' in response.keys():
                            start_from = response['next_from']
                            # print('    start_from', start_from) # для отладки
                            itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = bigSearch(
                                                                                                          API_keyS=API_keyS,
                                                                                                          count=count,
                                                                                                          end_time=int(datetime(year, int(month), int(calendar[month].dropna().index[-1])).timestamp()),
                                                                                                          fields=fields,
                                                                                                          iteration=iteration,
                                                                                                          keyOrder=keyOrder,
                                                                                                          latitude=latitude,
                                                                                                          longitude=longitude,
                                                                                                          pause=pause,
                                                                                                          q=q,
                                                                                                          start_from=start_from,
                                                                                                          start_time=int(datetime(year, int(month), 1).timestamp())
                                                                                                          )
                            itemsYearlyAdditional = dfsProcessor(
                                                                 complicatedNamePart=complicatedNamePart,
                                                                 coLabFolder=coLabFolder,
                                                                 fileFormatChoice=fileFormatChoice,
                                                                 goS=goS,
                                                                 dfAdd=itemsMonthlyAdditional,
                                                                 dfFinal=itemS,
                                                                 dfIn=itemS,
                                                                 method=method,
                                                                 momentCurrent=momentCurrent,
                                                                 q=q,
                                                                 slash=slash,
                                                                 stage=stage,
                                                                 targetCount=targetCount,
                                                                 year=year,
                                                                 yearsRange=yearsRange
                                                                 )
                            time.sleep(pause)
                    itemS = dfsProcessor(
                                         complicatedNamePart=complicatedNamePart,
                                         coLabFolder=coLabFolder,
                                         fileFormatChoice=fileFormatChoice,
                                         goS=goS,
                                         dfAdd=itemsYearlyAdditional,
                                         dfFinal=itemS,
                                         dfIn=itemS,
                                         method=method,
                                         momentCurrent=momentCurrent,
                                         q=q,
                                         slash=slash,
                                         stage=stage,
                                         targetCount=targetCount,
                                         year=year,
                                         yearsRange=yearsRange
                                         )
                    # display(itemS.head())
                    # print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0])

                    if len(itemsYearlyAdditional) == 0:
                        print(f'\nВыдача для года {year} -- пуста'
                              , '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать двигаться к следующему месяцу вглубь веков, просто нажмите Enter'
                              , '\n--- Если хотите, нажмите пробел и затем Enter')
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
        print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal"')

# 2.1.2 Экспорт выгрузки метода search и финальное завершение скрипта
    df2file.df2fileShell(
                         complicatedNamePart=complicatedNamePart,
                         dfIn=itemS,
                         fileFormatChoice=fileFormatChoice,
                         method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
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
'''
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
# warnings.filterwarnings("ignore")
# print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть')
# input()
# sys.exit()
