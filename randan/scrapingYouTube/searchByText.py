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
        import os, pandas, re, shutil, time, warnings
        import googleapiclient.discovery as api
        import googleapiclient.errors
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        if module == 'googleapiclient': module = 'google-api-python-client'
        print('Пакет', module, 'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print('Пакет', module
                  , 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,'
                  , 'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
            break


# In[ ]:


# # 1 Авторские функции


# In[ ]:


# 1.0 для метода search из API YouTube, помогающая работе с ключами
def bigSearch(
              API_keyS
              , channelIdForSearch # согласно документации API YouTube, подать можно лишь один channelId
              , contentType
              , goS
              , iteration
              , keyOrder
              , q
              , publishedAfter
              , publishedBefore
              , order
              , pageToken
              , year
              , channelType
              , eventType
              , location
              , locationRadius
              , regionCode
              , relevanceLanguage
              , topicId
              , safeSearch
              , videoCaption
              , videoCategoryId
              , videoDefinition
              , videoDimension
              , videoDuration
              , videoEmbeddable
              , videoLicense
              , videoPaidProductPlacement
              , videoType
              , videoSyndicated
              ):
    goC = True
    while goC:
        try:
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            request = youtube.search().list(channelId=channelIdForSearch
                                            , channelType=channelType
                                            , eventType=eventType
                                            , maxResults=50
                                            , order=order
                                            , pageToken=pageToken
                                            , part="snippet"
                                            , publishedAfter=publishedAfter
                                            , publishedBefore=publishedBefore
                                            , q=q
                                            , type=contentType   
                                            , location=location
                                            , locationRadius=locationRadius
                                            , regionCode=regionCode
                                            , relevanceLanguage=relevanceLanguage
                                            , topicId=topicId
                                            , safeSearch=safeSearch
                                            , videoCaption=videoCaption
                                            , videoCategoryId=videoCategoryId
                                            , videoDefinition=videoDefinition
                                            , videoDimension=videoDimension
                                            , videoDuration=videoDuration
                                            , videoEmbeddable=videoEmbeddable
                                            , videoLicense=videoLicense
                                            , videoPaidProductPlacement=videoPaidProductPlacement
                                            , videoType=videoType
                                            , videoSyndicated=videoSyndicated)
        # channelType="any",
        # eventType="live",
        # publishedAfter="1970-01-01T00:00:00Z",
        # q="мусорная+реформа",
        # safeSearch="moderate",
        # type="video",
        # videoCaption="any",
        # videoDefinition="any",
        # videoDimension="any",
        # videoDuration="any",
        # videoEmbeddable="any",
        # videoLicense="any",
        # videoPaidProductPlacement="any",
        # videoSyndicated="any",
        # videoType="any"
            response = request.execute()

            # Для визуализации процесса
            print('      Итерация №', iteration, ', number of items', len(response['items'])
                  , '' if year == None else f', year {year}'
                  , '' if order == None else f', order {order}'
                  , '          ', end='\r')
            iteration += 1
            goC = False

        except googleapiclient.errors.HttpError:
            errorDescription = sys.exc_info()
            goC, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder)

        except IndexError:
            errorDescription = sys.exc_info()
            response = {'kind': 'youtube#searchListResponse'
                        , 'pageInfo': {'totalResults': 0, 'resultsPerPage': 0}
                        , 'items': []} # принудительная выдача для response без request.execute()
            goC, goS = indexError(errorDescription)
    addItemS = pandas.json_normalize(response['items'])
    return addItemS, goS, iteration, keyOrder, response # от response отказаться нельзя, т.к. в нём много важных ключей, даже если их знчения нули

# 1.1 для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessing(complicatedNamePart, fileFormatChoice, dfAdd, dfFinal, dfIn, goS, method, q, slash, stage, targetCount, today, year, yearsRange):
    df = pandas.concat([dfIn, dfAdd])        
    columnsForCheck = []
    for column in df.columns: # выдача многих методов содержит столбец id, он оптимален для проверки дублирующихся строк
        if 'id' == column:
            columnsForCheck.append(column)
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id.' in column:
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

# 1.2 для выгрузки комментариев
def downloadComments(API_keyS, goS, sourceId, idS, keyOrder, maxResults, method, page, pageToken, part): # id видео или комментария
    goC_0 = True
    while goC_0: # этот цикл позволяет возвращяться со следующим keyOrder к прежнему id при истечении квоты текущего ключа
        try:
            addCommentS = pandas.DataFrame()
            goC_1 = True
            errorDescription = None
            problemItemId = None
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            while goC_1:
                if method == 'comments':
                    response = youtube.comments().list(maxResults=maxResults, pageToken=pageToken, part=part, parentId=sourceId).execute()
                    goC_1 = False
                elif method == 'commentThreads':
                    response = youtube.commentThreads().list(maxResults=maxResults, pageToken=pageToken, part=part, videoId=sourceId).execute()
                    goC_1 = False
                else:
                    print('--- В функцию downloadComments подано неправильное имя метода выгрузки комментариев'
                          , '\n--- Подайте, пожалуйста, правильное..')
            addCommentS = pandas.json_normalize(response['items'])
            pageToken = response['nextPageToken'] if 'nextPageToken' in response.keys() else None
            # print('nextPageToken', pageToken)
            if page != '':
                print('  Видео №', idS.index(sourceId) + 1, 'из', len(idS), '. Страница выдачи №', page, '          ', end='\r')
                page += 1
            goC_0 = False
            break
    
        except googleapiclient.errors.HttpError: # истечение квоты ключа и ошибка выгрузки комментария относятся к HttpError
            errorDescription = sys.exc_info()
            # print('\n    ', errorDescription[1])
            # if 'comment' in str(errorDescription[1]):
            #     print('  Ограничение выгрузки комментари[ев я] для id', id)
            #     problemItemId = id
            #     goC_0 = False # нет смысла возвращяться со следующим keyOrder к прежнему id
            #     break
            # elif 'quotaExceeded' in str(errorDescription[1]):
            #     print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
            #     keyOrder += 1                    
            # else:
            #     print('Похоже, проблема не в огрничении выгрузки комментари[ев я] и не в истечении квоты текущего ключа((')       
            #     problemItemId = id
            #     goC_0 = False
            #     break
            goC_0, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder, sourceId)
    
        except IndexError:
            errorDescription = sys.exc_info()
            # print('\n    ', errorDescription[1])
            goC_0, goS = indexError(errorDescription)
                   
        except TimeoutError:
            errorDescription = sys.exc_info()
            print(errorDescription[1])
            time.sleep(10)
        
    return addCommentS, errorDescription, goS, keyOrder, page, pageToken, problemItemId

# 1.3 для обработки ошибок
def googleapiclientError(errorDescription, keyOrder, *arg): # арки: id
    # print('\n    ', errorDescription[1])
    if 'quotaExceeded' in str(errorDescription[1]):
        print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
        keyOrder += 1 # смена ключа
        # print('  keyOrder', keyOrder)
        problemItemId = None # для унификации со следующим блоком условий
        goC = True # для повторного обращения к API с новым ключом
    else:
        if len(arg) == 1:
            print('  Проблема может быть связана с обрабатываемым объектом, поэтому фиксирую его id:', id)
            problemItemId = arg[0]
        if 'comment' in str(errorDescription[1]):
            print('  Ограничение выгрузки комментари[ев я] для id', id)
        else:
            print('  Похоже, проблема не в огрничении выгрузки комментари[ев я] и не в истечении квоты текущего ключа((')
        goC = False # нет смысла повторного обращения к API ни с этим id, ни пока не ясна суть ошибки
    return goC, keyOrder, problemItemId

# 1.4 для прерывания скрипта при исчерпании ключей
def indexError(errorDescription):
    print('\n    ', errorDescription[1])
    print('Похоже, ключи закончились. Подождите сутки для восстановления ключей или подготовьте новый ключ -- и запустите скрипт с начала')
    goS = False # нет смысла продолжать исполнение скрипта
    goC = False # и, следовательно, нет смысла в новых итерациях цикла
    return goC, goS

# 1.5 для визуализации процесса через итерации
def iterationVisualization(idS, portion):
    iterationUpperBound = len(idS)

    # Дробная часть после деления числа idS должна увеличить iterationUpperBound на единицу
    iterationUpperBound = str(round(len(idS) / portion, 0))\
        if (portion > 1) & (iterationUpperBound % portion == 0)\
        else str(round(len(idS) / portion, 0) + 1)

    # И `.0` лишние                      
    if '.' in iterationUpperBound: iterationUpperBound = iterationUpperBound.split('.')[0]

    print('  Порция №', iteration + 1, 'из', iterationUpperBound, end='\r')
    if portion > 1: print('   Сколько в порции наблюдений?', len(response['items']), end='\r')
        
# 1.6 для порционной выгрузки, когда метод предполагает подачу ему id порциями
def portionProcessing(complicatedNamePart, goS, idS, keyOrder, method, q, slash, stage, stageTarget, targetCount, year):
    bound = 0
    chplviS = pandas.DataFrame()
    goC = True
    iteration = 0 # номер итерации применения текущего метода
    while (bound < len(idS)) & (goC):
    # while (bound < 100) & (goC): # для отладки
        try:
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            if method == 'channels':
                response = youtube.channels().list(part=["snippet"
                                                        , "brandingSettings"
                                                        , "contentDetails"
                                                        , "id"
                                                        , "localizations"
                                                        , "statistics"
                                                        , "status"
                                                        , "topicDetails"]
                                                   , id=channelIdS[bound:bound + 50]
                                                   , maxResults=50
                                                   ).execute()
            if method == 'playlists':
                response = youtube.playlists().list(part=["snippet"
                                                         , "contentDetails"
                                                         , "localizations"
                                                         , "status"]
                                                    , id=idS[bound:bound + 50]
                                                    ).execute()
            if method == 'playlistItems':
                response = youtube.playlistItems().list(part=["snippet"]
                                                        , playlistId=idS[bound:bound + 50]
                                                        ).execute()
            if method == 'videos':
                response = youtube.videos().list(part=["snippet"
                                                      , "contentDetails"
                                                      , "localizations"
                                                      , "statistics"
                                                      , "status"
                                                      , "topicDetails"]
                                                 , id=videoIdS[bound:bound + 50]
                                                 , maxResults=50
                                                 ).execute()        
            # Для визуализации процесса через итерации
            iterationUpperBound = len(idS)
        
            # Дробная часть после деления числа idS должна увеличить iterationUpperBound на единицу
            iterationUpperBound = str(round(len(idS) / 50, 0)) if iterationUpperBound % 50 == 0 else str(round(len(idS) / 50, 0) + 1)
        
            # И `.0` лишние                      
            if '.' in iterationUpperBound: iterationUpperBound = iterationUpperBound.split('.')[0]
        
            print('  Порция №', iteration + 1, 'из', iterationUpperBound, '; сколько в порции наблюдений?', len(response['items']), end='\r')
        
            bound += 50
            iteration += 1

            addChplviS = pandas.json_normalize(response['items'])
            chplviS = dfsProcessing(complicatedNamePart, fileFormatChoice, addChplviS, chplviS, chplviS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
        except googleapiclient.errors.HttpError:
            errorDescription = sys.exc_info()
            # print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
            # keyOrder += 1
            googleapiclientError(errorDescription, keyOrder)
    
        except IndexError:
            errorDescription = sys.exc_info()
            print(errorDescription[1])
            print('Поскольку ключи закончились,'
                  , f'сохраняю выгруженный контент и текущий этап поиска в директорию "{complicatedNamePart} Temporal"')     
            if not os.path.exists(f'{complicatedNamePart} Temporal'):
                    os.makedirs(f'{complicatedNamePart} Temporal')
                    print(f'Директория "{complicatedNamePart} Temporal" создана')
            else:
                print(f'Директория "{complicatedNamePart} Temporal" существует')
            saveSettings(channelIdForSearch, complicatedNamePart, contentType, itemS, method, q, slash, stage, targetCount, year, yearsRange)
            goC = False
    return chplviS

# 1.7 чтобы избавиться от префиксов в названиях столбцов датафрейма с комментариями
def prefixDropper(df):
        dfNewColumnS = []
        for column in df.columns:
            if 'snippet.topLevelComment.' in column:
                column = column.replace('snippet.topLevelComment.', '')
            dfNewColumnS.append(column)
        df.columns = dfNewColumnS # перезаписать названия столбцов
        return df

# 1.8 для сохранения следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
def saveSettings(channelIdForSearch, complicatedNamePart, contentType, fileFormatChoice, itemS, method, q, slash, stageTarget, targetCount, today, year, yearsRange):
    file = open(f'{today}{complicatedNamePart}_Temporal{slash}channelIdForSearch.txt', 'w+') # открыть на запись
    file.write(channelIdsForSearch if channelIdsForSearch != None else '')
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}contentType.txt', 'w+') # открыть на запись
    file.write(contentType if contentType != None else '')
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}method.txt', 'w+') # открыть на запись
    file.write(method)
    file.close()

    file = open(f'{today}{complicatedNamePart}_Temporal{slash}q.txt', 'w+') # открыть на запись
    file.write(q if q != None else '')
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


def searchByText(access_token=None, channelIdForSearch=None, contentType=None, publishedAfter=None, publishedBefore=None, q=None
                 , channelType=None, eventType=None, location=None, locationRadius=None, regionCode=None, relevanceLanguage=None, safeSearch=None, topicId=None
                 , videoCaption=None, videoCategoryId=None, videoDefinition=None, videoDimension=None, videoDuration=None, videoEmbeddable=None, videoLicense=None, videoPaidProductPlacement=None
                 , videoSyndicated=None, videoType=None):
    """
    Функция для выгрузки характеристик контента YouTube методами его API: search, playlists & playlistItems, videos, commentThreads & comments, channels -- ключевым из которых выступает search
    Причём количество объектов выгрузки максимизируется путём её пересортировки аргументом order и сегментирования по годам
    
    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://developers.google.com/youtube/v3/docs/search/list
             access_token : str
       channelIdForSearch : str
           -- это аналог channelId

              contentType : str
                  -- это аналог type
  
           publishedAfter : str, readable by datetime
          publishedBefore : str, readable by datetime
                        q : str

              channelType : str
                eventType : str
                 location : str
           locationRadius : str
               regionCode : str
        relevanceLanguage : str
               safeSearch : str
                  topicId : str
             videoCaption : str
          videoCategoryId : str
          videoDefinition : str
           videoDimension : str
            videoDuration : str
          videoEmbeddable : str
             videoLicense : str
videoPaidProductPlacement : str
          videoSyndicated : str
                videoType : str
    """
    if (access_token == None) & (channelIdForSearch == None) & (contentType == None) & (publishedAfter == None) & (publishedBefore == None) & (q == None)\
        & (channelType == None) & (eventType == None) & (location == None) & (locationRadius == None) & (regionCode == None) & (relevanceLanguage == None) & (safeSearch == None) & (topicId == None)\
        & (videoCaption == None) & (videoCategoryId == None) & (videoDefinition == None) & (videoDimension == None) & (videoDuration == None) & (videoEmbeddable == None) & (videoLicense == None)\
        & (videoPaidProductPlacement == None) & (videoSyndicated == None) & (videoType == None):
        # print('Пользователь не подал аргументы')
        expiriencedMode = False
    else: expiriencedMode = True

    if expiriencedMode == False:
        print('    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными).'
              , 'Но от пользователя требуется предварительно получить API key для авторизации в API YouTube по ключу (см. примерную видео-инструкцию: https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ).'
              , 'Для получения API key следует создать проект, авторизовать его, подключить к нему API нужного сервиса Google.'
              , 'Проект -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому сервисов Google'
              , 'и применения на этой основе API разных сервисов Google в рамках установленных Гуглом ограничений (the units of quota).'
              , 'Разные уровни авторизации -- это авторизация ключом (представляющим собой код) и полная авторизация (ключ + протокол Google OAuth 2.0, реализующийся в формате файла JSON).'
              , 'Авторизация ключом нужна, чтобы использовать любой метод любого API.'
              , 'Её достаточно, если выполнять действия, которые были бы доступны Вам как пользователю сервисов Google без Вашего входа в аккаунт:'
              , 'посмотреть видео, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления видео, то Вам придётся пройти полную авторизацию.'
              , 'Далее разные API как бы подключаются к проектам (кнопка Enable APIs and servises), используются, затем отключаются (кнопка Disable APIs).'
              , '\n    Квоты одного ключа может не хватить (quota is exceeded) для выгрузки всего предоставляемого ЮТьюбом по запросу пользователя контента.'
              , 'К счастью, использованный ключ ежесуточно восстанавливается ЮТьюбом. скрипт позволяет сохранить промежуточную выгрузку и после восстановления ключа автоматически продолжит её дополнять'
              , 'с момента остановки. В момент остановки появится надпись: "Поскольку ключи закончились, исполнение скрипта завершаю. Подождите сутки для восстановления ключей или подготовьте новый ключ'
              , '-- и запустите скрипт с начала", а исполнение скрипта прервётся. Не пугайтесь, нажмите OK и следуйте этой инструкции.')
    print('    Скрипт нацелен на выгрузку характеристик контента YouTube семью методами его API: search, videos, commentThreads и comments, channels, playlists и playlistItems.'
    , 'Причём количество объектов выгрузки максимизируется путём её пересортировки и сегментирования по годам.'
    , '\n    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.'
    , '\n    Преимущества скрипта перед выгрузкой контента из YouTube вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel.'
    , 'Преимущества скрипта перед выгрузкой контента через непосредственно API YouTube: гораздо быстрее, гораздо большее количество контента с одним и тем же ключом,'
    , 'не требуется тщательно изучать обширную документацию семи методов API YouTube (search, videos, commentThreads и comments, channels, playlists и playlistItems),'
    , 'выстроена логика обрашения к этим методам')
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API YouTube
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
        print('Проверяю наличие файла credentialsYouTube.txt с ключ[ом ами], гипотетически сохранённым[и] при первом запуске скрипта')
        if 'credentialsYouTube.txt' in rootNameS:
            file = open('credentialsYouTube.txt')
            API_keyS = file.read()
            print('Нашёл файл credentialsYouTube.txt; далее буду использовать ключ[и] из него:', API_keyS)
        else:
            print('\n--- НЕ нашёл файл credentialsYouTube.txt . Введите в окно Ваш API key для авторизации в API YouTube по ключу'
                  , '(примерная видео-инструкция, как создать API key, доступна по ссылке https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ).'
                  , 'Для увеличения размера выгрузки желательно создать несколько ключей (пять -- отлично) и ввести их без кавычек через запятую с пробелом'
                  , '\n--- После ввода нажмите Enter')
            while True:
                API_keyS = input()
                if len(API_keyS) != 0:
                    print('-- далее буд[е у]т использован[ы] эт[от и] ключ[и]')
                
                    from randan.tools.textPreprocessing import multispaceCleaner # авторский модуль для предобработки нестандартизированного текста
                    API_keyS = multispaceCleaner(API_keyS)
                    while API_keyS[-1] == ',': API_keyS = API_keyS[:-1] # избавиться от запятых в конце текста
                
                    file = open("credentialsYouTube.txt", "w+") # открыть на запись
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
        
            file = open(f'{rootName}{slash}contentType.txt')
            contentType = file.read()
            file.close()
    
            file = open(f'{rootName}{slash}channelIdForSearch.txt')
            channelIdForSearch = file.read()
            file.close()
    
            file = open(f'{rootName}{slash}q.txt', encoding='utf-8')
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
                  , '\n- было выявлено целевое число записей (totalResults)', targetCount
                  , '\n- скрипт остановился на методе', method)
            if year < int(today[:4]): print('- и на годе (при сегментировани по годам)', year)
            print('- пользователь НЕ определил тип контента' if contentType == '' else  f'- пользователь определил тип контента как "{contentType}"')
            if contentType == 'video':
                print('- пользователь НЕ выбрал конкретный канал для выгрузки видео' if channelIdForSearch == '' else  f'- пользователь выбрал канал с id "{channelIdForSearch}" для выгрузки видео')
            print('- пользователь НЕ сформулировал запрос-фильтр' if q == '' else  f'- пользователь сформулировал запрос-фильтр как "{q}"')
            print('- пользователь НЕ ограничил временнОй диапазон' if yearsRange == '' else  f'- пользователь ограничил временнОй диапазон границами {yearsRange}')
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
# Данные, сохранённые при прошлом запуске скрипта, загружены; их метаданные (q, contentType, yearsRange, stageTarget) будут использоваться при исполнении скрипта
                break
            elif decision == 'R': shutil.rmtree(rootName, ignore_errors=True)

# 2.0.3 Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if temporalName == None: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта')
        print('--- Возможно, Вы располагаете файлом, в котором есть, как минимум, столбец id, и который хотели бы дополнить?'
              , 'Или планируете первичный сбор контента?'
              , '\n--- Если планируете первичный сбор, нажмите Enter'
              , '\n--- Если располагаете файлом, укажите полный путь, включая название файла, и нажмите Enter')
        while True:
            folderFile = input()
            if len(folderFile) == 0:
                folderFile = None # для унификации
                break
            else:
                itemS, error, fileName, folder, slash = files2df.excel2df(folderFile)
                if error != None:
                    if 'No such file or directory' in error:
                        print('Файл:', folder + slash + fileName, '-- не существует; попробуйте, пожалуйста, ещё раз..')
                else: break
            # display(itemS)
# Теперь определены объекты: folder и folderFile (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский), slash

# 2.0.4 Пользовательские настройки запроса к API YouTube
    # Контент: канал или видео? Или вообще плейлист?
        if contentType == None: # если пользователь не подал этот аргумент в рамках experiencedMode
            while True:
                print('--- Если НЕ требуется определить тип контента, нажмите Enter'
                      , ' \n--- Если требуется определить, введите символ: c -- channel, p -- playlist, v -- video -- и нажмите Enter')
                contentType = input()
                if contentType.lower() == '':
                    contentType = ''
                    break
                elif contentType.lower() == 'c':
                    contentType = 'channel'
                    break
                elif contentType.lower() == 'p':
                    contentType = 'playlist'
                    break
                elif contentType.lower() == 'v':
                    contentType = 'video'
                    break
                else:
                    print('--- Вы ввели что-то не то; попробуйте, пожалуйста, ещё раз..')
    
        if (channelIdForSearch == None) & (contentType == 'video'): # если пользователь не подал аргумент channelIdForSearch в рамках experiencedMode
            print('\n--- Вы выбрали тип контента video'
                  , '\n--- Если НЕ предполагается поиск видео в пределах конкретного канала, нажмите Enter'
                  , '\n--- Если предполагается такой поиск, введите id канала, после чего нажмите Enter.'
                  , 'Этот id можете найти либо в URL-адресе интересующего канала,'
                  , 'либо -- если прежде выгружали контент из YouTube -- в столбце "snippet.channelId" выдачи методов search, playlistItems, videos или в столбце "id" метода cannels')
            while True:
                channelIdForSearch = input()
                if len(channelIdForSearch) == 0:
                    channelIdForSearch = None # для унификации
                    break
# --- Если развивать опцию подачи списка каналов
            # elif '.xlsx' in channelIdsForSearch:
            #     channelIdsForSearch, error, fileName, folder, slash = excel2df(channelIdsForSearch)
            #     if error != None:
            #         if 'No such file or directory' in error:
            #             print('Файл:', folder + slash + fileName, '-- не существует; попробуйте, пожалуйста, ещё раз..')
            #     else:
            #         try:
            #             channelIdsForSearch = channelIdsForSearch['snippet.channelId'].to_list()
            #             print('Количество id каналов:', len(channelIdsForSearch), '\n')
            #             break
            #         except KeyError:
            #             errorDescription = sys.exc_info()
            #             if 'snippet.channelId' in str(errorDescription[1]):
            #                 print('В предлагаемом Вами файле, к сожалению, нет требуемого столбца snippet.channelId . Попробуйте ещё раз..')
# ЛАЙФХАК: автоматизированно оперировать многими каналами можно через файл channelIdForSearch директорий _Temporal , в котором подаются стартовые настройки + id каждого из интересующих каналов
# ----------
                else:
                    from randan.tools.textPreprocessing import multispaceCleaner
                    channelIdForSearch = multispaceCleaner(channelIdForSearch)
                    while channelIdForSearch[-1] == ',': channelIdForSearch = channelIdForSearch[:-1] # избавиться от запятых в конце текста
                    channelIdForSearch = channelIdForSearch.split(', ')
                    print('Количество id каналов:', len(channelIdForSearch), '\n')
                    if len(channelIdForSearch) > 1:
                        channelIdForSearch = channelIdForSearch[0]
                        print('В качестве аргумента метода search будет использован ТОЛЬКО первый из id\n')
                    break
        if q == None: # если пользователь не подал этот аргумент в рамках experiencedMode
            print('--- Если НЕ предполагается поиск контента по текстовому запросу-фильтру, нажмите Enter'
                  , '\n--- Если предполагается такой поиск, введите текст запроса-фильтра, который ожидаете в атрибутах и характеристиках'
                  , '(описание, название, теги, категории и т.п.) релевантного YouTube-контента, после чего нажмите Enter')
            if folderFile != None: print('ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла'
                , folderFile
                , 'будут дополнены актуальными данными из выдачи скрипта (возможно появление новых записей и новых столбцов, а также актуализация содержимого столбцов),'
                , 'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла')
            q = input()

    # Ограничения временнОго диапазона
        if (publishedAfter == None) & (publishedBefore == None) & (yearsRange == None): # если пользователь не подал эти аргументы в рамках experiencedMode
            print('\nАлгоритм API Youtube для ограничения временнОго диапазона выдаваемого контента работает со странностями.'
                  , 'Поэтому если требуется конкретный временнОй диапазон, то лучше использовать его НЕ на текущем этапе выгрузки данных,'
                  , 'а на следующем этапе -- предобработки датафрейма с выгруженными данными')
            print('--- Если НЕ требуется задать временнОй диапазон на этапе выгрузки данных, нажмите Enter'
                  , '\n--- Если всё же требуется задать временнОй диапазон, настоятельная рекомендация задать его годами,'
                  , 'а не более мелкими единицами времени. Для задания диапазона введите без кавычек минимальный год диапазона, тире,'
                  , 'максимальный год диапазона (минимум и максимум могут совпадать в такой записи: "год-тот же год") и нажмите Enter')
            while True:
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
        if publishedAfter != None:
            yearMinByUser = int(datetime.strptime(publishedAfter,"%Y-%m-%dT%H:%M:%SZ").strftime('%Y')) # из experiencedMode
            # print('elif start_time != None:', yearMinByUser) # для отладки
        
        if publishedBefore != None:
            yearMaxByUser = int(datetime.strptime(publishedBefore,"%Y-%m-%dT%H:%M:%SZ").strftime('%Y')) # из experiencedMode
            # print('elif end_time != None:', yearMaxByUser) # для отладки
            year = yearMaxByUser
        
        if (yearMinByUser != None) & (yearMaxByUser == None): yearMaxByUser = int(today[:4]) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
        elif (yearMinByUser == None) & (yearMaxByUser != None): yearMaxByUser = 1970 # в случае отсутствия пользовательской нижней временнОй границы при наличии верхней
            
        # print('yearMinByUser', yearMinByUser) # для отладки
        # print('yearMaxByUser', yearMaxByUser) # для отладки

        if (publishedAfter == None) & (yearMinByUser != None): publishedAfter = datetime.datetime(yearMinByUser, 1, 1).isoformat() + 'Z'
        if (publishedBefore == None) & (yearMaxByUser != None): publishedBefore = datetime.datetime(yearMaxByUser, 12, 31).isoformat() + 'Z'

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_YouTube'
    complicatedNamePart += "" if contentType == None else "_" + contentType
    complicatedNamePart += "" if channelIdForSearch == None else "_channelId" + channelIdForSearch
    complicatedNamePart += "" if q == None else "_" + q[50]
    complicatedNamePart += "" if ((yearMinByUser == None) & (yearMaxByUser == None)) else "_" + str(yearMinByUser) + '-' + str(yearMaxByUser)
# print('complicatedNamePart', complicatedNamePart)

# 2.1 Первичный сбор контента методом search
# 2.1.0 Первый заход БЕЗ аргумента order (этап stage = 0)
    stage = 0
    iteration = 0 # номер итерации применения текущего метода
    method = 'search'
    print(f'\nВ скрипте используются следующие аргументы метода {method} API YouTube:'
          , 'channelId, maxResults, order, pageToken, part, publishedAfter, publishedBefore, q, type.'
          , 'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
          , f'Если хотите добавить другие аргументы метода {method} API YouTube, доступные по ссылке https://developers.google.com/youtube/v3/docs/search ,'
          , f'-- можете сделать это внутри метода {method} в чанке 1.1 исполняемого сейчас скрипта')
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')

    if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
        print('\nЗаход на первую страницу выдачи')
        addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                 API_keyS
                                                                 , channelIdForSearch
                                                                 , contentType
                                                                 , goS
                                                                 , iteration
                                                                 , keyOrder
                                                                 , q
                                                                 , publishedAfter=publishedAfter
                                                                 , publishedBefore=publishedBefore
                                                                 , order=None
                                                                 , pageToken=None
                                                                 , year=None
                                                                 , channelType=channelType
                                                                 , eventType=eventType
                                                                 , location=location
                                                                 , locationRadius=locationRadius
                                                                 , regionCode=regionCode
                                                                 , relevanceLanguage=relevanceLanguage
                                                                 , topicId=topicId
                                                                 , safeSearch=safeSearch
                                                                 , videoCaption=videoCaption
                                                                 , videoCategoryId=videoCategoryId
                                                                 , videoDefinition=videoDefinition
                                                                 , videoDimension=videoDimension
                                                                 , videoDuration=videoDuration
                                                                 , videoEmbeddable=videoEmbeddable
                                                                 , videoLicense=videoLicense
                                                                 , videoPaidProductPlacement=videoPaidProductPlacement
                                                                 , videoType=videoType
                                                                 , videoSyndicated=videoSyndicated
                                                                )
        targetCount = response['pageInfo']['totalResults']
        itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
        # display('itemS', itemS) # для отладки
    
        print('  Проход по всем следующим страницам с выдачей          ')
        while 'nextPageToken' in response.keys():
            pageToken = response['nextPageToken']
            addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                     API_keyS
                                                                     , channelIdForSearch
                                                                     , contentType
                                                                     , goS
                                                                     , iteration
                                                                     , keyOrder
                                                                     , q
                                                                     , publishedAfter=publishedAfter
                                                                     , publishedBefore=publishedBefore
                                                                     , order=order
                                                                     , pageToken=pageToken
                                                                     , year=None
                                                                     , channelType=channelType
                                                                     , eventType=eventType
                                                                     , location=location
                                                                     , locationRadius=locationRadius
                                                                     , regionCode=regionCode
                                                                     , relevanceLanguage=relevanceLanguage
                                                                     , topicId=topicId
                                                                     , safeSearch=safeSearch
                                                                     , videoCaption=videoCaption
                                                                     , videoCategoryId=videoCategoryId
                                                                     , videoDefinition=videoDefinition
                                                                     , videoDimension=videoDimension
                                                                     , videoDuration=videoDuration
                                                                     , videoEmbeddable=videoEmbeddable
                                                                     , videoLicense=videoLicense
                                                                     , videoPaidProductPlacement=videoPaidProductPlacement
                                                                     , videoType=videoType
                                                                     , videoSyndicated=videoSyndicated
                                                                    )
            itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
        print('  Искомых объектов', targetCount
              , ', а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))
    elif stage < stageTarget:
        print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{rootName}"')

# 2.1.1 Цикл для прохода по значениям аргумента order, внутри которых проход по всем страницам выдачи (этап stage = 1)
    stage = 1
    orderS = ['date', 'rating', 'title', 'videoCount', 'viewCount']
    if stage >= stageTarget: # eсли НЕТ файла с id и нет временного файла stage.txt с указанием пропустить этап
        if len(itemS) < targetCount:
        # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ включения каких-либо значений аргумента order (в т.ч. вообще БЕЗ них)
            print('Проход по значениям аргумента order, внутри которых проход по всем страницам выдачи')
            for order in orderS:
                addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                         API_keyS
                                                                         , channelIdForSearch
                                                                         , contentType
                                                                         , goS
                                                                         , iteration
                                                                         , keyOrder
                                                                         , q
                                                                         , publishedAfter=publishedAfter
                                                                         , publishedBefore=publishedBefore
                                                                         , order=order
                                                                         , pageToken=None
                                                                         , year=year
                                                                         , channelType=channelType
                                                                         , eventType=eventType
                                                                         , location=location
                                                                         , locationRadius=locationRadius
                                                                         , regionCode=regionCode
                                                                         , relevanceLanguage=relevanceLanguage
                                                                         , topicId=topicId
                                                                         , safeSearch=safeSearch
                                                                         , videoCaption=videoCaption
                                                                         , videoCategoryId=videoCategoryId
                                                                         , videoDefinition=videoDefinition
                                                                         , videoDimension=videoDimension
                                                                         , videoDuration=videoDuration
                                                                         , videoEmbeddable=videoEmbeddable
                                                                         , videoLicense=videoLicense
                                                                         , videoPaidProductPlacement=videoPaidProductPlacement
                                                                         , videoType=videoType
                                                                         , videoSyndicated=videoSyndicated
                                                                        )
                itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    
                print('  Проход по всем следующим страницам с выдачей с тем же значением аргумента order:', order, '          ')
                while ('nextPageToken' in response.keys()) & (len(itemS) < targetCount) & (len(response["items"]) > 0):
                # -- второе условие -- для остановки алгоритма, если все искомые объекты найдены
                    # БЕЗ какой-то из следующих страниц (в т.ч. вообще БЕЗ них)
                    # третье условие -- для остановки алгоритма, если предыдущая страница выдачи содержит 0 объектов
    
                    pageToken = response['nextPageToken']
                    # print('pageToken', pageToken)
                    addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                             API_keyS
                                                                             , channelIdForSearch
                                                                             , contentType
                                                                             , goS
                                                                             , iteration
                                                                             , keyOrder
                                                                             , q
                                                                             , publishedAfter=publishedAfter
                                                                             , publishedBefore=publishedBefore
                                                                             , order=order
                                                                             , pageToken=pageToken
                                                                             , year=None
                                                                             , channelType=channelType
                                                                             , eventType=eventType
                                                                             , location=location
                                                                             , locationRadius=locationRadius
                                                                             , regionCode=regionCode
                                                                             , relevanceLanguage=relevanceLanguage
                                                                             , topicId=topicId
                                                                             , safeSearch=safeSearch
                                                                             , videoCaption=videoCaption
                                                                             , videoCategoryId=videoCategoryId
                                                                             , videoDefinition=videoDefinition
                                                                             , videoDimension=videoDimension
                                                                             , videoDuration=videoDuration
                                                                             , videoEmbeddable=videoEmbeddable
                                                                             , videoLicense=videoLicense
                                                                             , videoPaidProductPlacement=videoPaidProductPlacement
                                                                             , videoType=videoType
                                                                             , videoSyndicated=videoSyndicated
                                                                            )
                    itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
            print('  Искомых объектов', targetCount, ', а найденных С включением аргумента order:', len(itemS))
        else:
            print('Все искомые объекты найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
    elif stage < stageTarget:
        print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{rootName}"')

# 2.1.2 Этап сегментирования по годам (stage = 2)
    stage = 2 
    if stage >= stageTarget: # eсли НЕТ файла с id и нет временного файла stage.txt с указанием пропустить этап
        if len(itemS) < targetCount:
        # для остановки алгоритма, если все искомые объекты найдены БЕЗ включения каких-либо значений аргумента order (в т.ч. вообще БЕЗ них)
            print('Увы'
                  , f'\nЧисло найденных объектов: {len(itemS)} -- менее числа искомых: {targetCount}')
            print('\n--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам, просто нажмите Enter, но учтите, что поиск может занять минуты и даже часы'
                  , '\n--- Если НЕ хотите, введите любой символ и нажмите Enter')
            if len(input()) == 0:
                print('Внутри каждого года прохожу по значениям аргумента order, внутри которых прохожу по всем страницам выдачи')
                goC = True
# ********** из фрагмента 2.1.0 + условие для goC
                while (len(itemS) < targetCount) & (goC):
                    print(f'  Для года {year} заход на первую страницу выдачи БЕЗ аргумента order')
                    addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                             API_keyS
                                                                             , channelIdForSearch
                                                                             , contentType
                                                                             , goS
                                                                             , iteration
                                                                             , keyOrder
                                                                             , q
                                                                             , publishedAfter = f'{year}-01-01T00:00:00Z'
                                                                             , publishedBefore = f'{year + 1}-01-01T00:00:00Z'
                                                                             , order=None
                                                                             , pageToken=None
                                                                             , year=year
                                                                             , channelType=channelType
                                                                             , eventType=eventType
                                                                             , location=location
                                                                             , locationRadius=locationRadius
                                                                             , regionCode=regionCode
                                                                             , relevanceLanguage=relevanceLanguage
                                                                             , topicId=topicId
                                                                             , safeSearch=safeSearch
                                                                             , videoCaption=videoCaption
                                                                             , videoCategoryId=videoCategoryId
                                                                             , videoDefinition=videoDefinition
                                                                             , videoDimension=videoDimension
                                                                             , videoDuration=videoDuration
                                                                             , videoEmbeddable=videoEmbeddable
                                                                             , videoLicense=videoLicense
                                                                             , videoPaidProductPlacement=videoPaidProductPlacement
                                                                             , videoType=videoType
                                                                             , videoSyndicated=videoSyndicated
                                                                            )
                    if len(addItemS) == 0:
                        print(f'\n--- Первая страница выдачи БЕЗ аргумента order для года {year} -- пуста'
                              , '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать предыдущий год, просто нажмите Enter'
                              , '\n--- Если хотите, введите любой символ и нажмите Enter')
                        if len(input()) == 0:
                            goC = False
                            break
                    itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    
                    print(f'    Проход по всем следующим страницам с выдачей для года {year} БЕЗ аргумента order')
                    while 'nextPageToken' in response.keys():
                        pageToken = response['nextPageToken']
                        addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                 API_keyS
                                                                                 , channelIdForSearch
                                                                                 , contentType
                                                                                 , goS
                                                                                 , iteration
                                                                                 , keyOrder
                                                                                 , q
                                                                                 , publishedAfter = f'{year}-01-01T00:00:00Z'
                                                                                 , publishedBefore = f'{year + 1}-01-01T00:00:00Z'
                                                                                 , order=None
                                                                                 , pageToken=pageToken
                                                                                 , year=year
                                                                                 , channelType=channelType
                                                                                 , eventType=eventType
                                                                                 , location=location
                                                                                 , locationRadius=locationRadius
                                                                                 , regionCode=regionCode
                                                                                 , relevanceLanguage=relevanceLanguage
                                                                                 , topicId=topicId
                                                                                 , safeSearch=safeSearch
                                                                                 , videoCaption=videoCaption
                                                                                 , videoCategoryId=videoCategoryId
                                                                                 , videoDefinition=videoDefinition
                                                                                 , videoDimension=videoDimension
                                                                                 , videoDuration=videoDuration
                                                                                 , videoEmbeddable=videoEmbeddable
                                                                                 , videoLicense=videoLicense
                                                                                 , videoPaidProductPlacement=videoPaidProductPlacement
                                                                                 , videoType=videoType
                                                                                 , videoSyndicated=videoSyndicated
                                                                                )
                        itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    
                    print(f'    Искомых объектов в году {year}: {targetCount},'
                          , 'а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))      
# ********** из фрагмента 2.1.1
                    if len(itemS) < targetCount:
                        print(f'  Для года {year} проход по значениям аргумента order,'
                              , 'внутри которых проход по всем страницам выдачи')
                        for order in orderS:
                            addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                     API_keyS
                                                                                     , channelIdForSearch
                                                                                     , contentType
                                                                                     , goS
                                                                                     , iteration
                                                                                     , keyOrder
                                                                                     , q
                                                                                     , publishedAfter = f'{year}-01-01T00:00:00Z'
                                                                                     , publishedBefore = f'{year + 1}-01-01T00:00:00Z'
                                                                                     , order=order
                                                                                     , pageToken=None
                                                                                     , year=year
                                                                                     , channelType=channelType
                                                                                     , eventType=eventType
                                                                                     , location=location
                                                                                     , locationRadius=locationRadius
                                                                                     , regionCode=regionCode
                                                                                     , relevanceLanguage=relevanceLanguage
                                                                                     , topicId=topicId
                                                                                     , safeSearch=safeSearch
                                                                                     , videoCaption=videoCaption
                                                                                     , videoCategoryId=videoCategoryId
                                                                                     , videoDefinition=videoDefinition
                                                                                     , videoDimension=videoDimension
                                                                                     , videoDuration=videoDuration
                                                                                     , videoEmbeddable=videoEmbeddable
                                                                                     , videoLicense=videoLicense
                                                                                     , videoPaidProductPlacement=videoPaidProductPlacement
                                                                                     , videoType=videoType
                                                                                     , videoSyndicated=videoSyndicated
                                                                                    )
                            itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    
                            print(f'    Для года {year} проход по всем следующим страницам с выдачей'
                                  , f'с тем же значением аргумента order:', order)
                            while ('nextPageToken' in response.keys()) & (len(itemS) < targetCount) & (len(response["items"]) > 0):
                                pageToken = response['nextPageToken']
                                addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                         API_keyS
                                                                                         , channelIdForSearch
                                                                                         , contentType
                                                                                         , goS
                                                                                         , iteration
                                                                                         , keyOrder
                                                                                         , q
                                                                                         , publishedAfter = f'{year}-01-01T00:00:00Z'
                                                                                         , publishedBefore = f'{year + 1}-01-01T00:00:00Z'
                                                                                         , order=order
                                                                                         , pageToken=pageToken
                                                                                         , year=year
                                                                                         , channelType=channelType
                                                                                         , eventType=eventType
                                                                                         , location=location
                                                                                         , locationRadius=locationRadius
                                                                                         , regionCode=regionCode
                                                                                         , relevanceLanguage=relevanceLanguage
                                                                                         , topicId=topicId
                                                                                         , safeSearch=safeSearch
                                                                                         , videoCaption=videoCaption
                                                                                         , videoCategoryId=videoCategoryId
                                                                                         , videoDefinition=videoDefinition
                                                                                         , videoDimension=videoDimension
                                                                                         , videoDuration=videoDuration
                                                                                         , videoEmbeddable=videoEmbeddable
                                                                                         , videoLicense=videoLicense
                                                                                         , videoPaidProductPlacement=videoPaidProductPlacement
                                                                                         , videoType=videoType
                                                                                         , videoSyndicated=videoSyndicated
                                                                                        )
                                itemS = dfsProcessing(complicatedNamePart, fileFormatChoice, addItemS, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
                        print('    Искомых объектов', targetCount
                              , ', а найденных с добавлением сегментирования по году (год'
                              , year
                              , ') и включением аргумента order:', len(itemS))
                    else:
                        print('  Все искомые объекты в году', year, 'найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
                    year -= 1
                    if yearMinByUser != None: # если пользователь ограничил временнОй диапазон            
                        if (year) <= yearMinByUser:
                            goC = False
                            print(f'\nЗавершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser} (с точностью до года)')
    elif stage < stageTarget:
        print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{today}{complicatedNamePart}_Temporal"')

# 2.1.3 Экспорт выгрузки метода search и опциональное завершение скрипта
    df2file.df2fileShell(complicatedNamePart, itemS, fileFormatChoice, method, today)

    print('Выгрузка метода search содержит НЕ ВСЕ доступные для выгрузки из API YouTube характеристки контента'
          , '\n--- Если хотите выгрузить дополнительные характеристики (ссылки для ознакомления с ними появятся ниже), нажмите Enter'
          , '\n--- Если НЕ хотите их выгрузить, введите любой символ и нажмите Enter. Тогда исполнение скрипта завершится')
    
    if len(input()) > 0:
        print('Скрипт исполнен')
        if os.path.exists(rootName):
            print('Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы,'
                  , 'УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта')
            shutil.rmtree(rootName, ignore_errors=True)
    
        warnings.filterwarnings("ignore")
        print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'
              , '\nМодуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
        sys.exit()

# 2.2 Выгрузка дополнительных характеристик и контента методами playlists и playlistItems, videos, commentThreads и comments, channels
# 2.2.0 Этап stage = 3
    stage = 3

# 2.2.1 Выгрузка дополнительных характеристик плейлистов
    snippetContentType = 'playlist'
    if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # если в выдаче есть плейлисты
        playlistIdS = itemS[itemS['id.kind'] == f'youtube#{snippetContentType}']
        playlistIdS =\
            playlistIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in playlistIdS.columns else playlistIdS['id'].to_list()
        # playlistIdS = playlistIdS[:5] # для отладки
        # print(playlistIdS)
    
        method = 'playlists'
        print('В скрипте используются следующие аргументы метода', method, 'API YouTube:'
              , 'part=["snippet", "contentDetails", "localizations", "status"], id, maxResults .'
              , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
              , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
              , 'https://developers.google.com/youtube/v3/docs/playlists')
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
        bound = 0
        goC = True
        iteration = 0 # номер итерации применения текущего метода
        portion = 50
        playlistS = pandas.DataFrame()
        print('\nПроход порциями по', portion, 'плейлистов для выгрузки характеристик плейлистов')
        while (bound < len(playlistIdS)) & (goC):
            try:
                youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
                response = youtube.playlists().list(part=["snippet"
                                                         , "contentDetails"
                                                         , "localizations"
                                                         , "status"]
                                                    , id=playlistIdS[bound:bound + 50]
                                                    ).execute()
                iterationVisualization(playlistIdS, portion) # для визуализации процесса через итерации
                bound += 50
                iteration += 1
                addPlaylistS = pandas.json_normalize(response['items'])
                playlistS = dfsProcessing(complicatedNamePart, fileFormatChoice, addPlaylistS, playlistS, playlistS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    
            except googleapiclient.errors.HttpError:
                errorDescription = sys.exc_info()
                goC, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder)
        
            except IndexError:
                errorDescription = sys.exc_info()
                goC, goS = indexError(errorDescription)
    
        method = 'playlistItems'
        print('\nВ скрипте используются следующие аргументы метода', method, 'API YouTube:'
              , 'part=["snippet"], playlistId, maxResults .'
              , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
              , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
              , 'https://developers.google.com/youtube/v3/docs/playlists')
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
        goC = True
        iteration = 0 # номер итерации применения текущего метода
        playlistVideoChannelS = pandas.DataFrame()
        portion = 1
        print('\nПроход по плейлистам для выгрузки id видео, составляющих плейлисты, и каналов, к которым они принадлежат')
        for playlistId in playlistIdS:
            try:
                youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
                response = youtube.playlistItems().list(part=["snippet"]
                                                        , playlistId=playlistId
                                                        ).execute()
                iterationVisualization(playlistIdS, portion) # для визуализации процесса через итерации
                iteration += 1
                addPlaylistVideoChannelS = pandas.json_normalize(response['items'])
                playlistVideoChannelS = dfsProcessing(complicatedNamePart, fileFormatChoice, addPlaylistVideoChannelS, playlistVideoChannelS, playlistVideoChannelS
                                                      , goS, method, q, slash, stage, targetCount, today, year, yearsRange)
            except googleapiclient.errors.HttpError:
                errorDescription = sys.exc_info()
                goC, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder)
    
            except IndexError:
                errorDescription = sys.exc_info()
                goC, goS = indexError(errorDescription)
    
        # Перечислить сначала id всех составляющих каждый плейлист видео через запятую и записать в ячейку,
            # затем id всех канадов, к которым относятся составляющие каждый плейлист видео, через запятую и записать в ячейку
        # display('playlistVideoChannelS', playlistVideoChannelS) # для отладки
        for playlistId in playlistIdS:
            for column in ['snippet.resourceId.videoId', 'snippet.videoOwnerChannelId']:
                playlistVideoChannelS_snippet = playlistVideoChannelS[playlistVideoChannelS[column].notna()]
                playlistS.loc[playlistS[playlistS['id'] == playlistId].index[0], column] =\
                    ', '.join(playlistVideoChannelS_snippet[playlistVideoChannelS_snippet['snippet.playlistId'] == playlistId][column].to_list())
        # display(playlistS)
        df2file.df2fileShell(complicatedNamePart, playlistS, fileFormatChoice, method, today)

# 2.2.2 Выгрузка дополнительных характеристик видео
    snippetContentType = 'video'
    videoS = pandas.DataFrame() # не внутри условия, чтобы следующий чанк исполнялся даже при невыполнении этого условия
    if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # если в выдаче есть видео
        iteration = 0 # номер итерации применения текущего метода
        method = 'videos'
        print('\nВ скрипте используются следующие аргументы метода', method, 'API YouTube:'
              , 'part=["snippet", "contentDetails", "localizations", "statistics", "status", "topicDetails"], id, maxResults .'
              , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
              , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
              , 'https://developers.google.com/youtube/v3/docs/videos')
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
    
        videoIdS = itemS[itemS['id.kind'] == f'youtube#{snippetContentType}']
        videoIdS = videoIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in videoIdS.columns else videoIdS['id'].to_list()
        # videoIdS = videoIdS[:5] # для отладки
        # print(videoIdS)

# ********** Дополнение списка id видео из itemS списком id видео из playlistS
        if len(playlistS) > 0:
            print('\n--- Если хотите дополнить спискок id видео, выгруженных методом search, списком id видео, составляющих выгруженные плейлисты'
                  , 'просто нажмите Enter (это увеличит совокупность изучаемых видео)'
                  , '\n--- Если НЕ хотите дополнить спискок, введите любой символ и нажмите Enter')
            if len(input()) == 0:
            
                # Список списков, каждый из которых соответствует одному плейлисту
                playlistVideoId_list = playlistS['snippet.resourceId.videoId'].str.split(', ').to_list()
                # print('playlistVideoId_list:', playlistVideoId_list) # для отладки
            
                playlistVideoIdS = []
                for playlistVideoIdSnippet in playlistVideoId_list:
                    playlistVideoIdS.extend(playlistVideoIdSnippet)
            
                videoIdS.extend(playlistVideoIdS)
                videoIdS = list(dict.fromkeys(videoIdS))
    
        print('Проход порциями по 50 видео')
        bound = 0
        goC = True
        while (bound < len(videoIdS)) & (goC):
        # while (bound < 100) & (goC): # для отладки
            try:
                youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
                response = youtube.videos().list(part=["snippet"
                                                      , "contentDetails"
                                                      , "localizations"
                                                      , "statistics"
                                                      , "status"
                                                      , "topicDetails"]
                                                , id=videoIdS[bound:bound + 50]
                                                , maxResults=50
                                                ).execute()
                # Для визуализации процесса
                print('  Порция №', iteration + 1, 'из', round(len(videoIdS) / 50, 0), '; сколько в порции наблюдений?', len(response['items']), end='\r')
                bound += 50
                iteration += 1
    
                addVideoS = pandas.json_normalize(response['items'])
                videoS = dfsProcessing(complicatedNamePart, fileFormatChoice, addVideoS, videoS, videoS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    
            except googleapiclient.googleapiclient.errors.HttpError:
                errorDescription = sys.exc_info()
                print(errorDescription[1])
                print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
                keyOrder += 1
    
            except IndexError:
                errorDescription = sys.exc_info()
                print(errorDescription[1])
                print('Поскольку ключи закончились,'
                      , f'сохраняю выгруженный контент и текущий этап поиска в директорию "{complicatedNamePart} Temporal"')     
                if not os.path.exists(f'{complicatedNamePart} Temporal'):
                        os.makedirs(f'{complicatedNamePart} Temporal')
                        print(f'Директория "{complicatedNamePart} Temporal" создана')
                else:
                    print(f'Директория "{complicatedNamePart} Temporal" существует')
                saveSettings(channelIdForSearch, complicatedNamePart, contentType, itemS, method, q, slash, stage, targetCount, year, yearsRange)
                goC = False

# ********** categoryId
        # Взять столбец snippet.categoryId, удалить из него дубликаты кодов категорий и помеcтить уникальные коды в список
        uniqueCategorieS = videoS['snippet.categoryId'].drop_duplicates().to_list()
        # print('\nУникальные коды категорий в базе:', uniqueCategorieS, '\nЧисло уникальных категорий в базе:', len(uniqueCategorieS))
        try:
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            response = youtube.videoCategories().list(part='snippet', id=uniqueCategorieS).execute()
    
        except errors.HttpError:
            errorDescription = sys.exc_info()
            print(errorDescription[1])
            print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
            keyOrder += 1
    
        except IndexError:
            errorDescription = sys.exc_info()
            print(errorDescription[1])
            print('Поскольку ключи закончились,'
                  , f'сохраняю выгруженный контент и текущий этап поиска в директорию "{complicatedNamePart} Temporal"')     
            if not os.path.exists(f'{complicatedNamePart} Temporal'):
                    os.makedirs(f'{complicatedNamePart} Temporal')
                    print(f'Директория "{complicatedNamePart} Temporal" создана')
            else:
                print(f'Директория "{complicatedNamePart} Temporal" существует')
            saveSettings(complicatedNamePart, contentType, itemS, method, q, slash, stage, targetCount, year, yearsRange)
    
        # Оформить как датафрейм id категорий из списка uniqueCategorieS и их расшифровки
        categoryNameS = pandas.json_normalize(response['items'])
    
        # Заменить индексы датафрейма с расшифровками значениями столбца id
        categoryNameS.index = categoryNameS['id'].to_list()
    
        # Добавить расшифровки категорий в новый столбец categoryName датафрейма с видео
        for row in categoryNameS.index:
            videoS.loc[videoS['snippet.categoryId'] == row, 'categoryName'] = categoryNameS['snippet.title'][row]
    
        df2file.df2fileShell(complicatedNamePart, videoS, fileFormatChoice, method, today)
        commentS = pandas.DataFrame() # не в следующем ченке, чтобы иметь возможность перезапускать его, не затирая промежуточный результат выгрузки

# 2.2.3 Выгрузка комментариев к видео
    if len(videoS) > 0:
        print('\n--- Если хотите выгрузить комментарии к видео (в отдельный файл),'
              , f'содержащимся в файле "{today}{complicatedNamePart} {method}.xlsx" директории "{today}{complicatedNamePart}",'
              , 'просто нажмите Enter'
              , '\n--- Если НЕ хотите выгрузить комментарии, введите любой символ и нажмите Enter')
        if len(input()) == 0:
        
# ********** commentS
            # commentS = pandas.DataFrame() # фрагмент вынесен в предыдущий ченк, чтобы иметь возможность перезапускать этот чанк,
            # не затирая промежуточный результат выгрузки    
            maxResults = 100
            method = 'commentThreads'
            part = 'id, replies, snippet'
            problemVideoIdS = pandas.DataFrame()
            print('\nВ скрипте используются следующие аргументы метода', method, 'API YouTube:'
                  , 'part=["snippet", "id", "replies"], maxResults, videoId .'
                  , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
                  , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
                  , 'https://developers.google.com/youtube/v3/docs/commentThreads')
            if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
        
            # Переназначить объект videoIdS для целей текущего чанка
            videoIdS = videoS[videoS['statistics.commentCount'].notna()]
            videoIdS = videoIdS[videoIdS['statistics.commentCount'].astype(int) > 0]
            videoIdS = videoIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in videoIdS.columns else videoIdS['id'].to_list()
            print('Число видео с комментариями:', len(videoIdS))
        
            print('\nВыгрузка родительских (topLevel) комментариев')
            for videoId in videoIdS:
            # for videoId in videoS['id'][4576:]: # для отладки
                # print('videoId', videoId)
                page = 0 # номер страницы выдачи
                addCommentS, errorDescription, goS, keyOrder, page, pageToken, problemVideoId =\
                    downloadComments(API_keyS, goS, videoId, videoIdS, keyOrder, maxResults, method, page, None, part)
                commentS = dfsProcessing(complicatedNamePart, fileFormatChoice, addCommentS, commentS, commentS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
                if errorDescription != None: problemVideoIdS.loc[problemVideoId, 'errorDescription'] = errorDescription
                while pageToken != None:
                    addCommentS, errorDescription, goS, keyOrder, page, pageToken, problemItemId =\
                        downloadComments(API_keyS, goS, videoId, videoIdS, keyOrder, maxResults, method, page, pageToken, part)
                    commentS = dfsProcessing(complicatedNamePart, fileFormatChoice, addCommentS, commentS, commentS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
                    if errorDescription != None: problemVideoIdS.loc[problemVideoId, 'errorDescription'] = errorDescription
            commentS = commentS.drop(['kind', 'etag', 'id', 'snippet.channelId', 'snippet.videoId'], axis=1) # т.к. дублируются содержательно
            commentS = prefixDropper(commentS)
            df2file.df2fileShell(complicatedNamePart, commentS, fileFormatChoice, 'commentS', today)

# ********** replieS
            print('\nПроход по строкам всех родительских (topLevel) комментариев, имеющих ответы')
            replieS = pandas.DataFrame()
            for row in tqdm(commentS[commentS['snippet.totalReplyCount'] > 0].index):
                addReplieS = pandas.json_normalize(commentS['replies.comments'][row])
        
                # Записать разницу между ожданиями и реальностью в новый столбец `Недостача_ответов`
                commentS.loc[row, 'Недостача_ответов'] = commentS['snippet.totalReplyCount'][row] - len(addReplieS)
            
                replieS = pandas.concat([replieS, addReplieS]).reset_index(drop=True)
        
            replieS.loc[:, 'snippet.totalReplyCount'] = 0
            replieS.loc[:, 'Недостача_ответов'] = 0
            replieS = prefixDropper(replieS)
            df2file.df2fileShell(complicatedNamePart, replieS, fileFormatChoice, 'replieS', today)
        
            commentReplieS = commentS.copy() # копия датафрейма c родительскими (topLevel) комментариями -- основа будущего общего датафрейма
            # Найти столбцы, совпадающие для датафреймов c родительскими (topLevel) комментариями и с комментариями-ответами
            mutualColumns = []
            for column in commentReplieS.columns:
                if column in replieS.columns:
                    mutualColumns.append(column)
    
# ********** commentReplieS
            # Оставить только совпадающие столбцы датафреймов с родительскими (topLevel) комментариями и с комментариями-ответами
            commentReplieS = commentReplieS[mutualColumns]
            replieS = replieS[mutualColumns]
            commentReplieS = dfsProcessing(complicatedNamePart, fileFormatChoice, replieS, commentReplieS, commentReplieS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
            method = 'comments'
            part = 'id, snippet'
            textFormat = 'plainText' # = 'html' по умолчанию
            problemCommentIdS = pandas.DataFrame()
            replieS = pandas.DataFrame() # зачем? См. этап 4.2 ниже
            print('\nВ скрипте используются следующие аргументы метода', method, 'API YouTube:'
                  , 'part=["snippet", "id"], maxResults, parentId, textFormat .'
                  , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
                  , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
                  , 'https://developers.google.com/youtube/v3/docs/commentThreads')
            if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
            print('Проход по id всех родительских (topLevel) комментариев с недостачей ответов для выгрузки этих ответов')
            commentIdS = commentReplieS['id'][commentReplieS['Недостача_ответов'] > 0]
            for commentId in tqdm(commentIdS):
                page = 0 # номер страницы выдачи
                addReplieS, errorDescription, goS, keyOrder, page, pageToken, problemCommentId =\
                    downloadComments(API_keyS, goS, commentId, commentIdS, keyOrder, maxResults, method, '', None, part)
                if errorDescription != None: problemCommentIdS.loc[problemCommentId, 'errorDescription'] = errorDescription
                replieS = dfsProcessing(complicatedNamePart, fileFormatChoice, addReplieS, replieS, replieS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
                while pageToken != None:
                    addReplieS, errorDescription, goS, keyOrder, page, pageToken, problemCommentId =\
                        downloadComments(API_keyS, goS, commentId, commentIdS, keyOrder, maxResults, method, '', pageToken, part)
                    if errorDescription != None: problemCommentIdS.loc[problemCommentId, 'errorDescription'] = errorDescription
                    replieS = dfsProcessing(complicatedNamePart, fileFormatChoice, addReplieS, replieS, replieS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
            print('Ответов выгружено', len(replieS)
                  , '; проблемные родительские (topLevel) комментарии:', problemCommentIdS if len(problemCommentIdS) > 0  else 'отсутствуют')
        
            # Для совместимости датафреймов добавить столбцы`snippet.totalReplyCount` и `Недостача_ответов`
            replieS.loc[:, 'snippet.totalReplyCount'] = 0
            replieS.loc[:, 'Недостача_ответов'] = 0
        
            # Удалить столбец `snippet.parentId`, т.к. и из столбца `id` всё ясно
            replieS = replieS.drop('snippet.parentId', axis=1)
        
            commentReplieS = dfsProcessing(complicatedNamePart, fileFormatChoice, replieS, commentReplieS, commentReplieS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
            df2file.df2fileShell(complicatedNamePart, commentReplieS, fileFormatChoice, 'commentReplieS', today)

# 2.2.4 Выгрузка дополнительных характеристик каналов
    snippetContentType = 'channel'
    if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # если в выдаче есть каналы
        channelS = pandas.DataFrame()
        iteration = 0 # номер итерации применения текущего метода
        method = 'channels'
        print('\nВ скрипте используются следующие аргументы метода', method, 'API YouTube:'
              , 'part=["snippet", "brandingSettings", "contentDetails", "id", "localizations", "statistics", "status", "topicDetails"], id, maxResults .'
              , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
              , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
              , 'https://developers.google.com/youtube/v3/docs/channels')
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
    
        channelIdS = itemS[itemS['id.kind'] == f'youtube#{snippetContentType}']
        channelIdS =\
            channelIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in channelIdS.columns else channelIdS['id'].to_list()
        # channelIdS = channelIdS[:5] # для отладки
        # print(channelIdS)

# ********** Дополнение списка id каналов из itemS списком id каналов из videoS
        if len(videoS) > 0:
            print('--- Если хотите дополнить спискок id каналов, выгруженных методом search, списком id каналов, к которым относятся выгруженные видео'
                  , 'просто нажмите Enter (это увеличит совокупность изучаемых каналов)'
                  , '\n--- Если НЕ хотите дополнить спискок, введите любой символ и нажмите Enter')
            if len(input()) == 0:
                channelIdS.extend(videoS['snippet.channelId'].to_list())
                channelIdS = list(dict.fromkeys(channelIdS))

# ********** Дополнение списка id каналов из itemS списком id каналов из playlistS
        if len(playlistS) > 0:
            print('--- Если хотите дополнить спискок id видео, выгруженных методом search, списком id видео, составляющих выгруженные плейлисты'
                  , 'просто нажмите Enter (это тоже увеличит совокупность изучаемых каналов)'
                  , '\n--- Если НЕ хотите дополнить спискок, введите любой символ и нажмите Enter')
            if len(input()) == 0:
            
                # Список списков, каждый из которых соответствует одному плейлисту
                playlistChannelId_list = playlistS['snippet.videoOwnerChannelId'].str.split(', ').to_list()
            
                playlistChannelIdS = []
                for snippet in playlistChannelId_list:
                    playlistChannelIdS.extend(snippet)
                channelIdS.extend(playlistChannelIdS)
                channelIdS = list(dict.fromkeys(channelIdS))
        print('Проход порциями по 50 каналов')
        bound = 0
        goC = True
        while (bound < len(channelIdS)) & (goC):
        # while (bound < 100) & (goC): # для отладки
            try:
                youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
                response = youtube.channels().list(part=["snippet"
                                                      , "brandingSettings"
                                                      , "contentDetails"
                                                      , "id"
                                                      , "localizations"
                                                      , "statistics"
                                                      , "status"
                                                      , "topicDetails"]
                                                , id=channelIdS[bound:bound + 50]
                                                , maxResults=50
                                                ).execute()
                # Для визуализации процесса
                print('  Порция №', iteration + 1, 'из', round(len(channelIdS) / 50, 0)
                      , '; сколько в порции наблюдений?', len(response['items']), '          ', end='\r')
                bound += 50
                iteration += 1
    
                addСhannelS = pandas.json_normalize(response['items'])
                channelS = dfsProcessing(complicatedNamePart, fileFormatChoice, addСhannelS, channelS, channelS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
            except googleapiclient.errors.HttpError:
                errorDescription = sys.exc_info()
                print(errorDescription[1])
                print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
                keyOrder += 1
    
            except IndexError:
                errorDescription = sys.exc_info()
                print(errorDescription[1])
                print('Поскольку ключи закончились,'
                      , f'сохраняю выгруженный контент и текущий этап поиска в директорию "{complicatedNamePart} Temporal"')     
                if not os.path.exists(f'{complicatedNamePart} Temporal'):
                        os.makedirs(f'{complicatedNamePart} Temporal')
                        print(f'Директория "{complicatedNamePart} Temporal" создана')
                else:
                    print(f'Директория "{complicatedNamePart} Temporal" существует')
                saveSettings(channelIdForSearch, complicatedNamePart, contentType, itemS, method, q, slash, stage, targetCount, year, yearsRange)
                goC = False
        df2file.df2fileShell(complicatedNamePart, channelS, fileFormatChoice, method, today)

# 2.2.5 Экспорт выгрузки метода search и финальное завершение скрипта
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

# https://stackoverflow.com/questions/30475309/get-youtube-trends-v3-country-wise-in-json -- про тренды
