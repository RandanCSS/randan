#!/usr/bin/env python
# coding: utf-8


# In[ ]:


# # 0 Активировать требуемые для работы скрипта модули и пакеты + пререквизиты


# In[ ]:


# В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
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
        from tqdm import tqdm
        import os, pandas, re, shutil, time, warnings
        import googleapiclient.discovery as api
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
            print('Пакет', module, 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,',
                  'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
            break

# In[ ]:


# # 1 Авторские функции


# In[ ]:


# 1.0 для метода search из API YouTube, помогающая работе с ключами
def bigSearch(
              API_keyS,
              channelIdForSearch, # согласно документации API YouTube, подать можно лишь один channelId
              channelType,
              contentType,
              eventType,
              iteration,
              keyOrder,
              location,
              locationRadius,
              order,
              pageToken,
              publishedAfter,
              publishedBefore,
              q,
              regionCode,
              relevanceLanguage,
              safeSearch,
              topicId,
              videoCaption,
              videoCategoryId,
              videoDefinition,
              videoDimension,
              videoDuration,
              videoEmbeddable,
              videoLicense,
              videoPaidProductPlacement,
              videoType,
              videoSyndicated,
              year
              ):
    goS = True
    response = {
                'kind': 'youtube#searchListResponse',
                'pageInfo': {'totalResults': 0, 'resultsPerPage': 0},
                'items': []
                } # принудительная выдача для response на случай неуспеха request.execute()
    addItemS = pandas.DataFrame() # принудительная выдача для response на случай неуспеха request.execute()
    goC = True
    while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
        try:
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            request = youtube.search().list(
                                            channelId=channelIdForSearch,
                                            channelType=channelType,
                                            eventType=eventType,
                                            location=location,
                                            locationRadius=locationRadius,
                                            maxResults=50,
                                            order=order,
                                            pageToken=pageToken,
                                            part="snippet",
                                            publishedAfter=publishedAfter,
                                            publishedBefore=publishedBefore,
                                            q=q,
                                            regionCode=regionCode,
                                            relevanceLanguage=relevanceLanguage,
                                            safeSearch=safeSearch,
                                            topicId=topicId,
                                            type=contentType,
                                            videoCaption=videoCaption,
                                            videoCategoryId=videoCategoryId,
                                            videoDefinition=videoDefinition,
                                            videoDimension=videoDimension,
                                            videoDuration=videoDuration,
                                            videoEmbeddable=videoEmbeddable,
                                            videoLicense=videoLicense,
                                            videoPaidProductPlacement=videoPaidProductPlacement,
                                            videoType=videoType,
                                            videoSyndicated=videoSyndicated
                                            )
            response = request.execute()
            addItemS = pandas.json_normalize(response['items'])

            # Для визуализации процесса
            print(
'      Итерация №', iteration, ', number of items', len(response['items']), '' if year == None else f', year {year}', '' if order == None else f', order {order}', '          ', end='\r'
                  )
            iteration += 1
            goC = False

        except:
            print('\nОшибка внутри авторской функции bigSearch') # для отладки
            goC, goS, keyOrder, problemItemId = errorProcessor(
                                                               errorDescription=sys.exc_info(),
                                                               keyOrder=keyOrder,
                                                               sourceId=channelIdForSearch
                                                                )
    return addItemS, goS, iteration, keyOrder, response # от response отказаться нельзя, т.к. в нём много важных ключей, даже если их значения нули

# 1.1 для обработки выдачи метода channels, помогающая работе с ключами
def channelProcessor(API_keyS, channelIdForSearch, coLabFolder, complicatedNamePart, contentType, dfIn, expiriencedMode, fileFormatChoice, goS, keyOrder, momentCurrent, playlistS, q, rootName, slash, snippetContentType, stage, targetCount, year, yearsRange, videoS):
    df = dfIn.copy()
    if len(df) > 0: # если использовался search и успешно, id каналов берутся из него
        channelIdS = df[df['id.kind'] == f'youtube#{snippetContentType}']
        if len(channelIdS) > 0:
            channelIdS =\
                channelIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in channelIdS.columns else channelIdS['id'].to_list()
    else: channelIdS = [channelIdForSearch] # в другом случае id канала подаётся пользователем
    # channelIdS = channelIdS[:5] # для отладки
    # print(channelIdS)

    method = 'channels'
    print(
'В скрипте используются следующие аргументы метода', method, 'API YouTube: part=["snippet", "brandingSettings", "contentDetails", "id", "localizations", "statistics", "status", "topicDetails"], id, maxResults .',
'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/channels')
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
    print('') # для отступа

# ********** Дополнение списка id каналов из df списком id каналов из playlistS
    if (len(playlistS) > 0) | (len(videoS) > 0):
        print(
'''--- Если стоит задача сформировать релевантную запросу базу каналов и хотите пополнить список каналов теми, к которым относятся выгруженные плейлисты и видео,
просто нажмите Enter (это увеличит совокупность выгруженных каналов, но нет гарантии, что если плейлисты и видео релевантны, то и каналы, к которым они относятся, тоже релевантны)
--- Если НЕ хотите пополнить список, нажмите пробел и затем Enter'''
              )
        if len(input()) == 0:
            if len(playlistS) > 0:

                # Список списков, каждый из которых соответствует одному плейлисту
                playlistChannelId_list = playlistS['snippet.videoOwnerChannelId'].str.split(', ').to_list()
    
                playlistChannelIdS = []
                for snippet in playlistChannelId_list:
                    playlistChannelIdS.extend(snippet)
                channelIdS.extend(playlistChannelIdS)
                channelIdS = list(dict.fromkeys(channelIdS))

# ********** Дополнение списка id каналов из df списком id каналов из videoS
            if len(videoS) > 0:
                channelIdS.extend(videoS['snippet.channelId'].to_list())
                channelIdS = list(dict.fromkeys(channelIdS))

    if len(channelIdS) > 0:
        print(f'''Проход по каналам{' порциями по 50 штук' if 90 > 50 else ''} для выгрузки их характеристик (дополнительных к выруженным методом search)''')
        channelS = portionsProcessor(
                                     API_keyS=API_keyS,
                                     channelIdForSearch=channelIdForSearch,
                                     coLabFolder=coLabFolder,
                                     complicatedNamePart=complicatedNamePart,
                                     contentType=contentType, # snippetContentType -- не то же самое, что contentType , т.к. contentType исходно подаётся пользователем
                                     dfFinal=df,
                                     fileFormatChoice=fileFormatChoice,
                                     idS=channelIdS,
                                     keyOrder=keyOrder,
                                     method=method,
                                     momentCurrent=momentCurrent,
                                     q=q,
                                     rootName=rootName,
                                     slash=slash,
                                     stage=stage,
                                     targetCount=targetCount,
                                     year=year,
                                     yearsRange=yearsRange
                                     )
        df2file.df2fileShell(
                             complicatedNamePart=complicatedNamePart,
                             dfIn=channelS,
                             fileFormatChoice=fileFormatChoice,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                             coLabFolder=coLabFolder,
                             currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             )
    return channelS

# 1.2 для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessor(
                  channelIdForSearch,
                  coLabFolder,
                  complicatedNamePart,
                  contentType,
                  fileFormatChoice,
                  dfAdd,
                  dfFinal, # на обработке какой бы ни было выгрузки не возникла бы непреодолима ошибка, сохранить следует выгрузку метода search
                  dfIn,
                  goS, # единственная из функций, принимающая этот аргумент
                  method,
                  q,
                  rootName,
                  slash,
                  stageTarget,
                  targetCount,
                  momentCurrent,
                  year,
                  yearsRange
                  ):
    df = pandas.concat([dfIn, dfAdd])
    columnsForCheck = []
    for column in df.columns: # выдача многих методов содержит столбец id, он оптимален для проверки дублирующихся строк
        if 'id' == column:
            columnsForCheck.append(column)
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующихся строк возможна по столбцам, содержащим в имени id
        for column in df.columns:
            if 'id.' in column:
                columnsForCheck.append(column)
    # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck) # для отладки
    df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True) # при дублировании объектов из itemS из Temporal и от пользователя и новых объектов, оставить новые

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
    if goS == False:
        print(
f'Поскольку исполнение скрипта натолкнулось на непреодолимую ошибку, сохраняю выгруженный контент и текущий этап поиска в директорию "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal"'
              )
        if not os.path.exists(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal'):
                os.makedirs(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal')
                print(f'Директория "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal" создана')
        # else:
            # print(f'Директория "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal" существует') # для отладки

        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}channelIdForSearch.txt', 'w+') # открыть на запись
        file.write(channelIdForSearch if channelIdForSearch != None else '')
        file.close()

        file = open(f'{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal{slash}contentType.txt', 'w+') # открыть на запись
        file.write(contentType if contentType != None else '')
        file.close()

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
'''Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'''
              )
        sys.exit()
    return df

# 1.3 для выгрузки комментариев
def downloadComments(
                     API_keyS,
                     sourceId,
                     keyOrder,
                     method
                     ):
    goS = True
    commentS = pandas.DataFrame()
    pageToken = None
    problemItemId = None
    youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
    while True: # прерывается командой break при отсутствии nextPageToken
        goC = True
        while goC: # цикл позволяет возвращяться со следующим keyOrder к прежнему id при истечении квоты текущего ключа
            try:
                if method == 'comments':
                    response = youtube.comments().list(part='id, snippet', parentId=sourceId, maxResults=100, pageToken=pageToken).execute()

                if method == 'commentThreads':
                    response = youtube.commentThreads().list(maxResults=100, pageToken=pageToken, part='id, replies, snippet', videoId=sourceId).execute()

                goC = False

            except:
                print('\nОшибка внутри авторской функции downloadComments') # для отладки
                goC, goS, keyOrder, problemItemId = errorProcessor(
                                                                    errorDescription=sys.exc_info(),
                                                                    keyOrder=keyOrder,
                                                                    sourceId=sourceId
                                                                    )
        commentS = pandas.concat([commentS, pandas.json_normalize(response['items'])])
        if 'nextPageToken' in response.keys():
            pageToken = response['nextPageToken']
            # print('nextPageToken', pageToken) # для отладки
        else: break

    return commentS, goS, keyOrder, problemItemId

# 1.4 для обработки ошибок
def errorProcessor(errorDescription, keyOrder, sourceId):
    goS = True
    goC = True
    problemItemId = sourceId
    print(errorDescription[1])
    if ('exceeded' in str(errorDescription[1]).lower()) & ('quota' in str(errorDescription[1]).lower()):
        print('!!! Похоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
        # print('  keyOrder ДО смены ключа', keyOrder) # для отладки
        keyOrder += 1 # смена ключа
        # print('  keyOrder ПОСЛЕ смены ключа', keyOrder) # для отладки
    elif ('desabled' in str(errorDescription[1]).lower()) & ('key' in str(errorDescription[1]).lower()):
        print('!!! Похоже, текущий ключ деактивирован владельцем или Гуглом; пробую перейти к следующему ключу')
        # print('  keyOrder ДО смены ключа', keyOrder) # для отладки
        keyOrder += 1 # смена ключа
        # print('  keyOrder ПОСЛЕ смены ключа', keyOrder) # для отладки
    elif 'index out of range' in str(errorDescription[1]):
        print('!!! Похоже, ключи закончились. Что делать? (а) Подождите сутки для восстановления ключей или',
              '(б) подготовьте новый ключ, найдите и удадите файл credentialsYouTube.txt -- и запустите скрипт с начала')
        goS = False # нет смысла продолжать исполнение скрипта
        goC = False # и, следовательно, нет смысла в новых итерациях цикла (вовне этой функции)
    elif ('channel' in str(errorDescription[1]).lower()) | ('comment' in str(errorDescription[1]).lower()) | ('playlist' in str(errorDescription[1]).lower()) | ('video' in str(errorDescription[1]).lower()):
        print('  Проблема может быть связана с ограничением доступа к обрабатываемому объекту, поэтому фиксирую его id:', problemItemId)
        # print('problemItemId:', problemItemId) # для отладки
    elif 'TimeoutError' in str(errorDescription[0]):
        print('!!! Похоже, проблема в слишком высокой частоте запросов к удалённому серверу; засыпаю на 10 миллисекунд')
        time.sleep(10)
    else:
        print('!!! Похоже, проблема не в ограничении доступа к обрабатываемому объекту и не в истечении квоты текущего ключа((')
        goC = False # нет смысла повторного обращения к API ни с этим id, ни пока не ясна суть ошибки
    return goC, goS, keyOrder, problemItemId

# 1.5 для визуализации процесса через итерации
def iterationVisualization(idS, iteration, portion, response):
    if idS != None: iterationUpperBound = int(str(len(idS) / portion).split('.')[0]) + 1 # дробная часть после деления числа idS должна увеличить iterationUpperBound на единицу
    print(
f'''  Порция № {iteration + 1}{f' из {iterationUpperBound}' if idS != None else ''}.{f' Сколько в порции наблюдений? {len(response["items"])}' if portion > 1 else ''}''', end='\r'
          )

# 1.6 для обработки выдачи методов playlists и playlistItems, помогающая работе с ключами
def playListProcessor(API_keyS, channelIdForSearch, coLabFolder, complicatedNamePart, contentType, dfFinal, expiriencedMode, fileFormatChoice, goS, keyOrder, momentCurrent, playlistIdS, q, rootName, slash, snippetContentType, stage, targetCount, year, yearsRange):
    method = 'playlists'
    print('В скрипте используются следующие аргументы метода', method, 'API YouTube: part=["snippet", "contentDetails", "localizations", "status"], id, maxResults .',
          'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
          'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:',
          'https://developers.google.com/youtube/v3/docs/playlists')
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
    print('') # для отступа
    
    if len(playlistIdS) > 0:
        print(f'''Проход по плейлистам{' порциями по 50 штук' if len(playlistIdS) > 50 else ''} для выгрузки их характеристик (дополнительных к выруженным методом search)''')
        playlistS = portionsProcessor(
                                      API_keyS=API_keyS,
                                      channelIdForSearch=channelIdForSearch,
                                      coLabFolder=coLabFolder,
                                      complicatedNamePart=complicatedNamePart,
                                      contentType=contentType,
                                      dfFinal=dfFinal,
                                      fileFormatChoice=fileFormatChoice,
                                      idS=playlistIdS,
                                      keyOrder=keyOrder,
                                      method=method,
                                      momentCurrent=momentCurrent,
                                      q=q,
                                      rootName=rootName,
                                      slash=slash,
                                      stage=stage,
                                      targetCount=targetCount,
                                      year=year,
                                      yearsRange=yearsRange
                                      )

        method = 'playlistItems'
        print('В скрипте используются следующие аргументы метода', method, 'API YouTube: part=["snippet"], playlistId, maxResults .',
              'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
              'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:',
              'https://developers.google.com/youtube/v3/docs/playlistitems')
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа
    
        iteration = 0 # номер итерации применения текущего метода
        playlistVideoChannelS = pandas.DataFrame() # хотя датафреймы и глобальны как переменные, пусть и тут инициализируется
        portion = 50
        print('Проход по плейлистам для выгрузки id видео, составляющих плейлисты, и каналов, к которым они принадлежат')
        for playlistId in playlistIdS:
            pageToken = None
            while True:
                goC = True
                while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
                    try:
                        youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
                        response = youtube.playlistItems().list(part='snippet', maxResults=50, pageToken=pageToken, playlistId=playlistId).execute()
                        addPlaylistVideoChannelS = pandas.json_normalize(response['items'])
                        playlistVideoChannelS = dfsProcessor(
                                                             channelIdForSearch=channelIdForSearch,
                                                             coLabFolder=coLabFolder,
                                                             complicatedNamePart=complicatedNamePart,
                                                             contentType=contentType,
                                                             fileFormatChoice=fileFormatChoice,
                                                             dfAdd=addPlaylistVideoChannelS,
                                                             dfFinal=dfFinal,
                                                             dfIn=playlistVideoChannelS,
                                                             goS=goS,
                                                             method=method,
                                                             q=q,
                                                             rootName=rootName,
                                                             slash=slash,
                                                             stageTarget=stage,
                                                             targetCount=targetCount,
                                                             momentCurrent=momentCurrent,
                                                             year=year,
                                                             yearsRange=yearsRange
                                                             )
                        goC = False # если try успешно исполнился, то цикл прекращается
                    except: goC, goS, keyOrder, problemItemId = errorProcessor(
                                                                               errorDescription=sys.exc_info(),
                                                                               keyOrder=keyOrder,
                                                                               sourceId=None
                                                                               )
                iterationVisualization(idS=None, iteration=iteration, portion=portion, response=response) # для визуализации процесса через итерации
                iteration += 1
                if 'nextPageToken' in response.keys(): pageToken = response['nextPageToken']
                else: break
        print('                                                                                                              ') # затираю последнюю визуализацию

        # Перечислить сначала id всех составляющих каждый плейлист видео через запятую и записать в ячейку,
            # затем id всех канадов, к которым относятся составляющие каждый плейлист видео, через запятую и записать в ячейку
        # display('playlistVideoChannelS', playlistVideoChannelS) # для отладки
        for playlistId in playlistIdS:
            for column in ['snippet.resourceId.videoId', 'snippet.videoOwnerChannelId']:
                playlistVideoChannelS_snippet = playlistVideoChannelS[playlistVideoChannelS[column].notna()]
                playlistS.loc[playlistS[playlistS['id'] == playlistId].index[0], column] =\
                    ', '.join(playlistVideoChannelS_snippet[playlistVideoChannelS_snippet['snippet.playlistId'] == playlistId][column].to_list())
        # display(playlistS)
        df2file.df2fileShell(
                             complicatedNamePart=complicatedNamePart,
                             dfIn=playlistVideoChannelS,
                             fileFormatChoice=fileFormatChoice,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                             coLabFolder=coLabFolder,
                             currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                 )
    return playlistS, playlistVideoChannelS

# 1.7 для порционной выгрузки, когда метод предполагает подачу ему id порциями
def portionsProcessor(API_keyS, channelIdForSearch, coLabFolder, complicatedNamePart, contentType, dfFinal, fileFormatChoice, idS, keyOrder, method, momentCurrent, q, rootName, slash, stage, targetCount, year, yearsRange):
    # print('method', method) # для отладки
    bound = 0
    chplviS = pandas.DataFrame()
    goS = True
    iteration = 0 # номер итерации применения текущего метода
    portion = 50
    while bound < len(idS):
    # while bound < 100: # для отладки
        goC = True
        while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
            try:
                youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
                if method == 'channels':
                    response = youtube.channels().list(
                                                       part='snippet, brandingSettings, contentDetails, id, localizations, statistics, status, topicDetails'
                                                       , id=idS[bound:bound + portion]
                                                       , maxResults=50
                                                       ).execute()
                if method == 'playlists':
                    response = youtube.playlists().list(
                                                        part='snippet, contentDetails, localizations, status'
                                                        , id=idS[bound:bound + portion]
                                                       , maxResults=50
                                                        ).execute()
                if method == 'videos':
                    response = youtube.videos().list(
                                                     part='snippet, contentDetails, localizations, statistics, status, topicDetails'
                                                     , id=idS[bound:bound + portion]
                                                     , maxResults=50
                                                     ).execute()
                addChplviS = pandas.json_normalize(response['items'])
                chplviS = dfsProcessor(
                                        channelIdForSearch=channelIdForSearch,
                                        coLabFolder=coLabFolder,
                                        complicatedNamePart=complicatedNamePart,
                                        contentType=contentType,
                                        fileFormatChoice=fileFormatChoice,
                                        dfAdd=addChplviS,
                                        dfFinal=dfFinal, # itemS подаются как значение аргумента оборачивающей функции
                                        dfIn=chplviS,
                                        goS=goS,
                                        method=method,
                                        momentCurrent=momentCurrent,
                                        q=q,
                                        rootName=rootName,
                                        slash=slash,
                                        stageTarget=stage,
                                        targetCount=targetCount,
                                        year=year,
                                        yearsRange=yearsRange
                                        )
                goC = False # если try успешно исполнился, то цикл прекращается
            except:
                print('\nОшибка внутри авторской функции portionsProcessor') # для отладки
                goC, goS, keyOrder, problemItemId = errorProcessor(
                                                                    errorDescription=sys.exc_info(),
                                                                    keyOrder=keyOrder,
                                                                    sourceId=None
                                                                    )
        # print('len(idS):', len(idS)) # для отладки
        iterationVisualization(idS, iteration, portion, response) # для визуализации процесса через итерации
        iteration += 1
        bound += portion
        # display('chplviS:', chplviS) # для отладки
    print('                                                                                                              ') # затираю последнюю визуализацию
    return chplviS

# 1.6 чтобы избавиться от префиксов в названиях столбцов датафрейма с комментариями
def prefixDropper(df):
        dfNewColumnS = []
        for column in df.columns:
            if 'snippet.topLevelComment.' in column:
                column = column.replace('snippet.topLevelComment.', '')
            dfNewColumnS.append(column)
        df.columns = dfNewColumnS # перезаписать названия столбцов
        return df


# In[ ]:


# # 2 Авторская функция исполнения скрипта


# In[ ]:


def searchByText(
                 access_token=None,
                 channelIdForSearch=None,
                 contentType=None,
                 publishedAfter=None,
                 publishedBefore=None,
                 q=None,
                 channelType=None,
                 eventType=None,
                 location=None,
                 locationRadius=None,
                 regionCode=None,
                 relevanceLanguage=None,
                 safeSearch=None,
                 topicId=None,
                 videoCaption=None,
                 videoCategoryId=None,
                 videoDefinition=None,
                 videoDimension=None,
                 videoDuration=None,
                 videoEmbeddable=None,
                 videoLicense=None,
                 videoPaidProductPlacement=None,
                 videoSyndicated=None,
                 videoType=None,
                 returnDfs = False
                 ):
    """
    Функция для выгрузки характеристик контента YouTube методами его API: search, playlists & playlistItems, videos, commentThreads & comments, channels -- ключевым из которых выступает search
    Причём количество объектов выгрузки максимизируется путём её пересортировки аргументом order и сегментирования по годам

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://developers.google.com/youtube/v3/docs/search/list
             access_token : str
       channelIdForSearch : str -- это аналог channelId
              contentType : str -- это аналог type
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
                returnDfs : bool -- в случае True функция возвращает пять итоговые датафреймы с выдачей методов (1) search, (2) playlists и playlistItems, (3) videos, (4) commentThreads и comments, (5) channels
    """
    if (access_token == None) & (channelIdForSearch == None) & (contentType == None) & (publishedAfter == None) & (publishedBefore == None) & (q == None)\
        & (channelType == None) & (eventType == None) & (location == None) & (locationRadius == None) & (regionCode == None) & (relevanceLanguage == None) & (safeSearch == None) & (topicId == None)\
        & (videoCaption == None) & (videoCategoryId == None) & (videoDefinition == None) & (videoDimension == None) & (videoDuration == None) & (videoEmbeddable == None) & (videoLicense == None)\
        & (videoPaidProductPlacement == None) & (videoSyndicated == None) & (videoType == None) & (returnDfs == False):
        # print('Пользователь не подал аргументы')
        expiriencedMode = False
    else: expiriencedMode = True

    if expiriencedMode == False:
        print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API YouTube по ключу (см. примерную видео-инструкцию: https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ). Для получения API key следует создать проект, авторизовать его, подключить к нему API нужного сервиса Google. Проект -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому сервисов Google и применения на этой основе API разных сервисов Google в рамках установленных Гуглом ограничений (the units of quota). Разные уровни авторизации -- это авторизация ключом (представляющим собой код) и полная авторизация (ключ + протокол Google OAuth 2.0, реализующийся в формате файла JSON). Авторизация ключом нужна, чтобы использовать любой метод любого API. Её достаточно, если выполнять действия, которые были бы доступны Вам как пользователю сервисов Google без Вашего входа в аккаунт: посмотреть видео, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления видео, то Вам придётся пройти полную авторизацию. Далее разные API как бы подключаются к проектам (кнопка Enable APIs and servises), используются, затем отключаются (кнопка Disable APIs).
    Квоты одного ключа может не хватить (quota is exceeded) для выгрузки всего предоставляемого ЮТьюбом по запросу пользователя контента. К счастью, использованный ключ ежесуточно восстанавливается ЮТьюбом. скрипт позволяет сохранить промежуточную выгрузку и после восстановления ключа автоматически продолжит её дополнять с момента остановки. В момент остановки появится надпись: "Поскольку ключи закончились, исполнение скрипта завершаю. Подождите сутки для восстановления ключей или подготовьте новый ключ -- и запустите скрипт с начала", а исполнение скрипта прервётся. Не пугайтесь, нажмите OK и следуйте этой инструкции.'''
              )
    print(
'''    Скрипт нацелен на выгрузку характеристик контента YouTube семью методами его API: search, videos, commentThreads и comments, channels, playlists и playlistItems. Причём количество объектов выгрузки максимизируется путём её пересортировки и сегментирования по годам.
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из YouTube вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API YouTube: гораздо быстрее, гораздо большее количество контента с одним и тем же ключом, не требуется тщательно изучать обширную документацию семи методов API YouTube (search, videos, commentThreads и comments, channels, playlists и playlistItems), выстроена логика обрашения к этим методам'''
          )
    if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API YouTube
    channelS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    coLabFolder = coLabAdaptor.coLabAdaptor()
    commentReplieS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    fileFormatChoice = '.xlsx' # базовый формат сохраняемых файлов; формат .json добавляется опционально через наличие columnsToJSON
    folder = None
    folderFile = None
    goS = True
    itemS = pandas.DataFrame(columns=['id.kind']) # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    keyOrder = 0
    playlistVideoChannelS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    playlistS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    stageTarget = 0 # stageTarget принимает значения [0; 3] и относится к стадиям скрипта
    targetCount = 0
    temporalName = None
    videoS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    yearsRange = None

    momentCurrent = datetime.now() # запрос текущего момента
    print('\nТекущий момент:', momentCurrent.strftime("%Y%m%d_%H%M"), '-- он будет использована для формирования имён создаваемых директорий и файлов')
    year = int(momentCurrent.strftime("%Y")) # в случае отсутствия пользовательского временнОго диапазона
        # с этого года возможно сегментирование по годам вглубь веков (пока выдача не пустая)
    yearMinByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearMaxByUser = None # в случае отсутствия пользовательского временнОго диапазона

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    rootNameS = os.listdir() if coLabFolder == None else os.listdir(coLabFolder)
    # Поиск ключей
    if access_token == None:
        print('Проверяю наличие файла credentialsYouTube.txt с ключ[ом ами], гипотетически сохранённым[и] при первом запуске скрипта')
        if 'credentialsYouTube.txt' in rootNameS:
            file = open('credentialsYouTube.txt' if coLabFolder == None else coLabFolder + slash + "credentialsYouTube.txt")
            API_keyS = file.read()
            print('Нашёл файл credentialsYouTube.txt; далее буду использовать ключ[и] из него:', API_keyS)
        else:
            print(
'''--- НЕ нашёл файл credentialsYouTube.txt . Введите в окно Ваш API key для авторизации в API YouTube по ключу (примерная видео-инструкция, как создать API key, доступна по ссылке https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ). Для увеличения размера выгрузки желательно создать несколько ключей (пять -- отлично) и ввести их без кавычек через запятую с пробелом
--- После ввода нажмите Enter'''
                  )
            while True:
                API_keyS = input()
                if len(API_keyS) != 0:
                    print('-- далее буд[е у]т использован[ы] эт[от и] ключ[и]')

                    from randan.tools.textPreprocessor import multispaceCleaner # авторский модуль для предобработки нестандартизированного текста
                    API_keyS = multispaceCleaner(API_keyS)
                    while API_keyS[-1] == ',': API_keyS = API_keyS[:-1] # избавиться от запятых в конце текста

                    file = open("credentialsYouTube.txt" if coLabFolder == None else coLabFolder + slash + "credentialsYouTube.txt", "w+") # открыть на запись
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
    print(
'Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку'
          )
    for rootName in rootNameS:
        if 'Temporal' in rootName:
            if len(os.listdir(rootName)) == 9:
                file = open(f'{rootName}{slash}targetCount.txt')
                targetCountTemporal = file.read()
                file.close()
                targetCountTemporal = int(targetCountTemporal)
    
                file = open(f'{rootName}{slash}method.txt')
                methodTemporal = file.read()
                file.close()
    
                file = open(f'{rootName}{slash}year.txt')
                yearTemporal = file.read()
                file.close()
                yearTemporal = int(yearTemporal)
    
                file = open(f'{rootName}{slash}contentType.txt')
                contentTypeTemporal = file.read()
                file.close()
                if contentTypeTemporal == '': contentTypeTemporal = None # для единообразия
    
                file = open(f'{rootName}{slash}channelIdForSearch.txt')
                channelIdForSearchTemporal = file.read()
                file.close()
                if channelIdForSearchTemporal == '': channelIdForSearchTemporal = None # для единообразия
    
                file = open(f'{rootName}{slash}q.txt', encoding='utf-8')
                qTemporal = file.read()
                file.close()
                if qTemporal == '': qTemporal = None # для единообразия
    
                file = open(f'{rootName}{slash}yearsRange.txt')
                yearsRangeTemporal = file.read()
                file.close()
                if yearsRangeTemporal == '': yearsRangeTemporal = None # для единообразия
    
                file = open(f'{rootName}{slash}stageTarget.txt')
                stageTargetTemporal = file.read()
                file.close()
                stageTargetTemporal = int(stageTargetTemporal)
    
                print(f'Нашёл директорию "{rootName}". В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:',
                      '\n- было выявлено целевое число записей (totalResults)', targetCountTemporal,
                      '\n- скрипт остановился на методе', methodTemporal)
                if yearTemporal != None: print('- и на годе (при сегментировани по годам)', yearTemporal)
                print('- пользователь НЕ определил тип контента' if contentTypeTemporal == None else  f'- пользователь определил тип контента как "{contentTypeTemporal}"')
                if contentTypeTemporal == 'video':
                    print('- пользователь НЕ выбрал конкретный канал для выгрузки видео' if channelIdForSearchTemporal == None else  f'- пользователь выбрал канал с id "{channelIdForSearchTemporal}" для выгрузки видео')
                print('- пользователь НЕ сформулировал запрос-фильтр' if qTemporal == None else  f'- пользователь сформулировал запрос-фильтр как "{qTemporal}"')
                print('- пользователь НЕ ограничил временнОй диапазон' if yearsRangeTemporal == '' else  f'- пользователь ограничил временнОй диапазон границами {yearsRangeTemporal}')
                print(
'''--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter
--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите "R" и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter'''
                  )
                decision = input()
                if len(decision) == 0: # Temporal-значения обретают статус постоянных
                    targetCount = targetCountTemporal
                    method = methodTemporal
                    year = yearTemporal
                    contentType = contentTypeTemporal
                    channelIdForSearch = channelIdForSearchTemporal
                    q = qTemporal
                    yearsRange = yearsRangeTemporal
                    stageTarget = stageTargetTemporal
    
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
# Данные, сохранённые при прошлом запуске скрипта, загружены; их метаданные (q, contentType, yearsRange, stageTarget) будут использоваться при исполнении скрипта
                    break
                elif decision.upper() == 'R':
                    shutil.rmtree(rootName, ignore_errors=True)
                    print('')

# 2.0.3 Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if temporalName == None: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        rootName = 'No folder'
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта')
        print(
'''
Возможно, Вы располагаете файлом, в котором есть, как минимум, столбец id, и который хотели бы дополнить? Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом, укажите полный путь, включая название файла, и нажмите Enter'''
              )

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
                print('--- Если НЕ требуется определить тип искомого контента, нажмите Enter'
                      , ' \n--- Если требуется определить, введите символ: c -- channel, p -- playlist, v -- video -- и нажмите Enter')
                contentType = input()
                if contentType.lower() == '':
                    contentType = ''
                    break
                elif contentType.lower() == 'c':
                    contentType = 'channel'
                    print('')
                    break
                elif contentType.lower() == 'p':
                    contentType = 'playlist'
                    print('')
                    break
                elif contentType.lower() == 'v':
                    contentType = 'video'
                    print('')
                    break
                else:
                    print('--- Вы ввели что-то не то; попробуйте, пожалуйста, ещё раз..')

        if (channelIdForSearch == None) & (contentType == 'video'): # если пользователь не подал аргумент channelIdForSearch в рамках experiencedMode
            print(
'''--- Вы выбрали тип контента video
--- Если НЕ предполагается поиск видео в пределах конкретного канала, нажмите Enter
--- Если предполагается такой поиск, введите id канала, после чего нажмите Enter. Этот id можете найти либо в URL-адресе интересующего канала, либо -- если прежде выгружали контент из YouTube -- в столбце "snippet.channelId" выдачи методов search, playlistItems, videos или в столбце "id" метода cannels'''
                  )
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
                    from randan.tools.textPreprocessor import multispaceCleaner
                    channelIdForSearch = multispaceCleaner(channelIdForSearch)
                    while channelIdForSearch[-1] == ',': channelIdForSearch = channelIdForSearch[:-1] # избавиться от запятых в конце текста
                    # channelIdForSearch = channelIdForSearch.split(', ')
                    # print('Количество id каналов:', len(channelIdForSearch), '\n')
                    # if len(channelIdForSearch) > 1:
                        # channelIdForSearch = channelIdForSearch[0]
                        # print('В качестве аргумента метода search будет использован ТОЛЬКО первый из id\n')
                    # print('channelIdForSearch', channelIdForSearch) # для отладки
                    print('')
                    break
        if q == None: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать контент по текстовому запросу-фильтру. При этом если требуется контент конкретного канала, то лучше использовать запросу-фильтр НЕ на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными
--- Если НЕ требуется поиск контента по текстовому запросу-фильтру, нажмите Enter
--- Если требуется такой поиск, введите текст запроса-фильтра, который ожидаете в атрибутах и характеристиках
(описание, название, теги, категории и т.п.) релевантного YouTube-контента, после чего нажмите Enter'''
            )
            if folderFile != None: print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folderFile,
'будут дополнены актуальными данными из выдачи скрипта (возможно появление новых записей и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла'
                                         )
            q = input()
            if q == '': q = None # для единообразия
            else: print('')

    # Ограничения временнОго диапазона
        if (publishedAfter == None) & (publishedBefore == None) & (yearsRange == None): # если пользователь не подал эти аргументы в рамках experiencedMode
            print(
'''Алгоритм API Youtube для ограничения временнОго диапазона выдаваемого контента работает со странностями. Поэтому если требуется конкретный временнОй диапазон, то лучше использовать его НЕ на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными
--- Если НЕ требуется задать временнОй диапазон на этапе выгрузки данных, нажмите Enter
--- Если всё же требуется задать временнОй диапазон, настоятельная рекомендация задать его годами, а не более мелкими единицами времени. Для задания диапазона введите без кавычек минимальный год диапазона, тире, максимальный год диапазона (минимум и максимум могут совпадать в такой записи: "год-тот же год") и нажмите Enter'''
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
        if publishedAfter != None:
            yearMinByUser = int(datetime.strptime(publishedAfter,"%Y-%m-%dT%H:%M:%SZ").strftime('%Y')) # из experiencedMode
            # print('elif start_time != None:', yearMinByUser) # для отладки

        if publishedBefore != None:
            yearMaxByUser = int(datetime.strptime(publishedBefore,"%Y-%m-%dT%H:%M:%SZ").strftime('%Y')) # из experiencedMode
            # print('elif end_time != None:', yearMaxByUser) # для отладки
            year = yearMaxByUser

        if (yearMinByUser != None) & (yearMaxByUser == None): yearMaxByUser = int(momentCurrent.strftime("%Y")) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
        elif (yearMinByUser == None) & (yearMaxByUser != None): yearMaxByUser = 1970 # в случае отсутствия пользовательской нижней временнОй границы при наличии верхней

        # print('yearMinByUser', yearMinByUser) # для отладки
        # print('yearMaxByUser', yearMaxByUser) # для отладки

        if (publishedAfter == None) & (yearMinByUser != None): publishedAfter = datetime(yearMinByUser, 1, 1).isoformat() + 'Z'
        if (publishedBefore == None) & (yearMaxByUser != None): publishedBefore = datetime(yearMaxByUser, 12, 31).isoformat() + 'Z'

        if yearsRange != None: print('') # чтобы был отступ, если пользователь подал этот аргумент

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_YT'
    complicatedNamePart += "" if contentType == None else "_" + contentType
    complicatedNamePart += "" if channelIdForSearch == None else "_channelId_" + channelIdForSearch
    if q != None: complicatedNamePart += "_" + q if len(q) < 50 else "_" + q[:50]
    complicatedNamePart += "" if ((yearMinByUser == None) & (yearMaxByUser == None)) else "_" + str(yearMinByUser) + '-' + str(yearMaxByUser)
# print('complicatedNamePart', complicatedNamePart)

# 2.1 Первичный сбор контента методом search
# 2.1.0 Первый заход БЕЗ аргумента order (этап stage = 0)
    if (channelIdForSearch == None) | (q != None) | (yearsRange != None): # в противном случае search следует заменить на cannels + playlistItems
        stage = 0
        iteration = 0 # номер итерации применения текущего метода
        method = 'search'
        print(
f'В скрипте используются следующие аргументы метода {method} API YouTube: channelId, maxResults, order, pageToken, part, publishedAfter, publishedBefore, q, type.',
'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
f'Если хотите добавить другие аргументы метода {method} API YouTube, доступные по ссылке https://developers.google.com/youtube/v3/docs/search , -- можете сделать это внутри метода {method} в разделе 2 исполняемого сейчас скрипта'
              )
        if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа

        if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
            print('Заход на первую страницу выдачи')
            # print('publishedAfter', publishedAfter) # для отладки
            addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                     API_keyS=API_keyS,
                                                                     channelIdForSearch=channelIdForSearch,
                                                                     channelType=channelType,
                                                                     contentType=contentType,
                                                                     iteration=iteration,
                                                                     keyOrder=keyOrder,
                                                                     order=None,
                                                                     publishedAfter=publishedAfter,
                                                                     publishedBefore=publishedBefore,
                                                                     pageToken=None,
                                                                     q=q,
                                                                     eventType=eventType,
                                                                     location=location,
                                                                     locationRadius=locationRadius,
                                                                     regionCode=regionCode,
                                                                     relevanceLanguage=relevanceLanguage,
                                                                     safeSearch=safeSearch,
                                                                     topicId=topicId,
                                                                     videoCaption=videoCaption,
                                                                     videoCategoryId=videoCategoryId,
                                                                     videoDefinition=videoDefinition,
                                                                     videoDimension=videoDimension,
                                                                     videoDuration=videoDuration,
                                                                     videoEmbeddable=videoEmbeddable,
                                                                     videoLicense=videoLicense,
                                                                     videoPaidProductPlacement=videoPaidProductPlacement,
                                                                     videoType=videoType,
                                                                     videoSyndicated=videoSyndicated,
                                                                     year=None
                                                                     )
            targetCount = response['pageInfo']['totalResults']
            if targetCount == 0:
                print(
'''Искомых объектов на серверах YouTube по Вашему запросу, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта. Что делать? Поменяйте настройки запроса и запустите скрипт с начала'''
                      )
                warnings.filterwarnings("ignore")
                print(
'Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть',
'Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'
                      )
                sys.exit()
    
            itemS = dfsProcessor(
                                  channelIdForSearch=channelIdForSearch,
                                  coLabFolder=coLabFolder,
                                  complicatedNamePart=complicatedNamePart,
                                  contentType=contentType,
                                  fileFormatChoice=fileFormatChoice,
                                  dfAdd=addItemS,
                                  dfFinal=itemS,
                                  dfIn=itemS,
                                  goS=goS,
                                  method=method,
                                  q=q,
                                  rootName=rootName,
                                  slash=slash,
                                  stageTarget=stage,
                                  targetCount=targetCount,
                                  momentCurrent=momentCurrent,
                                  year=year,
                                  yearsRange=yearsRange
                                  )
            # display('itemS', itemS) # для отладки
    
            print('  Проход по всем следующим страницам с выдачей          ')
            while 'nextPageToken' in response.keys():
                pageToken = response['nextPageToken']
                addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                         API_keyS=API_keyS,
                                                                         channelIdForSearch=channelIdForSearch,
                                                                         channelType=channelType,
                                                                         contentType=contentType,
                                                                         eventType=eventType,
                                                                         iteration=iteration,
                                                                         keyOrder=keyOrder,
                                                                         order=None,
                                                                         location=location,
                                                                         locationRadius=locationRadius,
                                                                         publishedAfter=publishedAfter,
                                                                         publishedBefore=publishedBefore,
                                                                         pageToken=pageToken,
                                                                         q=q,
                                                                         regionCode=regionCode,
                                                                         relevanceLanguage=relevanceLanguage,
                                                                         safeSearch=safeSearch,
                                                                         topicId=topicId,
                                                                         videoCaption=videoCaption,
                                                                         videoCategoryId=videoCategoryId,
                                                                         videoDefinition=videoDefinition,
                                                                         videoDimension=videoDimension,
                                                                         videoDuration=videoDuration,
                                                                         videoEmbeddable=videoEmbeddable,
                                                                         videoLicense=videoLicense,
                                                                         videoPaidProductPlacement=videoPaidProductPlacement,
                                                                         videoType=videoType,
                                                                         videoSyndicated=videoSyndicated,
                                                                         year=None
                                                                         )
                itemS = dfsProcessor(
                                      channelIdForSearch=channelIdForSearch,
                                      coLabFolder=coLabFolder,
                                      complicatedNamePart=complicatedNamePart,
                                      contentType=contentType,
                                      fileFormatChoice=fileFormatChoice,
                                      dfAdd=addItemS,
                                      dfFinal=itemS,
                                      dfIn=itemS,
                                      goS=goS,
                                      method=method,
                                      q=q,
                                      rootName=rootName,
                                      slash=slash,
                                      stageTarget=stage,
                                      targetCount=targetCount,
                                      momentCurrent=momentCurrent,
                                      year=year,
                                      yearsRange=yearsRange
                                      )
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
                                                                             API_keyS=API_keyS,
                                                                             channelIdForSearch=channelIdForSearch,
                                                                             channelType=channelType,
                                                                             contentType=contentType,
                                                                             eventType=eventType,
                                                                             iteration=iteration,
                                                                             keyOrder=keyOrder,
                                                                             location=location,
                                                                             locationRadius=locationRadius,
                                                                             order=order,
                                                                             publishedAfter=publishedAfter,
                                                                             publishedBefore=publishedBefore,
                                                                             pageToken=None,
                                                                             q=q,
                                                                             regionCode=regionCode,
                                                                             relevanceLanguage=relevanceLanguage,
                                                                             safeSearch=safeSearch,
                                                                             topicId=topicId,
                                                                             videoCaption=videoCaption,
                                                                             videoCategoryId=videoCategoryId,
                                                                             videoDefinition=videoDefinition,
                                                                             videoDimension=videoDimension,
                                                                             videoDuration=videoDuration,
                                                                             videoEmbeddable=videoEmbeddable,
                                                                             videoLicense=videoLicense,
                                                                             videoPaidProductPlacement=videoPaidProductPlacement,
                                                                             videoType=videoType,
                                                                             videoSyndicated=videoSyndicated,
                                                                             year=None
                                                                             )
                    itemS = dfsProcessor(
                                          channelIdForSearch=channelIdForSearch,
                                          coLabFolder=coLabFolder,
                                          complicatedNamePart=complicatedNamePart,
                                          contentType=contentType,
                                          fileFormatChoice=fileFormatChoice,
                                          dfAdd=addItemS,
                                          dfFinal=itemS,
                                          dfIn=itemS,
                                          goS=goS,
                                          method=method,
                                          q=q,
                                          rootName=rootName,
                                          slash=slash,
                                          stageTarget=stage,
                                          targetCount=targetCount,
                                          momentCurrent=momentCurrent,
                                          year=year,
                                          yearsRange=yearsRange
                                          )
    
                    print('  Проход по всем следующим страницам с выдачей с тем же значением аргумента order:', order, '          ')
                    while ('nextPageToken' in response.keys()) & (len(itemS) < targetCount) & (len(response["items"]) > 0):
                    # -- второе условие -- для остановки алгоритма, если все искомые объекты найдены
                        # БЕЗ какой-то из следующих страниц (в т.ч. вообще БЕЗ них)
                        # третье условие -- для остановки алгоритма, если предыдущая страница выдачи содержит 0 объектов
    
                        pageToken = response['nextPageToken']
                        # print('pageToken', pageToken)
                        addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                 API_keyS=API_keyS,
                                                                                 channelIdForSearch=channelIdForSearch,
                                                                                 channelType=channelType,
                                                                                 contentType=contentType,
                                                                                 eventType=eventType,
                                                                                 iteration=iteration,
                                                                                 keyOrder=keyOrder,
                                                                                 location=location,
                                                                                 locationRadius=locationRadius,
                                                                                 order=order,
                                                                                 publishedAfter=publishedAfter,
                                                                                 publishedBefore=publishedBefore,
                                                                                 pageToken=pageToken,
                                                                                 q=q,
                                                                                 regionCode=regionCode,
                                                                                 relevanceLanguage=relevanceLanguage,
                                                                                 safeSearch=safeSearch,
                                                                                 topicId=topicId,
                                                                                 videoCaption=videoCaption,
                                                                                 videoCategoryId=videoCategoryId,
                                                                                 videoDefinition=videoDefinition,
                                                                                 videoDimension=videoDimension,
                                                                                 videoDuration=videoDuration,
                                                                                 videoEmbeddable=videoEmbeddable,
                                                                                 videoLicense=videoLicense,
                                                                                 videoPaidProductPlacement=videoPaidProductPlacement,
                                                                                 videoType=videoType,
                                                                                 videoSyndicated=videoSyndicated,
                                                                                 year=None
                                                                                 )
                        itemS = dfsProcessor(
                                              channelIdForSearch=channelIdForSearch,
                                              coLabFolder=coLabFolder,
                                              complicatedNamePart=complicatedNamePart,
                                              contentType=contentType,
                                              fileFormatChoice=fileFormatChoice,
                                              dfAdd=addItemS,
                                              dfFinal=itemS,
                                              dfIn=itemS,
                                              goS=goS,
                                              method=method,
                                              q=q,
                                              rootName=rootName,
                                              slash=slash,
                                              stageTarget=stage,
                                              targetCount=targetCount,
                                              momentCurrent=momentCurrent,
                                              year=year,
                                              yearsRange=yearsRange
                                              )
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
                print('Увы', f'\nЧисло найденных объектов: {len(itemS)} -- менее числа искомых: {targetCount}',
                      '\n--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам, просто нажмите Enter, но учтите, что поиск может занять минуты и даже часы',
                      '\n--- Если НЕ хотите, нажмите пробел и затем Enter')
                if len(input()) == 0:
                    print('Внутри каждого года прохожу по значениям аргумента order, внутри которых прохожу по всем страницам выдачи')
                    goC = True
# ********** из фрагмента 2.1.0 + условие для goC
                    while (len(itemS) < targetCount) & (goC):
                        print(f'  Для года {year} заход на первую страницу выдачи БЕЗ аргумента order')
                        addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                 API_keyS=API_keyS,
                                                                                 channelIdForSearch=channelIdForSearch,
                                                                                 channelType=channelType,
                                                                                 contentType=contentType,
                                                                                 eventType=eventType,
                                                                                 iteration=iteration,
                                                                                 keyOrder=keyOrder,
                                                                                 location=location,
                                                                                 locationRadius=locationRadius,
                                                                                 order=None,
                                                                                 publishedAfter = f'{year}-01-01T00:00:00Z',
                                                                                 publishedBefore = f'{year + 1}-01-01T00:00:00Z',
                                                                                 pageToken=None,
                                                                                 q=q,
                                                                                 regionCode=regionCode,
                                                                                 relevanceLanguage=relevanceLanguage,
                                                                                 safeSearch=safeSearch,
                                                                                 topicId=topicId,
                                                                                 videoCaption=videoCaption,
                                                                                 videoCategoryId=videoCategoryId,
                                                                                 videoDefinition=videoDefinition,
                                                                                 videoDimension=videoDimension,
                                                                                 videoDuration=videoDuration,
                                                                                 videoEmbeddable=videoEmbeddable,
                                                                                 videoLicense=videoLicense,
                                                                                 videoPaidProductPlacement=videoPaidProductPlacement,
                                                                                 videoType=videoType,
                                                                                 videoSyndicated=videoSyndicated,
                                                                                 year=year
                                                                                 )
                        if len(addItemS) == 0:
                            print(f'\n--- Первая страница выдачи БЕЗ аргумента order для года {year} -- пуста',
                                  '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать предыдущий год, просто нажмите Enter',
                                  '\n--- Если хотите, нажмите пробел и затем Enter')
                            if len(input()) == 0:
                                goC = False
                                break
                        itemS = dfsProcessor(
                                              channelIdForSearch=channelIdForSearch,
                                              coLabFolder=coLabFolder,
                                              complicatedNamePart=complicatedNamePart,
                                              contentType=contentType,
                                              fileFormatChoice=fileFormatChoice,
                                              dfAdd=addItemS,
                                              dfFinal=itemS,
                                              dfIn=itemS,
                                              goS=goS,
                                              method=method,
                                              q=q,
                                              rootName=rootName,
                                              slash=slash,
                                              stageTarget=stage,
                                              targetCount=targetCount,
                                              momentCurrent=momentCurrent,
                                              year=year,
                                              yearsRange=yearsRange
                                              )
    
                        print(f'    Проход по всем следующим страницам с выдачей для года {year} БЕЗ аргумента order')
                        while 'nextPageToken' in response.keys():
                            pageToken = response['nextPageToken']
                            addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                     API_keyS=API_keyS,
                                                                                     channelIdForSearch=channelIdForSearch,
                                                                                     channelType=channelType,
                                                                                     contentType=contentType,
                                                                                     eventType=eventType,
                                                                                     iteration=iteration,
                                                                                     keyOrder=keyOrder,
                                                                                     location=location,
                                                                                     locationRadius=locationRadius,
                                                                                     order=None,
                                                                                     publishedAfter = f'{year}-01-01T00:00:00Z',
                                                                                     publishedBefore = f'{year + 1}-01-01T00:00:00Z',
                                                                                     pageToken=pageToken,
                                                                                     q=q,
                                                                                     regionCode=regionCode,
                                                                                     relevanceLanguage=relevanceLanguage,
                                                                                     safeSearch=safeSearch,
                                                                                     topicId=topicId,
                                                                                     videoCaption=videoCaption,
                                                                                     videoCategoryId=videoCategoryId,
                                                                                     videoDefinition=videoDefinition,
                                                                                     videoDimension=videoDimension,
                                                                                     videoDuration=videoDuration,
                                                                                     videoEmbeddable=videoEmbeddable,
                                                                                     videoLicense=videoLicense,
                                                                                     videoPaidProductPlacement=videoPaidProductPlacement,
                                                                                     videoType=videoType,
                                                                                     videoSyndicated=videoSyndicated,
                                                                                     year=year
                                                                                     )
                        if len(addItemS) == 0:
                            itemS = dfsProcessor(
                                                  channelIdForSearch=channelIdForSearch,
                                                  coLabFolder=coLabFolder,
                                                  complicatedNamePart=complicatedNamePart,
                                                  contentType=contentType,
                                                  fileFormatChoice=fileFormatChoice,
                                                  dfAdd=addItemS,
                                                  dfFinal=itemS,
                                                  dfIn=itemS,
                                                  goS=goS,
                                                  method=method,
                                                  q=q,
                                                  rootName=rootName,
                                                  slash=slash,
                                                  stageTarget=stage,
                                                  targetCount=targetCount,
                                                  momentCurrent=momentCurrent,
                                                  year=year,
                                                  yearsRange=yearsRange
                                                  )
    
                        print(f'    Искомых объектов в году {year}: {targetCount}, а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))
# ********** из фрагмента 2.1.1
                        if len(itemS) < targetCount:
                            print(f'  Для года {year} проход по значениям аргумента order,'
                                  , 'внутри которых проход по всем страницам выдачи')
                            for order in orderS:
                                addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                         API_keyS=API_keyS,
                                                                                         channelIdForSearch=channelIdForSearch,
                                                                                         channelType=channelType,
                                                                                         contentType=contentType,
                                                                                         eventType=eventType,
                                                                                         iteration=iteration,
                                                                                         keyOrder=keyOrder,
                                                                                         location=location,
                                                                                         locationRadius=locationRadius,
                                                                                         order=order,
                                                                                         publishedAfter = f'{year}-01-01T00:00:00Z',
                                                                                         publishedBefore = f'{year + 1}-01-01T00:00:00Z',
                                                                                         pageToken=None,
                                                                                         q=q,
                                                                                         regionCode=regionCode,
                                                                                         relevanceLanguage=relevanceLanguage,
                                                                                         safeSearch=safeSearch,
                                                                                         topicId=topicId,
                                                                                         videoCaption=videoCaption,
                                                                                         videoCategoryId=videoCategoryId,
                                                                                         videoDefinition=videoDefinition,
                                                                                         videoDimension=videoDimension,
                                                                                         videoDuration=videoDuration,
                                                                                         videoEmbeddable=videoEmbeddable,
                                                                                         videoLicense=videoLicense,
                                                                                         videoPaidProductPlacement=videoPaidProductPlacement,
                                                                                         videoType=videoType,
                                                                                         videoSyndicated=videoSyndicated,
                                                                                         year=year
                                                                                         )
                                itemS = dfsProcessor(
                                                      channelIdForSearch=channelIdForSearch,
                                                      coLabFolder=coLabFolder,
                                                      complicatedNamePart=complicatedNamePart,
                                                      contentType=contentType,
                                                      fileFormatChoice=fileFormatChoice,
                                                      dfAdd=addItemS,
                                                      dfFinal=itemS,
                                                      dfIn=itemS,
                                                      goS=goS,
                                                      method=method,
                                                      q=q,
                                                      rootName=rootName,
                                                      slash=slash,
                                                      stageTarget=stage,
                                                      targetCount=targetCount,
                                                      momentCurrent=momentCurrent,
                                                      year=year,
                                                      yearsRange=yearsRange
                                                      )
    
                                print(
f'    Для года {year} проход по всем следующим страницам с выдачей с тем же значением аргумента order:', order
                                      )
                                while ('nextPageToken' in response.keys()) & (len(itemS) < targetCount) & (len(response["items"]) > 0):
                                    pageToken = response['nextPageToken']
                                    addItemS, goS, iteration, keyOrder, response = bigSearch(
                                                                                             API_keyS=API_keyS,
                                                                                             channelIdForSearch=channelIdForSearch,
                                                                                             channelType=channelType,
                                                                                             contentType=contentType,
                                                                                             eventType=eventType,
                                                                                             iteration=iteration,
                                                                                             keyOrder=keyOrder,
                                                                                             location=location,
                                                                                             locationRadius=locationRadius,
                                                                                             order=order,
                                                                                             publishedAfter = f'{year}-01-01T00:00:00Z',
                                                                                             publishedBefore = f'{year + 1}-01-01T00:00:00Z',
                                                                                             pageToken=pageToken,
                                                                                             q=q,
                                                                                             regionCode=regionCode,
                                                                                             relevanceLanguage=relevanceLanguage,
                                                                                             safeSearch=safeSearch,
                                                                                             topicId=topicId,
                                                                                             videoCaption=videoCaption,
                                                                                             videoCategoryId=videoCategoryId,
                                                                                             videoDefinition=videoDefinition,
                                                                                             videoDimension=videoDimension,
                                                                                             videoDuration=videoDuration,
                                                                                             videoEmbeddable=videoEmbeddable,
                                                                                             videoLicense=videoLicense,
                                                                                             videoPaidProductPlacement=videoPaidProductPlacement,
                                                                                             videoType=videoType,
                                                                                             videoSyndicated=videoSyndicated,
                                                                                             year=year
                                                                                             )
                                    itemS = dfsProcessor(
                                                          channelIdForSearch=channelIdForSearch,
                                                          coLabFolder=coLabFolder,
                                                          complicatedNamePart=complicatedNamePart,
                                                          contentType=contentType,
                                                          fileFormatChoice=fileFormatChoice,
                                                          dfAdd=addItemS,
                                                          dfFinal=itemS,
                                                          dfIn=itemS,
                                                          goS=goS,
                                                          method=method,
                                                          q=q,
                                                          rootName=rootName,
                                                          slash=slash,
                                                          stageTarget=stage,
                                                          targetCount=targetCount,
                                                          momentCurrent=momentCurrent,
                                                          year=year,
                                                          yearsRange=yearsRange
                                                          )
                            print(
f'''    Искомых объектов {targetCount}, а найденных с добавлением сегментирования по году (год {year}) и включением аргумента order: {len(itemS)}
'''
                                  )
                        else:
                            print('  Все искомые объекты в году', year, 'найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
                        year -= 1
                        if yearMinByUser != None: # если пользователь ограничил временнОй диапазон
                            if (year) <= yearMinByUser:
                                goC = False
                                print(f'Завершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser} (с точностью до года)\n')
    
# 2.1.3 Экспорт выгрузки метода search и опциональное завершение скрипта
            df2file.df2fileShell(
                                 complicatedNamePart=complicatedNamePart,
                                 dfIn=itemS,
                                 fileFormatChoice=fileFormatChoice,
                                 method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                                 coLabFolder=coLabFolder,
                                 currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                 )
        elif stage < stageTarget:
            print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}_Temporal"')

        print(
'''
Выгрузка метода search содержит НЕ ВСЕ доступные для выгрузки из API YouTube характеристки контента
--- Если хотите выгрузить дополнительные характеристики (ссылки для ознакомления с ними появятся ниже), нажмите Enter
--- Если НЕ хотите их выгрузить, нажмите пробел и затем Enter. Тогда исполнение скрипта завершится'''
              )
    
        if len(input()) > 0:
            print('Скрипт исполнен')
            if os.path.exists(rootName):
                print(
'Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта'
                      )
                shutil.rmtree(rootName, ignore_errors=True)
            warnings.filterwarnings("ignore")
            print(
'Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть',
'Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'
                  )
            if returnDfs: return itemS, playlistVideoChannelS, videoS, commentReplieS, channelS
            sys.exit()

# 2.2 Выгрузка дополнительных характеристик и контента методами playlists и playlistItems, videos, commentThreads и comments, channels
# 2.2.0 Этап stage = 3
    stage = 3

# 2.2.1 Выгрузка дополнительных характеристик плейлистов ИЛИ тот самый случай "в противном случае", когда search следует заменить на cannels + playlistItems
    snippetContentType = 'playlist'
    if len(itemS) > 0: # если использовался search..
        if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # .. и в его выдаче есть плейлисты
            playlistIdS = itemS[itemS['id.kind'] == f'youtube#{snippetContentType}']
            playlistIdS =\
            playlistIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in playlistIdS.columns else playlistIdS['id'].to_list()
            playlistS, playlistVideoChannelS = playListProcessor(
                                                                 API_keyS=API_keyS,
                                                                 channelIdForSearch=channelIdForSearch,
                                                                 coLabFolder=coLabFolder,
                                                                 complicatedNamePart=complicatedNamePart,
                                                                 contentType=contentType,
                                                                 dfFinal=itemS,
                                                                 expiriencedMode=expiriencedMode,
                                                                 fileFormatChoice=fileFormatChoice,
                                                                 goS=goS,
                                                                 keyOrder=keyOrder,
                                                                 momentCurrent=momentCurrent,
                                                                 playlistIdS=playlistIdS,
                                                                 q=q,
                                                                 rootName=rootName,
                                                                 slash=slash,
                                                                 snippetContentType=snippetContentType,
                                                                 stage=stage,
                                                                 targetCount=targetCount,
                                                                 year=year,
                                                                 yearsRange=yearsRange
                                                                 )
    else: # если НЕ использовался search (то есть пользователь подал id канала)
        channelS = channelProcessor(
                                    API_keyS=API_keyS,
                                    channelIdForSearch=channelIdForSearch,
                                    coLabFolder=coLabFolder,
                                    complicatedNamePart=complicatedNamePart,
                                    contentType=contentType,
                                    dfIn=itemS,
                                    expiriencedMode=expiriencedMode,
                                    fileFormatChoice=fileFormatChoice,
                                    goS=goS,
                                    keyOrder=keyOrder,
                                    momentCurrent=momentCurrent,
                                    playlistS=playlistS,
                                    q=q,
                                    rootName=rootName,
                                    slash=slash,
                                    snippetContentType=snippetContentType,
                                    stage=stage,
                                    targetCount=targetCount,
                                    year=year,
                                    yearsRange=yearsRange,
                                    videoS=videoS
                                    )
        playlistIdS = channelS['contentDetails.relatedPlaylists.uploads'].to_list()
        playlistS, playlistVideoChannelS = playListProcessor(
                                                             API_keyS=API_keyS,
                                                             channelIdForSearch=channelIdForSearch,
                                                             coLabFolder=coLabFolder,
                                                             complicatedNamePart=complicatedNamePart,
                                                             contentType=contentType,
                                                             dfFinal=channelS, # т.к. в отсутствие itemS channelS становится базовым датафреймом
                                                             expiriencedMode=expiriencedMode,
                                                             fileFormatChoice=fileFormatChoice,
                                                             goS=goS,
                                                             keyOrder=keyOrder,
                                                             momentCurrent=momentCurrent,
                                                             playlistIdS=playlistIdS,
                                                             q=q,
                                                             rootName=rootName,
                                                             slash=slash,
                                                             snippetContentType=snippetContentType,
                                                             stage=stage,
                                                             targetCount=targetCount,
                                                             year=year,
                                                             yearsRange=yearsRange
                                                             )        
    # print('playlistIdS:', playlistIdS) # для отладки

# 2.2.2 Выгрузка дополнительных характеристик видео
    method = 'videos'
    videoIdS = []
    if len(itemS) > 0: # если использовался search..
        snippetContentType = 'video'
        if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # .. и в его выдаче есть видео
            print(
'В скрипте используются следующие аргументы метода', method, 'API YouTube: part=["snippet", "contentDetails", "localizations", "statistics", "status", "topicDetails"], id, maxResults .',
'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/videos'
                  )
            if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
            print('') # для отступа

            iteration = 0 # номер итерации применения текущего метода
            videoIdS = itemS[itemS['id.kind'] == f'youtube#{snippetContentType}']
            videoIdS = videoIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in videoIdS.columns else videoIdS['id'].to_list()

# ********** Дополнение списка id видео из itemS списком id видео из playlistS
    if len(playlistS) > 0:
        print(
'''--- Если стоит задача сформировать релевантную запросу базу видео и хотите пополнить список видео теми, которые составляют выгруженные плейлисты,
просто нажмите Enter (это увеличит совокупность выгруженных видео, но нет гарантии, что если плейлисты релевантны, то и все содержащиеся в них видео тоже релевантны)
--- Если НЕ хотите пополнить список, нажмите пробел и затем Enter'''
              )
        if len(input()) == 0:

            # Список списков, каждый из которых соответствует одному плейлисту
            playlistVideoId_list = playlistS['snippet.resourceId.videoId'].str.split(', ').to_list()
            # print('playlistVideoId_list:', playlistVideoId_list) # для отладки

            playlistVideoIdS = []
            for playlistVideoIdSnippet in playlistVideoId_list:
                playlistVideoIdS.extend(playlistVideoIdSnippet)
    
            videoIdS.extend(playlistVideoIdS)
            videoIdS = list(dict.fromkeys(videoIdS))

    # print(videoIdS) # для отладки

    if len(videoIdS) > 0:
        print(f'''Проход по видео{' порциями по 50 штук' if len(videoIdS) > 50 else ''} для выгрузки их характеристик (дополнительных к выруженным методом search)''')
        videoS = portionsProcessor(
                                   API_keyS=API_keyS,
                                   channelIdForSearch=channelIdForSearch,
                                   coLabFolder=coLabFolder,
                                   complicatedNamePart=complicatedNamePart,
                                   contentType=contentType,
                                   fileFormatChoice=fileFormatChoice,
                                   dfFinal=itemS,
                                   idS=videoIdS,
                                   keyOrder=keyOrder,
                                   method=method,
                                   momentCurrent=momentCurrent,
                                   q=q,
                                   rootName=rootName,
                                   slash=slash,
                                   stage=stage,
                                   targetCount=targetCount,
                                   year=year,
                                   yearsRange=yearsRange
                                   )

# ********** categoryId
        # Взять столбец snippet.categoryId, удалить из него дубликаты кодов категорий и помеcтить уникальные коды в список
        # display(videoS) # для отладки
        # print('videoS.columns:', videoS.columns) # для отладки
        uniqueCategorieS = videoS['snippet.categoryId'].drop_duplicates().to_list()
        # print('\nУникальные коды категорий в базе:', uniqueCategorieS, '\nЧисло уникальных категорий в базе:', len(uniqueCategorieS))
        try:
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            response = youtube.videoCategories().list(part='snippet', id=uniqueCategorieS).execute()

        except: goC, goS, keyOrder, problemItemId = errorProcessor(
                                                                    errorDescription=sys.exc_info(),
                                                                    keyOrder=keyOrder,
                                                                    sourceId=None
                                                                    )
        # Оформить как датафрейм id категорий из списка uniqueCategorieS и их расшифровки
        categoryNameS = pandas.json_normalize(response['items'])

        # Заменить индексы датафрейма с расшифровками значениями столбца id
        categoryNameS.index = categoryNameS['id'].to_list()

        # Добавить расшифровки категорий в новый столбец categoryName датафрейма с видео
        for row in categoryNameS.index:
            videoS.loc[videoS['snippet.categoryId'] == row, 'categoryName'] = categoryNameS['snippet.title'][row]

        df2file.df2fileShell(
                             complicatedNamePart=complicatedNamePart,
                             dfIn=videoS,
                             fileFormatChoice=fileFormatChoice,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method, # чтобы избавиться от лишней точки в имени файла
                             coLabFolder=coLabFolder,
                             currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             )
        commentS = pandas.DataFrame() # не в следующем ченке, чтобы иметь возможность перезапускать его, не затирая промежуточный результат выгрузки

# 2.2.3 Выгрузка комментариев к видео
        print(
'\n--- Если хотите выгрузить (в отдельный файл) комментарии к видео,',
f'содержащимся в файле "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart} {method}.xlsx" директории "{momentCurrent.strftime("%Y%m%d")}{complicatedNamePart}",',
'просто нажмите Enter, но учтите, что выгрузка может занять минуты и даже часы',
'\n--- Если НЕ хотите выгрузить комментарии, нажмите пробел и затем Enter'
              )
        if len(input()) == 0:

# ********** commentS
            # commentS = pandas.DataFrame() # фрагмент вынесен в предыдущий ченк, чтобы иметь возможность перезапускать этот чанк,
            # не затирая промежуточный результат выгрузки
            maxResults = 100
            method = 'commentThreads'
            part = 'id, replies, snippet'
            problemVideoIdS = []
            print(
'В скрипте используются следующие аргументы метода', method, 'API YouTube: part=["snippet", "id", "replies"], maxResults, videoId .',
'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/commentThreads'
                  )
            if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
            print('') # для отступа

            # Переназначить объект videoIdS для целей текущего чанка
            videoIdS = videoS[videoS['statistics.commentCount'].notna()]
            videoIdS = videoIdS[videoIdS['statistics.commentCount'].astype(int) > 0]
            videoIdS = videoIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in videoIdS.columns else videoIdS['id'].to_list()
            print('Число видео с комментариями:', len(videoIdS))

            print('\nВыгрузка родительских (topLevel) комментариев')
            commentS = pandas.DataFrame()
            for videoId in tqdm(videoIdS):
            # for videoId in videoS['id'][4576:]: # для отладки
                # print('videoId', videoId) # для отладки
                commentsAdditional, goS, keyOrder, problemVideoId = downloadComments(
                                                                                     API_keyS=API_keyS,
                                                                                     sourceId=videoId,
                                                                                     keyOrder=keyOrder,
                                                                                     method=method
                                                                                     )
                if problemVideoId != None: problemVideoIdS.append(problemVideoId)
                commentS = dfsProcessor(
                                         channelIdForSearch=channelIdForSearch,
                                         coLabFolder=coLabFolder,
                                         complicatedNamePart=complicatedNamePart,
                                         contentType=contentType,
                                         fileFormatChoice=fileFormatChoice,
                                         dfAdd=commentsAdditional,
                                         dfFinal=itemS,
                                         dfIn=commentS,
                                         goS=goS,
                                         method=method,
                                         q=q,
                                         rootName=rootName,
                                         slash=slash,
                                         stageTarget=stage,
                                         targetCount=targetCount,
                                         momentCurrent=momentCurrent,
                                         year=year,
                                         yearsRange=yearsRange
                                         )
            commentS = commentS.drop(['kind', 'etag', 'id', 'snippet.channelId', 'snippet.videoId'], axis=1) # т.к. дублируются содержательно
            commentS = prefixDropper(commentS)
            df2file.df2fileShell(
                                 complicatedNamePart=complicatedNamePart,
                                 dfIn=commentS,
                                 fileFormatChoice=fileFormatChoice,
                                 method='commentS',
                                 coLabFolder=coLabFolder,
                                 currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                 )

# ********** replieS
            print('')
            if len(commentS[commentS['snippet.totalReplyCount'] > 0]) > 0: # есть ли хотя бы один отвеченный родительский (topLevel) комментарий?
                print('Проход по строкам всех родительских (topLevel) комментариев, имеющих ответы')
                replieS = pandas.DataFrame()
                for row in tqdm(commentS[commentS['snippet.totalReplyCount'] > 0].index):
                    addReplieS = pandas.json_normalize(commentS['replies.comments'][row])
    
                    # Записать разницу между ожданиями и реальностью в новый столбец `Недостача_ответов`
                    commentS.loc[row, 'Недостача_ответов'] = commentS['snippet.totalReplyCount'][row] - len(addReplieS)
    
                    replieS = pandas.concat([replieS, addReplieS]).reset_index(drop=True)
    
                replieS.loc[:, 'snippet.totalReplyCount'] = 0
                replieS.loc[:, 'Недостача_ответов'] = 0
                replieS = prefixDropper(replieS)
                df2file.df2fileShell(
                                     complicatedNamePart=complicatedNamePart,
                                     dfIn=replieS,
                                     fileFormatChoice=fileFormatChoice,
                                     method='replieS',
                                     coLabFolder=coLabFolder,
                                     currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                     )
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
                commentReplieS = dfsProcessor(
                                               channelIdForSearch=channelIdForSearch,
                                               coLabFolder=coLabFolder,
                                               complicatedNamePart=complicatedNamePart,
                                               contentType=contentType,
                                               fileFormatChoice=fileFormatChoice,
                                               dfAdd=replieS,
                                               dfFinal=itemS,
                                               dfIn=commentReplieS,
                                               goS=goS,
                                               method=method,
                                               q=q,
                                               rootName=rootName,
                                               slash=slash,
                                               stageTarget=stage,
                                               targetCount=targetCount,
                                               momentCurrent=momentCurrent,
                                               year=year,
                                               yearsRange=yearsRange
                                               )
                method = 'comments'
                part = 'id, snippet'
                textFormat = 'plainText' # = 'html' по умолчанию
                problemCommentIdS = []
                replieS = pandas.DataFrame() # зачем? См. этап 4.2 ниже
                print(
'\nВ скрипте используются следующие аргументы метода', method, 'API YouTube: part=["snippet", "id"], maxResults, parentId, textFormat .',
'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.',
'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/commentThreads'
                      )
                if expiriencedMode == False: input('--- После прочтения этой инструкции нажмите Enter')
                print('') # для отступа

                print('Проход по id всех родительских (topLevel) комментариев с недостачей ответов для выгрузки этих ответов')
                commentIdS = commentReplieS['id'][commentReplieS['Недостача_ответов'] > 0]
                for commentId in tqdm(commentIdS):
                    page = 0 # номер страницы выдачи
                    repliesAdditional, goS, keyOrder, problemCommentId = downloadComments(
                                                                                          API_keyS=API_keyS,
                                                                                          sourceId=commentId,
                                                                                          keyOrder=keyOrder,
                                                                                          method=method
                                                                                          )
                    if problemCommentId != None: problemCommentIdS.append(problemCommentId)
                    replieS = dfsProcessor(
                                            channelIdForSearch=channelIdForSearch,
                                            coLabFolder=coLabFolder,
                                            complicatedNamePart=complicatedNamePart,
                                            contentType=contentType,
                                            fileFormatChoice=fileFormatChoice,
                                            dfAdd=repliesAdditional,
                                            dfFinal=itemS,
                                            dfIn=replieS,
                                            goS=goS,
                                            method=method,
                                            q=q,
                                            rootName=rootName,
                                            slash=slash,
                                            stageTarget=stage,
                                            targetCount=targetCount,
                                            momentCurrent=momentCurrent,
                                            year=year,
                                            yearsRange=yearsRange
                                            )
                print(
'Ответов выгружено', len(replieS), '; проблемные родительские (topLevel) комментарии:', problemCommentIdS if len(problemCommentIdS) > 0  else 'отсутствуют\n'
                      )
    
                # Для совместимости датафреймов добавить столбцы`snippet.totalReplyCount` и `Недостача_ответов`
                replieS.loc[:, 'snippet.totalReplyCount'] = 0
                replieS.loc[:, 'Недостача_ответов'] = 0
    
                # Удалить столбец `snippet.parentId`, т.к. и из столбца `id` всё ясно
                replieS = replieS.drop('snippet.parentId', axis=1)
    
                commentReplieS = dfsProcessor(
                                               channelIdForSearch=channelIdForSearch,
                                               coLabFolder=coLabFolder,
                                               complicatedNamePart=complicatedNamePart,
                                               contentType=contentType,
                                               fileFormatChoice=fileFormatChoice,
                                               dfAdd=replieS,
                                               dfFinal=itemS,
                                               dfIn=commentReplieS,
                                               goS=goS,
                                               method=method,
                                               q=q,
                                               rootName=rootName,
                                               slash=slash,
                                               stageTarget=stage,
                                               targetCount=targetCount,
                                               momentCurrent=momentCurrent,
                                               year=year,
                                               yearsRange=yearsRange
                                               )
                df2file.df2fileShell(
                                     complicatedNamePart=complicatedNamePart,
                                     dfIn=commentReplieS,
                                     fileFormatChoice=fileFormatChoice,
                                     method='commentReplieS',
                                     coLabFolder=coLabFolder,
                                     currentMoment=momentCurrent.strftime("%Y%m%d_%H%M") # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                     )
            else: print('Нет ни одного откомментированного родительского (topLevel) комментария')

# 2.2.4 Выгрузка дополнительных характеристик каналов
    if len(itemS) > 0: # если использовался search..
        snippetContentType = 'channel'
        if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # .. и в его выдаче есть каналы
            channelS = channelProcessor(
                                        API_keyS=API_keyS,
                                        channelIdForSearch=channelIdForSearch,
                                        coLabFolder=coLabFolder,
                                        complicatedNamePart=complicatedNamePart,
                                        contentType=contentType,
                                        dfIn=itemS,
                                        expiriencedMode=expiriencedMode,
                                        fileFormatChoice=fileFormatChoice,
                                        goS=goS,
                                        keyOrder=keyOrder,
                                        momentCurrent=momentCurrent,
                                        playlistS=playlistS,
                                        q=q,
                                        rootName=rootName,
                                        slash=slash,
                                        snippetContentType=snippetContentType,
                                        stage=stage,
                                        targetCount=targetCount,
                                        year=year,
                                        yearsRange=yearsRange,
                                        videoS=videoS
                                        )

# 2.2.5 Экспорт выгрузки метода search и финальное завершение скрипта
    print('Скрипт исполнен. Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
    if os.path.exists(rootName):
        print(
'Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта'
              )
        shutil.rmtree(rootName, ignore_errors=True)
    if returnDfs: return itemS, playlistVideoChannelS, videoS, commentReplieS, channelS

# warnings.filterwarnings("ignore")
# print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть')
# input()
# sys.exit()

# https://stackoverflow.com/questions/30475309/get-youtube-trends-v3-country-wise-in-json -- про тренды
