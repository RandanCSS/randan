#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and maximizes YouTube content extraction using the platform's official API method search
(RU) Модуль для упрощения выгрузки контента YouTube методом его API search и максимизации размера этой выгрузки
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

        from randan.tools import calendarWithinYear, coLabAdaptor, df2file, files2df, scrapingTools, textPreprocessor # модули для
            # (а) работы с календарём конкретного года
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (г) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (д) упрощения скрапинга
            # (е) предобработки неформализованного текста

        from tqdm import tqdm
        import os, pandas, re, shutil, time, traceback, warnings
        import googleapiclient.discovery as api
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

# 1 Авторские функции
# 1.0 для метода search из API YouTube, помогающая работе с ключами
def bigSearch(api_keyS,
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
              year):

    goC = True
    goS = True
    goToPlayList = False
    itemS_additional = pandas.DataFrame() # принудительная выдача на случай неуспеха request.execute()

    response = {'kind': 'youtube#searchListResponse',
                'pageInfo': {'totalResults': 0, 'resultsPerPage': 0},
                'items': []} # принудительная выдача на случай неуспеха request.execute()

    MAX_ATTEMPTS = 3
    attempt_network = 0
    while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
        try:
            youtube = api.build('youtube', 'v3', developerKey=api_keyS[keyOrder])
            # print('contentType:', contentType) # для отладки

            request = youtube.search().list(channelId=channelIdForSearch,
                                            channelType=channelType,
                                            eventType=eventType,
                                            location=location,
                                            locationRadius=locationRadius,
                                            maxResults=50,
                                            order=order,
                                            pageToken=pageToken,
                                            part='snippet',
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
                                            videoSyndicated=videoSyndicated)

            response = request.execute()
            itemS_additional = pandas.json_normalize(response['items'])

            # Для визуализации процесса
            print(
'      Итерация №', iteration, ', number of items', len(response['items']), '' if not year else f', year {year}', '' if not order else f', order {order}', '          ', end='\r'
            )

            attempt_network = 0 # сброс при успех
            iteration += 1
            goC = False

        except Exception:
            print('Exception в bigSearch') # для отладки
            goC, goS, goToPlayList, keyOrder, problemItemId = errorProcessor(errorDescription=sys.exc_info(),
                                                                             keyOrder=keyOrder,
                                                                             sourceId=channelIdForSearch)

            if goC:
                print(f'Попытка сетевого соединения № {attempt_network} из {MAX_ATTEMPTS} НЕуспешна')                
                attempt_network += 1
                if attempt_network > MAX_ATTEMPTS:
                    print('!!! Превышено число попыток сетевого соединения')
                    goC = False
                    goS = False

    return itemS_additional, goS, goToPlayList, iteration, keyOrder, response # от response отказаться нельзя, т.к. в нём много важных ключей, даже если их значения нули

# 1.1 для обработки выдачи метода channels, помогающая работе с ключами
def channelProcessor(api_keyS,
                     channelIdForSearch,
                     complicatedNamePart,
                     contentType,
                     df_final,
                     experiencedMode,
                     fileFormatChoice,
                     folder,
                     goS,
                     keyOrder,
                     momentCurrent,
                     playlistS,
                     q,
                     name_in_folderCurrent,
                     snippetContentType,
                     stage,
                     targetCount,
                     year,
                     yearsRange,
                     videoS):

    channelS = pandas.DataFrame()
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    if len(df_final) > 0: # если использовался search и успешно, id каналов берутся из него
        channelIdS = df_final[df_final['id.kind'] == f'youtube#{snippetContentType}']
        if len(channelIdS) > 0:
            channelIdS =\
                channelIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in channelIdS.columns else channelIdS['id'].to_list()

    else: channelIdS = [channelIdForSearch] # в другом случае id канала подаётся пользователем
    # channelIdS = channelIdS[:5] # для отладки
    # print(channelIdS)

    method = 'channels'

    print(
f'''В скрипте используются следующие аргументы метода {method} API YouTube: part=['snippet', 'brandingSettings', 'contentDetails', 'id', 'localizations', 'statistics', 'status', 'topicDetails'], id, maxResults .
Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/channels'''
          )

    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
    print('') # для отступа

# ********** Дополнение списка id каналов из df_final списком id каналов из playlistS
    if (len(playlistS) > 0) | (len(videoS) > 0):

        print(
'''--- Если стоит задача сформировать релевантную запросу базу каналов и хотите пополнить список каналов теми, к которым относятся выгруженные плейлисты и видео,
просто нажмите Enter (это увеличит совокупность выгруженных каналов, но нет гарантии, что если плейлисты и видео релевантны, то и каналы, к которым они относятся, тоже релевантны)
--- Если НЕ хотите пополнить список, нажмите пробел и затем Enter'''
              )

        if len(input()) == 0:
            if len(playlistS) > 0:

                playlistChannelId_list = playlistS['snippet.videoOwnerChannelId'].str.split(', ').to_list()
                    # список списков, каждый из которых соответствует одному плейлисту

                playlistChannelIdS = []
                for snippet in playlistChannelId_list: playlistChannelIdS.extend(snippet)
                channelIdS.extend(playlistChannelIdS)
                channelIdS = list(dict.fromkeys(channelIdS))

# ********** Дополнение списка id каналов из df_final списком id каналов из videoS
            if len(videoS) > 0:
                channelIdS.extend(videoS['snippet.channelId'].to_list())
                channelIdS = list(dict.fromkeys(channelIdS))

    if len(channelIdS) > 0:

        print(f'''Проход по каналам{' порциями по 50 штук' if len(channelIdS) > 50 else ''} для выгрузки их характеристик''')
        if len(df_final) > 0 and 'kind' in df_final.columns:
            if 'search' in df_final['kind'].drop_duplicates().iloc[0]: # другими словами, если search использовался, потому что мог не search , а channels сформировать df_final
                print('-- дополнительных к выруженным методом search')

        channelS = portionsProcessor(api_keyS,
                                     channelIdForSearch,
                                     complicatedNamePart,
                                     contentType, # snippetContentType -- не то же самое, что contentType , т.к. contentType исходно подаётся пользователем
                                     df_final,
                                     fileFormatChoice,
                                     folder,
                                     idS=channelIdS,
                                     keyOrder=keyOrder,
                                     method=method,
                                     momentCurrent=momentCurrent,
                                     q=q,
                                     name_in_folderCurrent=name_in_folderCurrent,
                                     stage=stage,
                                     targetCount=targetCount,
                                     year=year,
                                     yearsRange=yearsRange)

        if len(channelS) > 0:
            df2file.df2fileShell(complicatedNamePart,
                                 currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                 df_in=channelS,
                                 fileFormatChoice=fileFormatChoice,
                                 folder=folder,
                                 method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method) # чтобы избавиться от лишней точки в имени файла

    return channelS

# 1.2 для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessor(channelIdForSearch,
                 complicatedNamePart,
                 contentType,
                 fileFormatChoice,
                 df_additional,
                 df_final, # на обработке какой бы ни было выгрузки не возникла бы непреодолима ошибка, сохранить следует выгрузку метода search
                 df_in,
                 folder,
                 goS, # единственная из функций, принимающая этот аргумент
                 method,
                 q,
                 name_in_folderCurrent,
                 stage,
                 targetCount,
                 momentCurrent,
                 year,
                 yearsRange):

    df = pandas.DataFrame()
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    print('folder:', folder) # для отладки

    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if len(df_additional) > 0 or len(df_in) > 0: # оба df пусты -- в случае применения функции dfsProcessor при завершении исполнения скрипта пользователем
        df = pandas.concat([df_in, df_additional])
        columnsForCheck = []
        for column in df.columns: # выдача многих методов содержит столбец id, он оптимален для проверки дублирующихся строк
            if 'id' == column: columnsForCheck.append(column)

        if not columnsForCheck: # для выдач, НЕ содержащих столбец id, проверка дублирующихся строк возможна по столбцам, содержащим в имени id
            for column in df.columns:
                if 'id.' in column: columnsForCheck.append(column)

        # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck) # для отладки

        df = df.drop_duplicates(columnsForCheck, keep='last').reset_index(drop=True) # при дублировании объектов из itemS из Temporal и от пользователя и новых объектов, оставить новые

# Сохранение следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
    if not goS:

        print(
f'''Поскольку исполнение скрипта натолкнулось на ошибку или принудительно прервано, сохраняю выгруженный контент и текущий этап поиска в директорию "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"'''
              )

        temporal_exportPath = folder + f"{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"
        print('temporal_exportPath:', temporal_exportPath) # для отладки
        if not os.path.exists(temporal_exportPath):
            os.makedirs(temporal_exportPath)
            print(f"Директория '{temporal_exportPath}' создана")
        # else:
            # print(f'''Директория "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal" существует''')

        scrapingTools.containerExport(temporal_exportPath + slash + 'channelIdForSearch.txt', channelIdForSearch if channelIdForSearch else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'contentType.txt', contentType if contentType else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'method.txt', method)
        scrapingTools.containerExport(temporal_exportPath + slash + 'q.txt', q if q else '')
        scrapingTools.containerExport(temporal_exportPath + slash + 'stageTarget.txt', str(stage)) # stage и stageTarget принимает значения [0; 3]
        scrapingTools.containerExport(temporal_exportPath + slash + 'targetCount.txt', str(targetCount) if targetCount else '0')
        scrapingTools.containerExport(temporal_exportPath + slash + 'year.txt', str(year)) # год, на котором остановилось исполнение скрипта
        scrapingTools.containerExport(temporal_exportPath + slash + 'yearsRange.txt', yearsRange if yearsRange else '')

        df2file.df2fileShell(complicatedNamePart=f'{complicatedNamePart}_Temporal',
                             currentMoment=momentCurrent.strftime('%Y%m%d'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                             df_in=df_final,
                             fileFormatChoice=fileFormatChoice,
                             folder=temporal_exportPath,
                             method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method)
                                 # чтобы избавиться от лишней точки в имени файла  

        print('Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')

    return df

# 1.3 для выгрузки комментариев
def downloadComments(api_keyS, keyOrder, method, sourceId):
    goS = True
    commentS = pandas.DataFrame()
    pageToken = None
    problemItemId = None
    youtube = api.build('youtube', 'v3', developerKey=api_keyS[keyOrder])
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
                print('Exception в downloadComments') # для отладки
                goC, goS, goToPlayList, keyOrder, problemItemId = errorProcessor(errorDescription=sys.exc_info(), keyOrder=keyOrder, sourceId=sourceId)

        commentS = pandas.concat([commentS, pandas.json_normalize(response['items'])])
        if 'nextPageToken' in response.keys():
            pageToken = response['nextPageToken']
            # print('nextPageToken', pageToken) # для отладки
        else: break

    return commentS, goS, keyOrder, problemItemId

# 1.4 для обработки ошибок
def errorProcessor(errorDescription, keyOrder, sourceId):
    goC = True
    goS = True
    goToPlayList = False
    problemItemId = sourceId
    # print(errorDescription) # для отладки
    print(errorDescription[1]) # для отладки
    if 'Request contains an invalid argument' in str(errorDescription[1]):
        # print('errorDescription:', errorDescription) # для отладки
        print('!!! Похоже, подан id не канала, а плейлиста, поэтому текстовый запрос-фильтр и временнОй диапазон (если таковые были поданы) будут проигнорированы; пробую перейти к методам playlistitems и playlists')
        goC = False # нет смысла в новых итерациях цикла (вовне этой функции)
        goToPlayList = True

    elif ('exceeded' in str(errorDescription[1]).lower()) & ('quota' in str(errorDescription[1]).lower()):
        print('!!! Похоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
        # print('  keyOrder ДО смены ключа', keyOrder) # для отладки
        keyOrder += 1 # смена ключа
        # print('  keyOrder ПОСЛЕ смены ключа', keyOrder) # для отладки

    elif ('disabled' in str(errorDescription[1]).lower()) & ('key' in str(errorDescription[1]).lower()):
        print('!!! Похоже, текущий ключ деактивирован владельцем или Гуглом; пробую перейти к следующему ключу')
        # print('  keyOrder ДО смены ключа', keyOrder) # для отладки
        keyOrder += 1 # смена ключа
        # print('  keyOrder ПОСЛЕ смены ключа', keyOrder) # для отладки

    elif ('not valid' in str(errorDescription[1]).lower()) & ('key' in str(errorDescription[1]).lower()):
        print('!!! Похоже, текущий ключ не валиден; пробую перейти к следующему ключу')
        # print('  keyOrder ДО смены ключа', keyOrder) # для отладки
        keyOrder += 1 # смена ключа
        # print('  keyOrder ПОСЛЕ смены ключа', keyOrder) # для отладки

    elif 'index out of range' in str(errorDescription[1]):
        print('!!! Похоже, ключи закончились. Что делать? (а) Подождите сутки для восстановления ключей или',
              '(б) подготовьте новый ключ, найдите и удалите файл credentialsYouTube.txt -- и запустите скрипт с начала')
        goS = False # нет смысла продолжать исполнение скрипта
        goC = False # и, следовательно, нет смысла в новых итерациях цикла (вовне этой функции)

    elif ('channel' in str(errorDescription[1]).lower()) | ('comment' in str(errorDescription[1]).lower()) | ('playlist' in str(errorDescription[1]).lower()) | ('video' in str(errorDescription[1]).lower()):
        print('  Проблема может быть связана с ограничением доступа к обрабатываемому объекту, поэтому фиксирую его id:', problemItemId)
        # print('problemItemId:', problemItemId) # для отладки
        goC = False # и, следовательно, нет смысла в новых итерациях цикла (вовне этой функции)        

    # elif 'TimeoutError' in str(errorDescription[0]):
    #     print('!!! Похоже, проблема в слишком высокой частоте запросов к удалённому серверу; засыпаю на 10 секунд')
    #     time.sleep(10)

    elif 'KeyboardInterrupt' in str(errorDescription):
        print(sys.exc_info()[1]) # для отладки
        goS = False # нет смысла продолжать исполнение скрипта
        goC = False # и, следовательно, нет смысла в новых итерациях цикла (вовне этой функции)

    elif isinstance(errorDescription[1], (ConnectionResetError, ConnectionAbortedError, TimeoutError)) \
        or 'WinError 10054' in str(errorDescription[1]) \
        or 'Connection reset' in str(errorDescription[1]) \
        or 'TimeoutError' in str(errorDescription[0]):
        print('!!! Похоже, проблема с сетью или слишком высокая частота запросов; засыпаю на 10 секунд')
        time.sleep(10)

    else:
        print('!!! Похоже, проблема не в ограничении доступа к обрабатываемому объекту и не в истечении квоты текущего ключа((')
        goC = False # нет смысла повторного обращения к API ни с этим id, ни пока не ясна суть ошибки

    return goC, goS, goToPlayList, keyOrder, problemItemId

# 1.5 для визуализации процесса через итерации
def iterationVisualization(idS, iteration, portion, response):
    if len(idS) > 0: iterationUpperBound = int(str(len(idS) / portion).split('.')[0]) + 1 # дробная часть после деления числа idS должна увеличить iterationUpperBound на единицу
    if 'items' not in response.keys():
        print("!!! Похоже, подан id не канала, а плейлиста, поэтому подаю пустой список в качестве значения словарного ключа 'items'")
        response['items'] = []

    print(
f'''  Порция № {iteration + 1}{f' из {iterationUpperBound}' if len(idS) > 0 else ''}.{f" Сколько в порции наблюдений? {len(response['items'])}" if len(idS) > 0 else ''}          ''', end='\r'
          )

# 1.6 для обработки выдачи методов playlists и playlistItems, помогающая работе с ключами
def playListProcessor(api_keyS,
                      channelIdForSearch,
                      complicatedNamePart,
                      contentType,
                      df_final,
                      experiencedMode,
                      fileFormatChoice,
                      folder,
                      goS,
                      keyOrder,
                      momentCurrent,
                      playlistIdS,
                      q,
                      name_in_folderCurrent,
                      stage,
                      targetCount,
                      year,
                      yearsRange):

    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    method = 'playlists'
    playlistS = pandas.DataFrame()
    playlistVideoChannelS = pandas.DataFrame()
    print(
f'''В скрипте используются следующие аргументы метода {method} API YouTube: part=['snippet', 'contentDetails', 'localizations', 'status'], id, maxResults .
Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/playlists'''
          )

    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
    print('') # для отступа

    if playlistIdS:
        # print('playlistIdS:', playlistIdS) # для отладки
        print(f'''Проход по плейлистам{' порциями по 50 штук' if len(playlistIdS) > 50 else ''} для выгрузки их характеристик''')
        if len(df_final) > 0 and 'kind' in df_final.columns:
            if 'search' in df_final['kind'].drop_duplicates().iloc[0]: # другими словами, если search использовался, потому что мог не search , а channels сформировать df_final
                print('-- дополнительных к выруженным методом search')

        playlistS = portionsProcessor(api_keyS,
                                      channelIdForSearch,
                                      complicatedNamePart,
                                      contentType,
                                      df_final,
                                      fileFormatChoice,
                                      folder,
                                      idS=playlistIdS,
                                      keyOrder=keyOrder,
                                      method=method,
                                      momentCurrent=momentCurrent,
                                      q=q,
                                      name_in_folderCurrent=name_in_folderCurrent,
                                      stage=stage,
                                      targetCount=targetCount,
                                      year=year,
                                      yearsRange=yearsRange)

        method = 'playlistItems'

        print(
f'''
В скрипте используются следующие аргументы метода {method} API YouTube: part=['snippet'], playlistId, maxResults .
Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/playlistitems'''
              )

        if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
        print('') # для отступа

        iteration = 0 # номер итерации применения текущего метода
        playlistVideoChannelS = pandas.DataFrame() # хотя датафреймы и глобальны как переменные, пусть и тут инициализируется
        # portion = 50
        print('Проход по плейлистам для выгрузки id видео, составляющих плейлисты, и каналов, к которым они принадлежат')
        for playlistId in playlistIdS:
            pageToken = None
            while True:
                goC = True
                while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
                    try:
                        youtube = api.build('youtube', 'v3', developerKey=api_keyS[keyOrder])
                        response = youtube.playlistItems().list(part='snippet', maxResults=50, pageToken=pageToken, playlistId=playlistId).execute()
                        playlistVideoChannelS_additional = pandas.json_normalize(response['items'])
                        playlistVideoChannelS = dfsProcessor(channelIdForSearch,
                                                             complicatedNamePart,
                                                             contentType,
                                                             fileFormatChoice,
                                                             df_additional=playlistVideoChannelS_additional,
                                                             df_final=df_final,
                                                             df_in=playlistVideoChannelS,
                                                             folder=folder,
                                                             goS=goS,
                                                             method=method,
                                                             q=q,
                                                             name_in_folderCurrent=name_in_folderCurrent,
                                                             stage=stage,
                                                             targetCount=targetCount,
                                                             momentCurrent=momentCurrent,
                                                             year=year,
                                                             yearsRange=yearsRange)

                        goC = False # если try успешно исполнился, то цикл прекращается
                    except: goC, goS, goToPlayList, keyOrder, problemItemId = errorProcessor(errorDescription=sys.exc_info(),
                                                                                             keyOrder=keyOrder,
                                                                                             sourceId=None)

                iterationVisualization(idS=None, iteration=iteration, portion=1, response=response) # для визуализации процесса через итерации; =portion
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

        if len(playlistS) > 0:
            df2file.df2fileShell(complicatedNamePart,
                                 currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                 df_in=playlistS,
                                 fileFormatChoice=fileFormatChoice,
                                 folder=folder,
                                 method='playlists')

    return playlistS, playlistVideoChannelS

# 1.7 для порционной выгрузки, когда метод предполагает подачу ему id порциями
def portionsProcessor(api_keyS,
                      channelIdForSearch,
                      complicatedNamePart,
                      contentType,
                      df_final,
                      fileFormatChoice,
                      folder,
                      idS,
                      keyOrder,
                      method,
                      momentCurrent,
                      q,
                      name_in_folderCurrent,
                      stage,
                      targetCount,
                      year,
                      yearsRange):

    # print('method', method) # для отладки
    bound = 0
    chplviS = pandas.DataFrame()
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    goS = True
    iteration = 0 # номер итерации применения текущего метода
    portion = 50

    # <Выгрузка плейлистов канала по id канала -- эксклюзивно для метода playlists >
    # Если пользователь подал id канала (то есть НЕ использовался search) и текущий метод -- playlists ,
    # то до цикла по bound дополнить список idS id всех плейлистов этого канала.
    # Вынесено из цикла по bound по двум причинам:
        #  (а) собрать ВСЕ плейлисты канала (через обход nextPageToken), а не только первую страницу выдачи;
        #  (б) не повторять этот сбор при каждой итерации цикла (в противном случае idS дублировался бы первыми 50 плейлистами)

    if method == 'playlists' and channelIdForSearch:
        pageToken = None
        while True: # прерывается командой break при отсутствии nextPageToken
            goC = True
            while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
                try:
                    youtube = api.build('youtube', 'v3', developerKey=api_keyS[keyOrder])
                    response = youtube.playlists().list(part='snippet, contentDetails, localizations, status',
                                                        channelId=channelIdForSearch,
                                                        maxResults=50,
                                                        pageToken=pageToken).execute()

                    goC = False # выход из while goC

                except:
                    print('\nОшибка внутри авторской функции portionsProcessor при сборе id плейлистов канала') # для отладки
                    goC, goS, goToPlayList, keyOrder, problemItemId = errorProcessor(errorDescription=sys.exc_info(),
                                                                                     keyOrder=keyOrder,
                                                                                     sourceId=channelIdForSearch)

            idS.extend(pandas.json_normalize(response['items'])['id'].tolist())
            if 'nextPageToken' in response.keys(): pageToken = response['nextPageToken']
            else: break
        idS = list(dict.fromkeys(idS))
            # так же дедуплицируются channelIdS в channelProcessor и videoIdS в searchByText ;
            # в частности, снимает возможное дублирование, если uploads-плейлист уже присутствует в idS
    # </Выгрузка плейлистов канала по id канала -- эксклюзивно для метода playlists >

    while bound < len(idS):
    # while bound < 100: # для отладки
        goC = True
        while goC: # цикл на случай истечения ключа: повторяет запрос после смены ключа
            try:
                youtube = api.build('youtube', 'v3', developerKey=api_keyS[keyOrder])
                if method == 'channels':
                    response = youtube.channels().list(part='snippet, brandingSettings, contentDetails, id, localizations, statistics, status, topicDetails',
                                                       id=idS[bound:bound + portion],
                                                       maxResults=50).execute()

                if method == 'playlists':
                    if channelIdForSearch: # если НЕ использовался search (то есть пользователь подал id канала)
                        response = youtube.playlists().list(part='snippet, contentDetails, localizations, status',
                                                            channelId=channelIdForSearch,
                                                            maxResults=50).execute()

                        idS.extend(pandas.json_normalize(response['items'])['id'].tolist())

                    # print('playlistIdS:', idS) # для отладки
                    response = youtube.playlists().list(part='snippet, contentDetails, localizations, status',
                                                        id=idS[bound:bound + portion],
                                                        maxResults=50).execute()

                if method == 'videos':
                    response = youtube.videos().list(part='snippet, contentDetails, localizations, statistics, status, topicDetails',
                                                     id=idS[bound:bound + portion],
                                                     maxResults=50).execute()

                addChplviS = pandas.json_normalize(response['items'])
                chplviS = dfsProcessor(channelIdForSearch=channelIdForSearch,
                                       complicatedNamePart=complicatedNamePart,
                                       contentType=contentType,
                                       fileFormatChoice=fileFormatChoice,
                                       df_additional=addChplviS,
                                       df_final=df_final, # itemS или channelS подаются как значение аргумента оборачивающей функции
                                       df_in=chplviS,
                                       folder=folder,
                                       goS=goS,
                                       method=method,
                                       momentCurrent=momentCurrent,
                                       q=q,
                                       name_in_folderCurrent=name_in_folderCurrent,
                                       stage=stage,
                                       targetCount=targetCount,
                                       year=year,
                                       yearsRange=yearsRange)

                goC = False # если try успешно исполнился, то цикл прекращается

            except:
                print('\nОшибка внутри авторской функции portionsProcessor') # для отладки
                goC, goS, goToPlayList, keyOrder, problemItemId = errorProcessor(errorDescription=sys.exc_info(),
                                                                                 keyOrder=keyOrder,
                                                                                 sourceId=None)

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
        if 'snippet.topLevelComment.' in column: column = column.replace('snippet.topLevelComment.', '')
        dfNewColumnS.append(column)

    df.columns = dfNewColumnS # перезаписать названия столбцов
    return df

# 2 Авторская функция исполнения скрипта
def searchByText(access_token=None,
                 channelIdForSearch=None,
                 contentType=None,
                 folder=None,
                 publishedAfter=None,
                 publishedBefore=None,
                 q=None,
                 channelType=None,
                 eventType=None,
                 location=None,
                 locationRadius=None,
                 regionCode=None,
                 relevanceLanguage=None,
                 returnDfs=False,
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
                 videoType=None):
    '''
    Функция для выгрузки характеристик контента YouTube методами его API: search, playlists & playlistItems, videos, commentThreads & comments, channels -- ключевым из которых выступает search
    Причём количество объектов выгрузки максимизируется путём её пересортировки аргументом order и сегментирования по годам

    Parameters
    ----------
    Аргументы этой функции аналогичны аргументам метода https://developers.google.com/youtube/v3/docs/search/list , за исключением аргументов channelIdForSearch returnDfs
             access_token : str
       channelIdForSearch : str -- это аналог аргумента channelId методов search и playlists и аргумента id метода channels ; также сюда можно подать вместо id канала id плейлиста
              contentType : str -- это аналог type
                   folder : str -- директория, с которой работает скрипт
           publishedAfter : str, readable by datetime
          publishedBefore : str, readable by datetime
                        q : str
              channelType : str
                eventType : str
                 location : str
           locationRadius : str
               regionCode : str
        relevanceLanguage : str
                returnDfs : bool -- в случае True функция возвращает пять итоговых датафреймов с выдачей методов (1) search, (2) playlists, (3) videos, (4) commentThreads и comments (общий датафрейм), (5) channels
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
    '''

    if not access_token and not channelIdForSearch and not contentType and not publishedAfter and not publishedBefore and not q and not channelType and not eventType\
        and not location and not locationRadius and not regionCode and not relevanceLanguage and not safeSearch and not topicId and not videoCaption and not videoCategoryId\
        and not videoDefinition and not videoDimension and not videoDuration and not videoEmbeddable and not videoLicense and not videoPaidProductPlacement and not videoSyndicated and not videoType and not returnDfs:
        # print('Пользователь не подал аргументы')

        experiencedMode = False
    else: experiencedMode = True

    if not experiencedMode:

        print(
'''    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрипты и файлы с данными). Но от пользователя требуется предварительно получить API key для авторизации в API YouTube по ключу (см. примерную видео-инструкцию: https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ). Для получения API key следует создать проект, авторизовать его, подключить к нему API нужного сервиса Google. Проект -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому сервисов Google и применения на этой основе API разных сервисов Google в рамках установленных Гуглом ограничений (the units of quota). Разные уровни авторизации -- это авторизация ключом (представляющим собой код) и полная авторизация (ключ + протокол Google OAuth 2.0, реализующийся в формате файла JSON). Авторизация ключом нужна, чтобы использовать любой метод любого API. Её достаточно, если выполнять действия, которые были бы доступны Вам как пользователю сервисов Google без Вашего входа в аккаунт: посмотреть видео, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления видео, то Вам придётся пройти полную авторизацию. Далее разные API как бы подключаются к проектам (кнопка Enable APIs and servises), используются, затем отключаются (кнопка Disable APIs).
    Квоты одного ключа может не хватить (quota is exceeded) для выгрузки всего предоставляемого ЮТьюбом по запросу пользователя контента. К счастью, использованный ключ ежесуточно восстанавливается ЮТьюбом. скрипт позволяет сохранить промежуточную выгрузку и после восстановления ключа автоматически продолжит её дополнять с момента остановки. В момент остановки появится надпись: 'Поскольку ключи закончились, исполнение скрипта завершаю. Подождите сутки для восстановления ключей или подготовьте новый ключ -- и запустите скрипт с начала', а исполнение скрипта прервётся. Не пугайтесь, нажмите OK и следуйте этой инструкции.'''
        )

    print(
'''    Скрипт нацелен на выгрузку характеристик контента YouTube семью методами его API: search, videos, commentThreads и comments, channels, playlists и playlistItems. Причём количество объектов выгрузки максимизируется путём её пересортировки и сегментирования по годам.
    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.
    Преимущества скрипта перед выгрузкой контента из YouTube вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API YouTube: гораздо быстрее, гораздо большее количество контента с одним и тем же ключом, не требуется тщательно изучать обширную документацию семи методов API YouTube (search, videos, commentThreads и comments, channels, playlists и playlistItems), выстроена логика обрашения к этим методам'''
    )

    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')

# 2.0 Настройки и авторизация
# 2.0.0 Некоторые базовые настройки запроса к API YouTube
    channelS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    commentReplieS = pandas.DataFrame() # чтобы обращаться к контейнеру, даже если функция, создающая его, не исполнялась
    fileFormatChoice = '.xlsx' # базовый формат сохраняемых файлов; формат .json добавляется опционально через наличие columnsToJSON
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    folderFile = None
    folderWithTemporal = None
    goS = True
    goToPlayList = False
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
    print('\nТекущий момент:', momentCurrent.strftime('%Y%m%d_%H%M'), '-- он будет использована для формирования имён создаваемых директорий и файлов')
    year = int(momentCurrent.strftime('%Y')) # в случае отсутствия пользовательского временнОго диапазона
        # с этого года возможно сегментирование по годам вглубь веков (пока выдача не пустая)
    yearMinByUser = None # в случае отсутствия пользовательского временнОго диапазона
    yearMaxByUser = None # в случае отсутствия пользовательского временнОго диапазона

# 2.0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
    nameS_in_folderCurrent = os.listdir(folder)
    api_keyS = scrapingTools.credentialsProcessor(access_token, 'YouTube', 'https://docs.google.com/document/d/15RpdkHe8C91AqD4IBE7PLr-naMfA56a_vFeMQQx8NY8', nameS_in_folderCurrent)

# 2.0.2 Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
    # Поиск данных
    print(
'Проверяю наличие директории Temporal с данными и их мета-данными, гипотетически сохранёнными при прошлом запуске скрипта, натолкнувшемся на ошибку'
          )
    for name_in_folderCurrent in nameS_in_folderCurrent:
        message =\
f"В текущей директории '{name_in_folderCurrent}' есть директория, похожая на искомую Temporal , но в ней не 6-7 файлов => если это она, то либо повреждена, либо создалась при безрезультатном запуске; тогда Вам следует удалить её вручную"

        if name_in_folderCurrent.endswith('Temporal') and 'YouTube' in name_in_folderCurrent:
            if len(os.listdir(name_in_folderCurrent)) < 9: print(message) # допустимо 9 или 10 файлов
            else:
                targetCountTemporal = scrapingTools.containerImport('targetCount', int)
                methodTemporal = scrapingTools.containerImport('method', str)
                yearTemporal = scrapingTools.containerImport('year', int)

                contentTypeTemporal = scrapingTools.containerImport('contentType', str)
                if contentTypeTemporal == '': contentTypeTemporal = None # для единообразия

                channelIdForSearchTemporal = scrapingTools.containerImport('channelIdForSearch', str)
                if channelIdForSearchTemporal == '': channelIdForSearchTemporal = None # для единообразия

                try: qTemporal = scrapingTools.containerImport('q', str)
                except Exception as excptn:
                    print('Exception в searchByText') # для отладки
                    print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
                    print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой
                    qTemporal = scrapingTools.containerImport('q', str)

                if qTemporal == '': qTemporal = None # для единообразия

                yearsRangeTemporal = scrapingTools.containerImport('yearsRange', str)
                if yearsRangeTemporal == '': yearsRangeTemporal = None # для единообразия

                stageTargetTemporal = scrapingTools.containerImport('stageTarget', int)
                stageTargetTemporal = int(stageTargetTemporal)

                print(f"Нашёл директорию '{name_in_folderCurrent}'. В этой директории следующие промежуточные результаты одного из прошлых запусков скрипта:",
                      '\n- было выявлено целевое число записей (totalResults)', targetCountTemporal,
                      '\n- скрипт остановился на методе', methodTemporal)

                if yearTemporal: print('- и на годе (при сегментировани по годам)', yearTemporal)
                print('- пользователь НЕ определил тип контента' if not contentTypeTemporal else  f"- пользователь определил тип контента как '{contentTypeTemporal}'")
                if contentTypeTemporal == 'video':
                    print('- пользователь НЕ выбрал конкретный канал для выгрузки видео' if not channelIdForSearchTemporal else  f"- пользователь выбрал канал с id '{channelIdForSearchTemporal}' для выгрузки видео")

                print('- пользователь НЕ сформулировал запрос-фильтр' if not qTemporal else  f"- пользователь сформулировал запрос-фильтр как '{qTemporal}'")
                print('- пользователь НЕ ограничил временнОй диапазон' if yearsRangeTemporal == '' else  f'- пользователь ограничил временнОй диапазон границами {yearsRangeTemporal}')

                folderWithTemporal = name_in_folderCurrent
                print(
'''--- Если хотите продолжить дополнять эти промежуточные результаты, нажмите Enter
--- Если эти промежуточные результаты уже не актуальны и хотите их удалить, введите 'R' и нажмите Enter
--- Если хотите найти другие промежуточные результаты, нажмите пробел и затем Enter'''
                     )

                decision = input()
                if not decision: # Temporal-значения обретают статус постоянных
                    targetCount = targetCountTemporal
                    method = methodTemporal
                    year = yearTemporal
                    contentType = contentTypeTemporal
                    channelIdForSearch = channelIdForSearchTemporal
                    q = qTemporal
                    yearsRange = yearsRangeTemporal
                    stageTarget = stageTargetTemporal

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

                    if yearsRange:
                        yearsRange = yearsRange.split('-')
                        yearMaxByUser, yearMinByUser, yearsRange = calendarWithinYear.yearsRangeParser(yearsRange)
# Метаданные этих данных (q, contentType, yearsRange, stageTarget) будут использоваться при исполнении скрипта

                elif decision.lower() == 'r':
                    shutil.rmtree(folderWithTemporal, ignore_errors=True)
                    print('')

# 2.0.3 Если такие данные, сохранённые при прошлом запуске скрипта, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
    if not temporalName: # если itemsTemporal, в т.ч. пустой, не существует
            # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
        name_in_folderCurrent = 'No folder'
        print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта')
        print(
'''
Возможно, Вы располагаете файлом, в котором есть, как минимум, столбец id, и который хотели бы дополнить? Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом, укажите полный путь, включая название файла, и нажмите Enter'''
              )

        while True:
            folderFile = input()
            if not folderFile:
                folderFile = None # для унификации
                break
            else:
                itemS, error, fileName, folder, slash = files2df.excel2df(folderFile)
                if error:
                    if 'No such file or directory' in error:
                        print('Файл:', folder + slash + fileName, '-- не существует; попробуйте, пожалуйста, ещё раз..')
                else: break
            # display(itemS)
# Теперь определены объекты: folder и folderFile (оба None или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский), slash

# 2.0.4 Пользовательские настройки запроса к API YouTube
    # Контент: канал или видео? Или вообще плейлист?
        if not contentType: # если пользователь не подал этот аргумент в рамках experiencedMode
            while True:
                print('--- Если НЕ требуется определить тип искомого контента, нажмите Enter'
                      , ' \n--- Если требуется определить, введите символ: c -- channel, p -- playlist, v -- video -- и нажмите Enter')
                contentType = input()
                if not contentType:
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

        if not channelIdForSearch and contentType == 'video': # если пользователь не подал аргумент channelIdForSearch в рамках experiencedMode
            print(
'''--- Вы выбрали тип контента video
--- Если НЕ предполагается поиск видео в пределах конкретного канала или плейлиста, нажмите Enter
--- Если предполагается такой поиск, введите id канала или плейлиста, после чего нажмите Enter. Этот id можете найти либо внутри ссылки интересующего канала или плейлиста, либо -- если прежде выгружали контент из YouTube -- в столбце с соответствующим id'''
            )

            while True:
                channelIdForSearch = input()
                if not channelIdForSearch:
                    channelIdForSearch = None # для унификации
                    break
# --- Если развивать опцию подачи списка каналов
            # elif '.xlsx' in channelIdsForSearch:
            #     channelIdsForSearch, error, fileName, folder, slash = excel2df(channelIdsForSearch)
            #     if error:
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
                    channelIdForSearch = textPreprocessor.multispaceCleaner(channelIdForSearch)
                    while channelIdForSearch[-1] == ',': channelIdForSearch = channelIdForSearch[:-1] # избавиться от запятых в конце текста
                    # channelIdForSearch = channelIdForSearch.split(', ')
                    # print('Количество id каналов:', len(channelIdForSearch), '\n')
                    # if len(channelIdForSearch) > 1:
                        # channelIdForSearch = channelIdForSearch[0]
                        # print('В качестве аргумента метода search будет использован ТОЛЬКО первый из id\n')
                    # print('channelIdForSearch', channelIdForSearch) # для отладки
                    print('')
                    break

        if not q: # если пользователь не подал этот аргумент в рамках experiencedMode
            print(
'''Скрипт умеет искать контент по ТЕКСТОВОМУ ЗАПРОСУ-ФИЛЬТРУ. При этом если требуется контент конкретного канала, то лучше использовать запрос-фильтр НЕ на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными
--- Если НЕ требуется поиск контента по текстовому запросу-фильтру, нажмите Enter
--- Если требуется такой поиск, введите текст запроса-фильтра, который ожидаете в атрибутах и характеристиках
(описание, название, теги, категории и т.п.) релевантного YouTube-контента, после чего нажмите Enter'''
            )

            if folderFile: print(
'ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла', folderFile,
'будут дополнены актуальными данными из выдачи скрипта (возможно появление новых записей и новых столбцов, а также актуализация содержимого столбцов),',
'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла'
            )

            q = input()
            if not q: q = None # для единообразия
            else: print('')

    # Ограничения временнОго диапазона
        if not publishedAfter and not publishedBefore and not yearsRange: # если пользователь не подал эти аргументы в рамках experiencedMode
            print(
'''Алгоритм API Youtube для ограничения временнОго диапазона выдаваемого контента работает со странностями. Поэтому если требуется конкретный временнОй диапазон, то лучше использовать его НЕ на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными
--- Если НЕ требуется задать временнОй диапазон на этапе выгрузки данных, нажмите Enter
--- Если всё же требуется задать временнОй диапазон, настоятельная рекомендация задать его годами, а не более мелкими единицами времени. Для задания диапазона введите без кавычек минимальный год диапазона, тире, максимальный год диапазона (минимум и максимум могут совпадать в такой записи: 'год-тот же год') и нажмите Enter'''
            )

            while True:
                yearsRange = input()
                if yearsRange:
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
        if publishedAfter:
            yearMinByUser = int(datetime.strptime(publishedAfter,'%Y-%m-%dT%H:%M:%SZ').strftime('%Y')) # из experiencedMode
            # print('elif start_time:', yearMinByUser) # для отладки

        if publishedBefore:
            yearMaxByUser = int(datetime.strptime(publishedBefore,'%Y-%m-%dT%H:%M:%SZ').strftime('%Y')) # из experiencedMode
            # print('elif end_time:', yearMaxByUser) # для отладки
            year = yearMaxByUser

        if yearMinByUser and not yearMaxByUser: yearMaxByUser = int(momentCurrent.strftime('%Y')) # в случае отсутствия пользовательской верхней временнОй границы при наличии нижней
        elif not yearMinByUser and yearMaxByUser: yearMaxByUser = 1970 # в случае отсутствия пользовательской нижней временнОй границы при наличии верхней

        # print('yearMinByUser', yearMinByUser) # для отладки
        # print('yearMaxByUser', yearMaxByUser) # для отладки

        if not publishedAfter and yearMinByUser: publishedAfter = datetime(yearMinByUser, 1, 1).isoformat() + 'Z'
        if not publishedBefore and yearMaxByUser: publishedBefore = datetime(yearMaxByUser, 12, 31).isoformat() + 'Z'

        if yearsRange: print('') # чтобы был отступ, если пользователь подал этот аргумент

# Сложная часть имени будущих директорий и файлов
    complicatedNamePart = '_YT'
    complicatedNamePart += '' if not contentType else '_' + contentType
    complicatedNamePart += '' if not channelIdForSearch else '_channelId_' + channelIdForSearch
    if q: complicatedNamePart += '_' + q if len(q) < 50 else '_' + q[:50]
    complicatedNamePart += '' if not yearMinByUser and not yearMaxByUser else '_' + str(yearMinByUser) + '-' + str(yearMaxByUser)
    # print('complicatedNamePart', complicatedNamePart)

# 2.1 Первичный сбор контента методом search
# 2.1.0 Первый заход БЕЗ аргумента order (этап stage = 0)
    method = 'search'
    stage = 0
    try: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        if not channelIdForSearch or yearsRange or (channelIdForSearch and q): # в противном случае используется не search , а channels + playlists
                # Если поданы вместе channelIdForSearch и q , то search, но если внутри channelIdForSearch подан id плейлиста, то search выдаст ошибку 'Request contains an invalid argument' и следует перейти к playListProcessor в else, причём q будет проигнорирован
                # yearsRange предполагает search , поскольку прочие методы не предполагают сегментирование по временнЫм периодам 
            iteration = 0 # номер итерации применения текущего метода

            print(
f'''В скрипте используются следующие аргументы метода {method} API YouTube: channelId, maxResults, order, pageToken, part, publishedAfter, publishedBefore, q, type .
Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, доступные по ссылке https://developers.google.com/youtube/v3/docs/search , -- можете сделать это внутри метода {method} в разделе 2 исполняемого сейчас скрипта'''
            )

            if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
            print('') # для отступа

            if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
                print('Заход на первую страницу выдачи')
                # print('publishedAfter', publishedAfter) # для отладки
                itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                               channelIdForSearch,
                                                                                               channelType,
                                                                                               contentType,
                                                                                               eventType,
                                                                                               iteration,
                                                                                               keyOrder,
                                                                                               location,
                                                                                               locationRadius,
                                                                                               order=None,
                                                                                               pageToken=None,
                                                                                               publishedAfter=publishedAfter,
                                                                                               publishedBefore=publishedBefore,
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
                                                                                               year=None)

                if not goS:
                    itemS = dfsProcessor(channelIdForSearch,
                                         complicatedNamePart,
                                         contentType,
                                         fileFormatChoice,
                                         df_additional=itemS_additional,
                                         df_final=itemS,
                                         df_in=itemS,
                                         folder=folder,
                                         goS=goS,
                                         method=method,
                                         q=q,
                                         name_in_folderCurrent=name_in_folderCurrent,
                                         stage=stage,
                                         targetCount=targetCount,
                                         momentCurrent=momentCurrent,
                                         year=year,
                                         yearsRange=yearsRange)

                    # display('itemS', itemS) # для отладки
                    if returnDfs: return itemS, playlistS, videoS, commentReplieS, channelS
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'''
                        )

                        sys.exit()

                targetCount = response['pageInfo']['totalResults']
                if not goToPlayList:
                    if targetCount == 0:
                        print(
'''Искомых объектов на серверах YouTube по Вашему запросу, увы, ноль, поэтому нет смысла в продолжении исполнения скрипта. Что делать? Поменяйте настройки запроса и запустите скрипт с начала'''
                        )

                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'''
                            )

                            sys.exit()

                    itemS = dfsProcessor(channelIdForSearch,
                                         complicatedNamePart,
                                         contentType,
                                         fileFormatChoice,
                                         df_additional=itemS_additional,
                                         df_final=itemS,
                                         df_in=itemS,
                                         folder=folder,
                                         goS=goS,
                                         method=method,
                                         q=q,
                                         name_in_folderCurrent=name_in_folderCurrent,
                                         stage=stage,
                                         targetCount=targetCount,
                                         momentCurrent=momentCurrent,
                                         year=year,
                                         yearsRange=yearsRange)

                    # display('itemS', itemS) # для отладки

                    print('  Проход по всем следующим страницам с выдачей          ')
                    while 'nextPageToken' in response.keys() and goS:
                        pageToken = response['nextPageToken']
                        itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                       channelIdForSearch,
                                                                                                       channelType,
                                                                                                       contentType,
                                                                                                       eventType,
                                                                                                       iteration,
                                                                                                       keyOrder,
                                                                                                       location,
                                                                                                       locationRadius,
                                                                                                       order=None,
                                                                                                       pageToken=pageToken,
                                                                                                       publishedAfter=publishedAfter,
                                                                                                       publishedBefore=publishedBefore,
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
                                                                                                       year=None)

                        itemS = dfsProcessor(channelIdForSearch,
                                             complicatedNamePart,
                                             contentType,
                                             fileFormatChoice,
                                             df_additional=itemS_additional,
                                             df_final=itemS,
                                             df_in=itemS,
                                             folder=folder,
                                             goS=goS,
                                             method=method,
                                             q=q,
                                             name_in_folderCurrent=name_in_folderCurrent,
                                             stage=stage,
                                             targetCount=targetCount,
                                             momentCurrent=momentCurrent,
                                             year=year,
                                             yearsRange=yearsRange)

                    print('  Искомых объектов', targetCount, ', а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))
            elif stage < stageTarget: print(f"Этап {stage} пропускаю согласно настройкам из файла stage.txt в директории '{name_in_folderCurrent}'\n")

# 2.1.1 Цикл для прохода по значениям аргумента order, внутри которых проход по всем страницам выдачи (этап stage = 1)
            stage = 1
            orderS = ['date', 'rating', 'title', 'videoCount', 'viewCount']
            if stage >= stageTarget and not goToPlayList: # eсли НЕТ временного файла stage.txt с указанием пропустить этап и нет указания перейти к методам playlistitems и playlists
                if len(itemS) < targetCount:
                # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ включения каких-либо значений аргумента order (в т.ч. вообще БЕЗ них)
                    print('Проход по значениям аргумента order, внутри которых проход по всем страницам выдачи')
                    for order in orderS:
                        itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                       channelIdForSearch,
                                                                                                       channelType,
                                                                                                       contentType,
                                                                                                       eventType,
                                                                                                       iteration,
                                                                                                       keyOrder,
                                                                                                       location,
                                                                                                       locationRadius,
                                                                                                       order,
                                                                                                       pageToken=None,
                                                                                                       publishedAfter=publishedAfter,
                                                                                                       publishedBefore=publishedBefore,
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
                                                                                                       year=None)

                        if not goS: break # на случай сигнала прерывания
                        itemS = dfsProcessor(channelIdForSearch,
                                             complicatedNamePart,
                                             contentType,
                                             fileFormatChoice,
                                             df_additional=itemS_additional,
                                             df_final=itemS,
                                             df_in=itemS,
                                             folder=folder,
                                             goS=goS,
                                             method=method,
                                             q=q,
                                             name_in_folderCurrent=name_in_folderCurrent,
                                             stage=stage,
                                             targetCount=targetCount,
                                             momentCurrent=momentCurrent,
                                             year=year,
                                             yearsRange=yearsRange)

                        print('  Проход по всем следующим страницам с выдачей с тем же значением аргумента order:', order, '          ')
                        while 'nextPageToken' in response.keys() and len(itemS) < targetCount and len(response['items']) > 0 and goS:
                        # -- второе условие -- для остановки алгоритма, если все искомые объекты найдены
                            # БЕЗ какой-то из следующих страниц (в т.ч. вообще БЕЗ них)
                            # третье условие -- для остановки алгоритма, если предыдущая страница выдачи содержит 0 объектов

                            pageToken = response['nextPageToken']
                            # print('pageToken', pageToken)
                            itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                           channelIdForSearch,
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
                                                                                                           year=None)

                            itemS = dfsProcessor(channelIdForSearch,
                                                 complicatedNamePart,
                                                 contentType,
                                                 fileFormatChoice,
                                                 df_additional=itemS_additional,
                                                 df_final=itemS,
                                                 df_in=itemS,
                                                 folder=folder,
                                                 goS=goS,
                                                 method=method,
                                                 q=q,
                                                 name_in_folderCurrent=name_in_folderCurrent,
                                                 stage=stage,
                                                 targetCount=targetCount,
                                                 momentCurrent=momentCurrent,
                                                 year=year,
                                                 yearsRange=yearsRange)

                    print('  Искомых объектов', targetCount, ', а найденных С включением аргумента order:', len(itemS))
                else: print('Все искомые объекты найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
            elif stage < stageTarget: print(f"Этап {stage} пропускаю согласно настройкам из файла stage.txt в директории '{name_in_folderCurrent}'\n")

# 2.1.2 Этап сегментирования по годам (stage = 2)
            stage = 2
            if stage >= stageTarget and not goToPlayList: # eсли НЕТ временного файла stage.txt с указанием пропустить этап и нет указания перейти к методам playlistitems и playlists
                if len(itemS) < targetCount:
                # для остановки алгоритма, если все искомые объекты найдены БЕЗ включения каких-либо значений аргумента order (в т.ч. вообще БЕЗ них)
                    print(
f'''Увы, число найденных объектов: {len(itemS)} -- менее числа искомых: {targetCount}
--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам, просто нажмите Enter. Поиск может занять минуты и даже часы, но Вы СМОЖЕТЕ ПРЕРВАТЬ ИСПОЛНЕНИЕ СКРИПТА КНОПКОЙ и ВЫГРУЖЕННЫЕ ДАННЫЕ СОХРАНЯТСЯ
--- Если НЕ хотите, нажмите пробел и затем Enter'''
                    )

                    if len(input()) == 0:
                        print('Внутри каждого года прохожу по значениям аргумента order, внутри которых прохожу по всем страницам выдачи')
                        goC = True
# ********** из фрагмента 2.1.0 + условие для goC
                        while len(itemS) < targetCount and goC and goS:
                            print(f'  Для года {year} заход на первую страницу выдачи БЕЗ аргумента order')
                            itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                           channelIdForSearch,
                                                                                                           channelType,
                                                                                                           contentType,
                                                                                                           eventType,
                                                                                                           iteration,
                                                                                                           keyOrder,
                                                                                                           location,
                                                                                                           locationRadius,
                                                                                                           order=None,
                                                                                                           pageToken=None,
                                                                                                           publishedAfter=f'{year}-01-01T00:00:00Z',
                                                                                                           publishedBefore=f'{year + 1}-01-01T00:00:00Z',
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
                                                                                                           year=year)

                            if len(itemS_additional) == 0:
                                print(f'\n--- Первая страница выдачи БЕЗ аргумента order для года {year} -- пуста',
                                      '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать предыдущий год, просто нажмите Enter',
                                      '\n--- Если хотите, нажмите пробел и затем Enter')

                                if len(input()) == 0:
                                    goC = False
                                    break

                            itemS = dfsProcessor(channelIdForSearch,
                                                 complicatedNamePart,
                                                 contentType,
                                                 fileFormatChoice,
                                                 df_additional=itemS_additional,
                                                 df_final=itemS,
                                                 df_in=itemS,
                                                 folder=folder,
                                                 goS=goS,
                                                 method=method,
                                                 q=q,
                                                 name_in_folderCurrent=name_in_folderCurrent,
                                                 stage=stage,
                                                 targetCount=targetCount,
                                                 momentCurrent=momentCurrent,
                                                 year=year,
                                                 yearsRange=yearsRange)

                            print(f'    Проход по всем следующим страницам с выдачей для года {year} БЕЗ аргумента order')
                            while ('nextPageToken' in response.keys()) & goS:
                                pageToken = response['nextPageToken']
                                itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                               channelIdForSearch,
                                                                                                               channelType,
                                                                                                               contentType,
                                                                                                               eventType,
                                                                                                               iteration,
                                                                                                               keyOrder,
                                                                                                               location,
                                                                                                               locationRadius,
                                                                                                               order=None,
                                                                                                               pageToken=pageToken,
                                                                                                               publishedAfter=f'{year}-01-01T00:00:00Z',
                                                                                                               publishedBefore=f'{year + 1}-01-01T00:00:00Z',
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
                                                                                                               year=year)

                            if len(itemS_additional) == 0:
                                itemS = dfsProcessor(channelIdForSearch,
                                                     complicatedNamePart,
                                                     contentType,
                                                     fileFormatChoice,
                                                     df_additional=itemS_additional,
                                                     df_final=itemS,
                                                     df_in=itemS,
                                                     folder=folder,
                                                     goS=goS,
                                                     method=method,
                                                     q=q,
                                                     name_in_folderCurrent=name_in_folderCurrent,
                                                     stage=stage,
                                                     targetCount=targetCount,
                                                     momentCurrent=momentCurrent,
                                                     year=year,
                                                     yearsRange=yearsRange)

                            print(f'    Искомых объектов в году {year}: {targetCount}, а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))
# ********** из фрагмента 2.1.1
                            if len(itemS) < targetCount:
                                print(f'  Для года {year} проход по значениям аргумента order,'
                                      , 'внутри которых проход по всем страницам выдачи')
                                for order in orderS:
                                    itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                                   channelIdForSearch,
                                                                                                                   channelType,
                                                                                                                   contentType,
                                                                                                                   eventType,
                                                                                                                   iteration,
                                                                                                                   keyOrder,
                                                                                                                   location,
                                                                                                                   locationRadius,
                                                                                                                   order,
                                                                                                                   pageToken=None,
                                                                                                                   publishedAfter=f'{year}-01-01T00:00:00Z',
                                                                                                                   publishedBefore=f'{year + 1}-01-01T00:00:00Z',
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
                                                                                                                   year=year)

                                    if not goS: break # на случай сигнала прерывания
                                    itemS = dfsProcessor(channelIdForSearch,
                                                         complicatedNamePart,
                                                         contentType,
                                                         fileFormatChoice,
                                                         df_additional=itemS_additional,
                                                         df_final=itemS,
                                                         df_in=itemS,
                                                         folder=folder,
                                                         goS=goS,
                                                         method=method,
                                                         q=q,
                                                         name_in_folderCurrent=name_in_folderCurrent,
                                                         stage=stage,
                                                         targetCount=targetCount,
                                                         momentCurrent=momentCurrent,
                                                         year=year,
                                                         yearsRange=yearsRange)

                                    print(
f'    Для года {year} проход по всем следующим страницам с выдачей с тем же значением аргумента order:', order
                                    )

                                    while 'nextPageToken' in response.keys() and len(itemS) < targetCount and len(response['items']) > 0 and goS:
                                        pageToken = response['nextPageToken']
                                        itemS_additional, goS, goToPlayList, iteration, keyOrder, response = bigSearch(api_keyS,
                                                                                                                       channelIdForSearch,
                                                                                                                       channelType,
                                                                                                                       contentType,
                                                                                                                       eventType,
                                                                                                                       iteration,
                                                                                                                       keyOrder,
                                                                                                                       location,
                                                                                                                       locationRadius,
                                                                                                                       order,
                                                                                                                       pageToken,
                                                                                                                       publishedAfter=f'{year}-01-01T00:00:00Z',
                                                                                                                       publishedBefore=f'{year + 1}-01-01T00:00:00Z',
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
                                                                                                                       year=year)

                                        itemS = dfsProcessor(channelIdForSearch,
                                                             complicatedNamePart,
                                                             contentType,
                                                             fileFormatChoice,
                                                             df_additional=itemS_additional,
                                                             df_final=itemS,
                                                             df_in=itemS,
                                                             folder=folder,
                                                             goS=goS,
                                                             method=method,
                                                             q=q,
                                                             name_in_folderCurrent=name_in_folderCurrent,
                                                             stage=stage,
                                                             targetCount=targetCount,
                                                             momentCurrent=momentCurrent,
                                                             year=year,
                                                             yearsRange=yearsRange)

                                print(
f'''    Искомых объектов {targetCount}, а найденных с добавлением сегментирования по году (год {year}) и включением аргумента order: {len(itemS)}
'''
                                )

                            else: print('  Все искомые объекты в году', year, 'найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
                            year -= 1
                            if yearMinByUser: # если пользователь ограничил временнОй диапазон
                                if year <= yearMinByUser:
                                    goC = False
                                    print(f'Завершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser} (с точностью до года)\n')

# 2.1.3 Экспорт выгрузки метода search и опциональное завершение скрипта
                df2file.df2fileShell(complicatedNamePart,
                                     currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                     df_in=itemS,
                                     fileFormatChoice=fileFormatChoice,
                                     folder=folder,
                                     method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method) # чтобы избавиться от лишней точки в имени файла

            elif stage < stageTarget:
                print(
f'''
Этап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}_Temporal"'''
                )

            if not goToPlayList: # eсли нет указания перейти к методам playlistitems и playlists
                print(
'''
Выгрузка метода search содержит НЕ ВСЕ доступные для выгрузки из API YouTube характеристки контента
--- Если хотите выгрузить дополнительные характеристики (ссылки для ознакомления с ними появятся ниже), нажмите Enter
--- Если НЕ хотите их выгрузить, нажмите пробел и затем Enter. Тогда исполнение скрипта завершится'''
                )

                if len(input()) > 0:
                    print('Скрипт исполнен')
                    if folderWithTemporal:
                        if os.path.exists(folderWithTemporal):
                            # print('folderWithTemporal:', folderWithTemporal) # для отладки
                            print(
            '''Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы, можно удалить её во избежание путаницы при следующих запусках скрипта
            --- Если хотите их удалить, введите 'R' и нажмите Enter'''
                            )

                            decision = input()
                            if decision == 'R': shutil.rmtree(folderWithTemporal, ignore_errors=True)

                    if returnDfs: return itemS, playlistS, videoS, commentReplieS, channelS
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        print(
'''Сейчас появится надпись: 'An exception has occurred, use %tb to see the full traceback.\nSystemExit' -- так и должно быть.
Модуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473'''
                        )

                        sys.exit()

# 2.2 Выгрузка дополнительных характеристик и контента методами playlists и playlistItems, videos, commentThreads и comments, channels
# 2.2.0 Этап stage = 3
            stage = 3

# 2.2.1 Выгрузка характеристик плейлистов как дополнительных к search, а ниже тот самый 'в противном случае', когда используется не search, а channels + playlists
            snippetContentType = 'playlist'
            if len(itemS) > 0: # если использовался search..
                if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # .. и в его выдаче есть плейлисты
                    playlistIdS = itemS[itemS['id.kind'] == f'youtube#{snippetContentType}']

                    playlistIdS =\
                        playlistIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in playlistIdS.columns else playlistIdS['id'].to_list()

                    playlistS, playlistVideoChannelS = playListProcessor(api_keyS,
                                                                         channelIdForSearch,
                                                                         complicatedNamePart,
                                                                         contentType,
                                                                         df_final=itemS,
                                                                         experiencedMode=experiencedMode,
                                                                         fileFormatChoice=fileFormatChoice,
                                                                         folder=folder,
                                                                         goS=goS,
                                                                         keyOrder=keyOrder,
                                                                         momentCurrent=momentCurrent,
                                                                         playlistIdS=playlistIdS,
                                                                         q=q,
                                                                         name_in_folderCurrent=name_in_folderCurrent,
                                                                         stage=stage,
                                                                         targetCount=targetCount,
                                                                         year=year,
                                                                         yearsRange=yearsRange)

            elif goToPlayList:
                playlistIdS = [channelIdForSearch] # поскольку подан id не канала, а плейлиста, этот id передатся из одного контейнера в другой
                channelIdForSearch = None
                playlistS, playlistVideoChannelS = playListProcessor(api_keyS,
                                                                     channelIdForSearch,
                                                                     complicatedNamePart=complicatedNamePart,
                                                                     contentType=contentType,
                                                                     df_final=channelS, # т.к. в отсутствие itemS channelS становится базовым датафреймом
                                                                     experiencedMode=experiencedMode,
                                                                     fileFormatChoice=fileFormatChoice,
                                                                     folder=folder,
                                                                     goS=goS,
                                                                     keyOrder=keyOrder,
                                                                     momentCurrent=momentCurrent,
                                                                     playlistIdS=playlistIdS,
                                                                     q=q,
                                                                     name_in_folderCurrent=name_in_folderCurrent,
                                                                     stage=stage,
                                                                     targetCount=targetCount,
                                                                     year=year,
                                                                     yearsRange=yearsRange)

                # print('playlistIdS:', playlistIdS) # для отладки   

        else: # в противном случае используется не search, а channels + playlists
                # то есть пользователь подал id канала или плейлиста и, следоватлеьно, НЕ использовался search
            channelS = channelProcessor(api_keyS,
                                        channelIdForSearch,
                                        complicatedNamePart,
                                        contentType,
                                        df_final=itemS,
                                        experiencedMode=experiencedMode,
                                        fileFormatChoice=fileFormatChoice,
                                        folder=folder,
                                        goS=goS,
                                        keyOrder=keyOrder,
                                        momentCurrent=momentCurrent,
                                        playlistS=playlistS,
                                        q=q,
                                        name_in_folderCurrent=name_in_folderCurrent,
                                        snippetContentType='channel',
                                        stage=stage,
                                        targetCount=targetCount,
                                        year=year,
                                        yearsRange=yearsRange,
                                        videoS=videoS)

            # display('channelS:', channelS) # для отладки

            if len(channelS) > 0: playlistIdS = channelS['contentDetails.relatedPlaylists.uploads'].to_list()
            else: # поскольку подан id не канала, а плейлиста, этот id передатся из одного контейнера в другой
                playlistIdS = [channelIdForSearch]
                channelIdForSearch = None

            playlistS, playlistVideoChannelS = playListProcessor(api_keyS,
                                                                 channelIdForSearch,
                                                                 complicatedNamePart,
                                                                 contentType,
                                                                 df_final=channelS, # т.к. в отсутствие itemS channelS становится базовым датафреймом !!!
                                                                 experiencedMode=experiencedMode,
                                                                 fileFormatChoice=fileFormatChoice,
                                                                 folder=folder,
                                                                 goS=goS,
                                                                 keyOrder=keyOrder,
                                                                 momentCurrent=momentCurrent,
                                                                 playlistIdS=playlistIdS,
                                                                 q=q,
                                                                 name_in_folderCurrent=name_in_folderCurrent,
                                                                 stage=stage,
                                                                 targetCount=targetCount,
                                                                 year=year,
                                                                 yearsRange=yearsRange)

        # print('playlistIdS:', playlistIdS) # для отладки

# 2.2.2 Выгрузка дополнительных характеристик видео
        method = 'videos'
        snippetContentType = 'video'
        videoIdS = []
        if len(itemS) > 0: # если использовался search..
            if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # .. и в его выдаче есть видео

                print(
f'''В скрипте используются следующие аргументы метода {method} API YouTube: part=['snippet', 'contentDetails', 'localizations', 'statistics', 'status', 'topicDetails'], id, maxResults .
Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/videos'''
                )

                if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
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
                for playlistVideoIdSnippet in playlistVideoId_list: playlistVideoIdS.extend(playlistVideoIdSnippet)
                videoIdS.extend(playlistVideoIdS)
                videoIdS = list(dict.fromkeys(videoIdS))

        # print(videoIdS) # для отладки

        if videoIdS:
            print(f'''Проход по видео{' порциями по 50 штук' if len(videoIdS) > 50 else ''} для выгрузки их характеристик''')
            if len(itemS) > 0: print('-- дополнительных к выруженным методом search')

            videoS = portionsProcessor(api_keyS,
                                       channelIdForSearch,
                                       complicatedNamePart,
                                       contentType,
                                       df_final=itemS,
                                       fileFormatChoice=fileFormatChoice,
                                       folder=folder,
                                       idS=videoIdS,
                                       keyOrder=keyOrder,
                                       method=method,
                                       momentCurrent=momentCurrent,
                                       q=q,
                                       name_in_folderCurrent=name_in_folderCurrent,
                                       stage=stage,
                                       targetCount=targetCount,
                                       year=year,
                                       yearsRange=yearsRange)

# ********** categoryId
            # Взять столбец snippet.categoryId, удалить из него дубликаты кодов категорий и помеcтить уникальные коды в список
            # display(videoS) # для отладки
            # print('videoS.columns:', videoS.columns) # для отладки
            uniqueCategorieS = videoS['snippet.categoryId'].drop_duplicates().to_list()
            # print('\nУникальные коды категорий в базе:', uniqueCategorieS, '\nЧисло уникальных категорий в базе:', len(uniqueCategorieS))
            try:
                youtube = api.build('youtube', 'v3', developerKey=api_keyS[keyOrder])
                response = youtube.videoCategories().list(part='snippet', id=uniqueCategorieS).execute()

            except: goC, goS, goToPlayList, keyOrder, problemItemId = errorProcessor(errorDescription=sys.exc_info(), keyOrder=keyOrder, sourceId=None)

            categoryNameS = pandas.json_normalize(response['items'])
                # оформить как датафрейм id категорий из списка uniqueCategorieS и их расшифровки

            categoryNameS.index = categoryNameS['id'].to_list() # заменить индексы датафрейма с расшифровками значениями столбца id

            for row in categoryNameS.index: # добавить расшифровки категорий в новый столбец categoryName датафрейма с видео
                videoS.loc[videoS['snippet.categoryId'] == row, 'categoryName'] = categoryNameS['snippet.title'][row]

            df2file.df2fileShell(complicatedNamePart,
                                 currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                 df_in=videoS,
                                 fileFormatChoice=fileFormatChoice,
                                 folder=folder,
                                 method=method.split('.')[0] + method.split('.')[1].capitalize() if '.' in method else method) # чтобы избавиться от лишней точки в имени файла

            commentS = pandas.DataFrame() # не в следующем ченке, чтобы иметь возможность перезапускать его, не затирая промежуточный результат выгрузки

# 2.2.3 Выгрузка комментариев к видео
            print(
'\n--- Если хотите выгрузить (в отдельный файл) комментарии к видео,',
f'содержащимся в файле "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart} {method}.xlsx" директории "{momentCurrent.strftime('%Y%m%d')}{complicatedNamePart}",',
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
f'''В скрипте используются следующие аргументы метода {method} API YouTube: part=['snippet', 'id', 'replies'], maxResults, videoId .
Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/commentThreads'''
                )

                if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
                print('') # для отступа

                # Переназначить объект videoIdS для целей текущего чанка
                videoIdS = videoS[videoS['statistics.commentCount'].notna()]
                videoIdS = videoIdS[videoIdS['statistics.commentCount'].astype(int) > 0]

                snippetContentType = 'video'
                videoIdS = videoIdS[f'id.{snippetContentType}Id'].to_list() if f'id.{snippetContentType}Id' in videoIdS.columns else videoIdS['id'].to_list()
                print('Число видео с комментариями:', len(videoIdS))
                print('\nВыгрузка родительских (topLevel) комментариев')
                commentS = pandas.DataFrame()
                for videoId in tqdm(videoIdS):
                # for videoId in videoS['id'][4576:]: # для отладки
                    # print('videoId:', videoId) # для отладки
                    commentS_additional, goS, keyOrder, problemVideoId = downloadComments(api_keyS, keyOrder, method, sourceId=videoId)
                    if not goS: break # на случай сигнала прерывания
                    if problemVideoId: problemVideoIdS.append(problemVideoId)
                    commentS = dfsProcessor(channelIdForSearch,
                                            complicatedNamePart,
                                            contentType,
                                            fileFormatChoice,
                                            df_additional=commentS_additional,
                                            df_final=itemS,
                                            df_in=commentS,
                                            folder=folder,
                                            goS=goS,
                                            method=method,
                                            q=q,
                                            name_in_folderCurrent=name_in_folderCurrent,
                                            stage=stage,
                                            targetCount=targetCount,
                                            momentCurrent=momentCurrent,
                                            year=year,
                                            yearsRange=yearsRange)

                commentS = commentS.drop(['kind', 'etag', 'id', 'snippet.channelId', 'snippet.videoId'], axis=1) # т.к. дублируются содержательно
                commentS = prefixDropper(commentS)
                df2file.df2fileShell(complicatedNamePart,
                                     currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                     df_in=commentS,
                                     fileFormatChoice=fileFormatChoice,
                                     folder=folder,
                                     method='commentS')

# ********** replieS
                print('') # для отступа
                commentS_replied_index = commentS[(commentS['replies.comments'].notna()) & commentS['snippet.totalReplyCount'] > 0].index
                if len(commentS_replied_index) > 0: # есть ли хотя бы один отвеченный родительский (topLevel) комментарий?
                    print('Проход по строкам всех родительских (topLevel) комментариев, имеющих ответы')
                    replieS = pandas.DataFrame()
                    for row in tqdm(commentS_replied_index):
                        replieS_additional = pandas.DataFrame()
                        try: replieS_additional = pandas.json_normalize(commentS['replies.comments'][row])
                        except Exception as excptn:
                            print('Exception 2 в searchByText') # для отладки
                            print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
                            print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой
                            print("commentS['replies.comments'][row]:", commentS['replies.comments'][row]) # для отладки !!!

                        commentS.loc[row, 'Недостача_ответов'] = commentS['snippet.totalReplyCount'][row] - len(replieS_additional)
                            # записать разницу между ожданиями и реальностью в новый столбец `Недостача_ответов`

                        replieS = pandas.concat([replieS, replieS_additional]).reset_index(drop=True)

                    replieS = replieS.reset_index(drop=True)
                    replieS.loc[:, 'snippet.totalReplyCount'] = 0
                    replieS.loc[:, 'Недостача_ответов'] = 0
                    replieS = prefixDropper(replieS)
                    df2file.df2fileShell(complicatedNamePart,
                                         currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                         df_in=replieS,
                                         fileFormatChoice=fileFormatChoice,
                                         folder=folder,
                                         method='replieS')

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
                    commentReplieS = dfsProcessor(channelIdForSearch,
                                                  complicatedNamePart,
                                                  contentType,
                                                  fileFormatChoice,
                                                  df_additional=replieS,
                                                  df_final=itemS,
                                                  df_in=commentReplieS,
                                                  folder=folder,
                                                  goS=goS,
                                                  method=method,
                                                  q=q,
                                                  name_in_folderCurrent=name_in_folderCurrent,
                                                  stage=stage,
                                                  targetCount=targetCount,
                                                  momentCurrent=momentCurrent,
                                                  year=year,
                                                  yearsRange=yearsRange)

                    method = 'comments'
                    part = 'id, snippet'
                    textFormat = 'plainText' # = 'html' по умолчанию
                    problemCommentIdS = []
                    replieS = pandas.DataFrame() # зачем? См. этап 4.2 ниже

                    print(
f'''
В скрипте используются следующие аргументы метода {method} API YouTube: part=['snippet', 'id'], maxResults, parentId, textFormat .
Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.
Если хотите добавить другие аргументы метода {method} API YouTube, можете ознакомиться с ними по ссылке: https://developers.google.com/youtube/v3/docs/commentThreads'''
                    )

                    if not experiencedMode: input('--- После прочтения этой инструкции нажмите Enter')
                    print('') # для отступа
                    commentIdS = commentReplieS['id'][commentReplieS['Недостача_ответов'] > 0]
                    # print('commentIdS:', commentIdS) # для отладки
                    if commentIdS: # блок 'если нет недостачи'
                        print('Проход по id всех родительских (topLevel) комментариев с недостачей ответов для выгрузки этих ответов')
                        for commentId in tqdm(commentIdS):
                            page = 0 # номер страницы выдачи
                            replieS_additional, goS, keyOrder, problemCommentId = downloadComments(api_keyS, keyOrder, method, sourceId)
                            if not goS: break # на случай сигнала прерывания
                            if problemCommentId: problemCommentIdS.append(problemCommentId)
                            replieS = dfsProcessor(channelIdForSearch,
                                                   complicatedNamePart,
                                                   contentType,
                                                   fileFormatChoice,
                                                   df_additional=replieS_additional,
                                                   df_final=itemS,
                                                   df_in=replieS,
                                                   folder=folder,
                                                   goS=goS,
                                                   method=method,
                                                   q=q,
                                                   name_in_folderCurrent=name_in_folderCurrent,
                                                   stage=stage,
                                                   targetCount=targetCount,
                                                   momentCurrent=momentCurrent,
                                                   year=year,
                                                   yearsRange=yearsRange)

                        print(
'Ответов выгружено', len(replieS), '; проблемные родительские (topLevel) комментарии:', problemCommentIdS if problemCommentIdS else 'отсутствуют\n'
                        )

                        # Для совместимости датафреймов добавить столбцы`snippet.totalReplyCount` и `Недостача_ответов`
                        # display('replieS:', replieS) # для отладки
                        # print('replieS.columns:', replieS.columns) # для отладки
                        replieS.loc[:, 'snippet.totalReplyCount'] = 0
                        replieS.loc[:, 'Недостача_ответов'] = 0
                        replieS = replieS.drop('snippet.parentId', axis=1) # удалить столбец `snippet.parentId`, т.к. и из столбца `id` всё ясно
                        commentReplieS = dfsProcessor(channelIdForSearch,
                                                      complicatedNamePart,
                                                      contentType,
                                                      fileFormatChoice,
                                                      df_additional=replieS,
                                                      df_final=itemS,
                                                      df_in=commentReplieS,
                                                      folder=folder,
                                                      goS=goS,
                                                      method=method,
                                                      q=q,
                                                      name_in_folderCurrent=name_in_folderCurrent,
                                                      stage=stage,
                                                      targetCount=targetCount,
                                                      momentCurrent=momentCurrent,
                                                      year=year,
                                                      yearsRange=yearsRange)

                        df2file.df2fileShell(complicatedNamePart,
                                             currentMoment=momentCurrent.strftime('%Y%m%d_%H%M'), # .strftime -- чтобы варьировать для итоговой директории и директории Temporal
                                             df_in=commentReplieS,
                                             fileFormatChoice=fileFormatChoice,
                                             folder=folder,
                                             method='commentReplieS')

                        # блок 'если нет недостачи' завершён
                        # print('''блок 'если нет недостачи' завершён''') # для отладки
                else: print('Нет ни одного откомментированного родительского (topLevel) комментария')

# 2.2.4 Выгрузка дополнительных характеристик каналов
        if len(itemS) > 0: # если использовался search..
            snippetContentType = 'channel'
            if sum(itemS['id.kind'].str.split('#').str[-1] == snippetContentType) > 0: # .. и в его выдаче есть каналы
                channelS = channelProcessor(api_keyS,
                                            channelIdForSearch,
                                            complicatedNamePart,
                                            contentType,
                                            df_final=itemS,
                                            experiencedMode=experiencedMode,
                                            fileFormatChoice=fileFormatChoice,
                                            folder=folder,
                                            goS=goS,
                                            keyOrder=keyOrder,
                                            momentCurrent=momentCurrent,
                                            playlistS=playlistS,
                                            q=q,
                                            name_in_folderCurrent=name_in_folderCurrent,
                                            snippetContentType='channel',
                                            stage=stage,
                                            targetCount=targetCount,
                                            year=year,
                                            yearsRange=yearsRange,
                                            videoS=videoS)

# 2.2.5 Экспорт выгрузки метода search и финальное завершение скрипта
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
        if returnDfs: return itemS, playlistS, videoS, commentReplieS, channelS

    except KeyboardInterrupt: # обработать сигнал прерывания, поданный на любом этапе сбора данных
        # display(itemS)
        if len(itemS) > 0: # в противном случае нет необходимости сохранять что бы то ни было в Temporal
            dfsProcessor(channelIdForSearch,
                         complicatedNamePart,
                         contentType,
                         fileFormatChoice,
                         df_additional=pandas.DataFrame(), # внутри функции dfsProcessor в данном случае отработает только блок goS = False , а df_additional обрабатывается над этим блоком
                         df_final=itemS,
                         df_in=pandas.DataFrame(), # внутри функции dfsProcessor в данном случае отработает только блок goS = False , а df_in обрабатывается над этим блоком
                         folder=folder,
                         goS=False,
                         method=method,
                         q=q,
                         name_in_folderCurrent=name_in_folderCurrent,
                         stage=stage,
                         targetCount=targetCount,
                         momentCurrent=momentCurrent,
                         year=year,
                         yearsRange=yearsRange)

            if returnDfs: return itemS, playlistS, videoS, commentReplieS, channelS

# Поработать со snippetContentType , возможно, вынести этот аргумент из channelProcessor
# Добавить возможность подавать множество ключей в accessToken
# https://stackoverflow.com/questions/30475309/get-youtube-trends-v3-country-wise-in-json -- про тренды
# Если подавать в скрипт располагаемый файл НЕ search , то в нём не будет столбца id.kind , следовательно, скрипт не поймёт, какие дополнительные характеристики выгружать

# problemItemId в errorProcessor инициализируется sourceId и никогда не сбрасывается при повторных итерациях, поэтому в ветке про «channel/comment/playlist/video» в лог попадёт исходный channelIdForSearch, а не реально проблемный videoId/commentId. В downloadComments это особенно заметно: при обходе videoIdS в цикле все проблемные видео получат один и тот же первый sourceId.
# Вынести в самодостаточные функции работу с видео, каналами, комментариями, плейлистами
# chart в https://developers.google.com/youtube/v3/docs/videos/list -- интересно

# Вы добавили:
# python
#     elif ('channel' in str(errorDescription[1]).lower()) | ...:
#         print('  Проблема может быть связана с ограничением доступа...')
#         goC = False
# Это останавливает цикл повторных попыток. Но в downloadComments после выхода из внутреннего while goC сразу идёт:

# python
#         commentS = pandas.concat([commentS, pandas.json_normalize(response['items'])])
# Если ошибка произошла на первом же запросе, переменная response может быть не определена. Это приведёт к UnboundLocalError. Раньше при такой ошибке цикл зацикливался, что тоже плохо. Правильнее было бы при goC == False не пытаться использовать response, а прервать внешний цикл или пропустить текущий sourceId. Но это отдельная доработка.
