#!/usr/bin/env python
# coding: utf-8

# # 0. Настройки и авторизация


# In[1]:


# 0.0 Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from tqdm import tqdm
        import datetime, os, pandas, re, time, warnings
        import googleapiclient.discovery as api
        import googleapiclient.errors
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[1] 
        if module == 'googleapiclient': module = 'google-api-python-client'
        print('Пакет', module, 'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

print('Для исполнения скрипта не обязательны пререквизиты (предшествующие скрпиты и файлы с данными). Но оно требует от пользователя предварительно получить API key для авторизации в API YouTube по ключу (см. примреную видео-инструкцию: https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ). Для получения API key следует создать проект, авторизовать его, подключить к нему API нужного сервиса Google. Проект -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому сервисов Google и применения на этой основе API разных сервисов Google в рамках установленных Гуглом ограничений (the units of quota). Разные уровни авторизации -- это авторизация ключом (представляющим собой код) и полная авторизация (ключ + протокол Google OAuth 2.0, реализующийся в формате файла JSON). Авторизация ключом нужна, чтобы использовать любой метод любого API. Её достаточно, если выполнять действия, которые были бы доступны Вам как пользователю сервисов Google без Вашего входа в аккаунт: посмотреть видео, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления видео, то Вам придётся пройти полную авторизацию. Далее разные API как бы подключаются к проектам (кнопка Enable APIs and servises), используются, затем отключаются (кнопка Disable APIs).'
      , '\nОдного ключа может не хватить (quota is exceeded) для выгрузки всего предоставляемого ЮТьюбом по запросу пользователя контента. К счастью, использованный ключ ежесуточно восстанавливается ЮТьюбом. Скрпит позволяет сохранить промежуточную выгрузку и после восстановления ключа автоматически продолжит её дополнять с момента остановки. В момент остановки пользователь увидит надпись: "Поскольку ключи закончились, исполнение скрипта завершаю. Подождите сутки для восстановления ключей или подготовьте новый ключ -- и запустите скрипт с начала", а исполнение скрипта прервётся. Не пугайтесь, нажмите OK и следуйте этой инструкции.'
      , '\nСкрипт нацелен на выгрузку характеристик контента YouTube семью методами его API: search, videos, commentThreads и comments, channels, playlists и playlistItems. Причём количество объектов выгрузки максимизируется путём её пересортировки и сегментирования по годам.'
      , '\nДля корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.'
      , '\nПреимущества скрипта перед выгрузкой контента из YouTube вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel. Преимущества скрипта перед выгрузкой контента через непосредственно API YouTube: гораздо быстрее, гораздо большее количество контента с одним и тем же ключом, не требуется тщательно изучать обширную документацию семи методов API YouTube (search, videos, commentThreads и comments, channels, playlists и playlistItems), выстроена логика обрашения к этим методам')
input('--- После прочтения этой инструкции нажмите Enter')

goS = True


# In[2]:


# 0.1 Подготовка ключей
if 'credentials.txt' not in os.listdir():
    print('\n--- Введите в окно Ваш API key для авторизации в API YouTube по ключу'
          , '(примерная видео-инструкция для создания API key доступна по ссылке https://www.youtube.com/watch?v=EXysYgWeapI&t=490s ).'
          , 'Для увеличения размера выгрузки желательно создать несколько ключей (пять -- отлично) и ввести их без кавычек через запятую с пробелом'
          , '\n--- После ввода нажмите Enter')
    API_keyS = input()
    print('-- далее буд[еу]т использован[ы] эт[и] ключ[и]')
    
    file = open("credentials.txt", "w+") # открыть на запись
    file.write(API_keyS)
    file.close()
else:
    file = open('credentials.txt')
    API_keyS = file.read()
    print('Нашёл файл credentials.txt с ключ[ом ами]; далее буду использовать ключ[и] из него:', API_keyS)

API_keyS = API_keyS.split(', ')
print('Количество ключей:', len(API_keyS), '\n')
keyOrder = 0


# In[3]:


# 0.2 Запрос сегодняшней даты в формате yyyymmdd
today = datetime.date.today().strftime("%Y%m%d")
print('Текущяя дата:', today, '-- она будет использована для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках\n')
# print('Сегодня год:', today[:4])
# print('Сегодня месяц:', today[4:6])
# print('Сегодня день:', today[6:])
year = int(today[:4])


# In[4]:


# 0.3 Некоторые базовые настройки запроса к API YouTube
q = ''
complicatedNamePart = '' # сложная часть имени будущих директорий и файлов в случае первичного сбора контента
stageTarget = 0 # stageTarget принимает значения [0; 3] и относится к стадиям скрипта
yearMinByUser = None
yearMaxByUser = None


# In[5]:


# 0.4 Пользовательские настройки запроса к API YouTube
print('--- Если НЕ располагаете файлом, в котором есть хотя бы его id, и планируете первичный сбор контента, нажмите Enter'
      , ' \n--- Если располагаете таким файлом, Укажите полный путь, включая название файла, и нажмите Enter')
while True:
    folderFile = input()
    if len(folderFile) == 0:
        break
    else:
        from files2df import excel2df
        itemS, error, fileName, folder, slash = excel2df(folderFile)
        if error != None:
            if 'No such file or directory' in error: print('Файл:', folder + slash + fileName, '-- не существует; попробуйте, пожалуйста, ещё раз..')
        else: break 
    # display(itemS)

if len(folderFile) == 0: # eсли НЕТ файла с id
    # Контент: канал или видео? Или вообще плейлист?
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    while True:
        print('--- Если НЕ требуется выбрать тип контента, нажмите Enter'
              , ' \n--- Если требуется выбрать, введите символ: c -- channel, p -- playlist, v -- video -- и нажмите Enter')
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
    
    if contentType == 'video':
        print('\n--- Вы выбрали тип контента video'
              , '\n--- Если НЕ предполагается поиск видео в пределах конкретного канала или списка каналов, нажмите Enter'
              , '\n--- Если предполагается такой поиск, введите id канала'
              , 'или путь к ранее созданногму Вами файлу, в котором есть столбец'
              , 'snippet.channelId с id каналов, после чего нажмите Enter')  
        channelId = input()
    
    print('--- Если НЕ предполагается поиск контента по текстовому запросу-фильтру, нажмите Enter'
          , '\n--- Если предполагается такой поиск, введите текст, который ожидаете в атрибутах и характеристиках'
          , '(описание, название, теги, категории и т.п.) релевантного YouTube-контента,'
          , 'после чего нажмите Enter'
          , '\nВАЖНО! Если запускаете скрипт после истечения ключа,'
          , 'убедитесь, что вводите тот же текст запроса-фильтра, что и перед истечением ключа')
    q = input()
    
    # Ограничения временнОго диапазона
    goC = True
    while goC:
        print('\nАлгоритм API Youtube для ограничения временнОго диапазона выдаваемого контента работает со странностями.'
              , 'Поэтому если требуется конкретный временнОй диапазон, то лучше использовать его НЕ на текущем этапе выгрузки данных,'
              , 'а на следующем этапе -- предобработки датафрейма с выгруженными данными')
        print('--- Если НЕ требуется задать временнОй диапазон на этапе выгрузки данных, нажмите Enter'
              , ' \n--- Если всё же требуется задать временнОй диапазон, настоятельная рекомендация задать его годами,'
              , 'а не более мелкими единицами времени. Для задания диапазона введите без кавычек минимальный год диапазона, тире,'
              , 'максимальный год диапазона (минимум и максимум могут совпадать) и нажмите Enter')
        yearS = input()
        if len(yearS) != 0:
            yearS = re.sub(r' *', '', yearS)
            if '-' in yearS:
                yearS = yearS.split('-')
                print('--- Вы ввели тире, но при этом ввели НЕ два года. Попробуйте ещё раз') if len(yearS) != 2  else ''
                yearS.sort()
                yearMinByUser = int(yearS[0])
                yearMaxByUser = int(yearS[-1])
                year = yearMaxByUser 
                publishedAfter=f'{yearMinByUser}-01-01T00:00:00Z'
                publishedBefore=f'{yearMaxByUser}-01-01T00:00:00Z'
                goC = False
            else:
                print('--- Вы НЕ ввели тире. Попробуйте ещё раз')
        else:
            goC = False
    # Сложная часть имени будущих директорий и файлов
    complicatedNamePart = f'{"" if len(q) == 0 else "_"}{q}{"" if len(contentType) == 0 else "_"}{contentType}'
    complicatedNamePart = complicatedNamePart if yearS == '' else complicatedNamePart + ' ' + str(yearMinByUser) + '-' + str(yearMaxByUser)


# In[6]:


# 0.5
def saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year):
    file = open(f'{complicatedNamePart} Temporal{slash}stage.txt', 'w+') # открыть на запись
    file.write(str(stage) + '|' + str(totalResults) + '|' + str(year)) # stageTarget принимает значения [0; 3]
    file.close()
    itemS.to_excel(f'{complicatedNamePart} Temporal{slash}{complicatedNamePart} {method}.xlsx')

if len(folderFile) == 0: # eсли НЕТ файла с id
    print('\nПроверяю наличие директории с настройками и контентом,'
          , f'сохранёнными при прошлом запросе "{complicatedNamePart[1:]}", к YouTube') # индексирование, чтобы избавиться от _
    if os.path.exists(f'{complicatedNamePart} Temporal'):
        print(f'Настройки найдены в файле stage.txt директории "{complicatedNamePart} Temporal"')
        
        print(f'Считываю их')
        file = open(f'{complicatedNamePart} Temporal{slash}stage.txt')
        file = file.read()
        stageTarget = int(file.split('|')[0])
        totalResults = int(file.split('|')[1])
        year = int(file.split('|')[2]) # если при предыдущих запусках скипт исполнился до этапа stage = 2
            
        print('Импортирую ранее выгруженный контент'
              , f'из файла "{complicatedNamePart} search.xlsx" директории "{complicatedNamePart} Temporal"')
        itemS = pandas.read_excel(f'{complicatedNamePart} Temporal{slash}{complicatedNamePart} search.xlsx', index_col=0)
    else:
        print(f'Настройки и контент по запросу "{complicatedNamePart[1:]}" не найдены. Возможно, они не сохранялись прежде') # индексирование, чтобы избавиться от _


# # 1. Первичный сбор контента методом search


# In[7]:


# 1.0 Авторские функции для обработки ошибок
def googleapiclientError(errorDescription, keyOrder, *arg): # арки: id
    # print('\n    ', errorDescription[1])
    if 'quotaExceeded' in str(errorDescription[1]):
        print('\nПохоже, квота текущего ключа закончилась; пробую перейти к следующему ключу')
        keyOrder += 1 # смена ключа
        goC = True # для повторного обращения к API с новым ключом
        problemItemId = None # для унификации со следующим блоком условий
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

def indexError(complicatedNamePart, errorDescription, method, q, slash, stageTarget, totalResults, year):
    # print('\n    ', errorDescription[1])
    print('Похоже, ключи закончились'
          , f'\nПоэтому ссохраняю выгруженный контент и текущий этап поиска в директорию "{complicatedNamePart} Temporal"')         
    if not os.path.exists(f'{complicatedNamePart} Temporal'):
        os.makedirs(f'{complicatedNamePart} Temporal')
        print(f'Директория "{complicatedNamePart} Temporal" создана')
    else:
        print(f'Директория "{complicatedNamePart} Temporal" существует')
    saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year)
    goC = False # нет смысла возвращяться со следующим keyOrder к прежнему id
    goS = False # нет смысла продолжать исполнение скрипта
    return goC, goS


# In[8]:


# 1.1 Авторская функция для метода search из API YouTube, помогающая работе с ключами
def bigSearch(API_keyS
              , channelId
              , contentType
              , goS
              , iteration
              , keyOrder
              , order
              , pageToken
              , publishedAfter
              , publishedBefore
              , q
              , year):
    goC = True
    while goC:
        try:
            youtube = api.build("youtube", "v3", developerKey = API_keyS[keyOrder])
            request = youtube.search().list(channelId=channelId
                                            , maxResults=50
                                            , order=order
                                            , pageToken=pageToken
                                            , part="snippet"
                                            , publishedAfter=publishedAfter
                                            , publishedBefore=publishedBefore
                                            , q=q
                                            , type=contentType)
            response = request.execute()

            # Для визуализации процесса
            print('      Итерация №', iteration, ', number of items', len(response['items'])
                  , '' if year == None else f', year {year - 1}'
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
            goC, goS = indexError(complicatedNamePart, errorDescription, method, q, slash, stageTarget, totalResults, year)
    addItemS = pandas.json_normalize(response['items'])
    return addItemS, goS, iteration, keyOrder, response # от response отказаться нельзя, т.к. в нём много важных ключей, даже если их знчения нули

if len(folderFile) == 0: # eсли НЕТ файла с id
    print('\nВ скрипте используются следующие аргументы метода search API YouTube:'
          , 'channelId, maxResults, order, pageToken, part, publishedAfter, publishedBefore, q, type.'
          , 'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта'
          , '\n--- Если хотите добавить другие аргументы метода search API YouTube, доступные по ссылке https://developers.google.com/youtube/v3/docs/search'
          , '-- можете сделать это внутри метода search в чанке 1.0 исполняемого сейчас скрипта')
    input('--- После прочтения этой инструкции нажмите Enter')

    channelId = None
    itemS = pandas.DataFrame()
    method = 'search'
    order = None
    orderS = ['date', 'rating', 'title', 'videoCount', 'viewCount']
    publishedAfter = None
    publishedBefore = None


# In[9]:


# 1.2 Авторская функция для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessing(complicatedNamePart, dfAdd, dfIn, goS, keyOrder, slash, stage):
    df = dfIn.copy()
    df = pandas.concat([df, dfAdd])
    columnsForCheck = []
    for column in df.columns: # выдача многих методов содержит столбец id, он оптимален для проверки дублирующхся строк
        if 'id' == column:
            columnsForCheck.append(column)
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующхся строк возможна по столбцам, содержаим в имени id.
        for column in df.columns:
            if 'id.' in column:
                columnsForCheck.append(column)
    # print('Столбцы, по которым проверяю дублирующиеся строки:', columnsForCheck)
    df = df.drop_duplicates(columnsForCheck).reset_index(drop=True)
   
    if keyOrder > (len(API_keyS) - 1):
        print('Поскольку ключи закончились,'
              , f'сохраняю выгруженный контент и текущий этап поиска в директорию "{complicatedNamePart} Temporal"')         
        if not os.path.exists(f'{complicatedNamePart} Temporal'):
                os.makedirs(f'{complicatedNamePart} Temporal')
                print(f'Директория "{complicatedNamePart} Temporal" создана')
        else:
            print(f'Директория "{complicatedNamePart} Temporal" существует')
        saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year)
        
        if goS == False:
            print('Поскольку ключи закончились, исполнение скрипта завершаю',
                  '\nПодождите сутки для восстановления ключей или подготовьте новый ключ -- и запустите скрипт с начала')
            sys.exit()
    return df


# In[10]:


# 1.3 Первый заход БЕЗ аргумента order (этап stage = 0)
stage = 0
if (len(folderFile) == 0) & (stage >= stageTarget): # eсли НЕТ файла с id и нет временных файлов с настройками и контентом
    iteration = 0 # номер итерации применения текущего метода
    
    print('\nЗаход на первую страницу выдачи')
    addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                               , channelId
                                                               , contentType
                                                               , goS
                                                               , iteration
                                                               , keyOrder
                                                               , order
                                                               , None
                                                               , publishedAfter
                                                               , publishedBefore
                                                               , q
                                                               , year)
    itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)

    totalResults = response['pageInfo']['totalResults']
    
    print('  Проход по всем следующим страницам с выдачей')
    while 'nextPageToken' in response.keys():
        pageToken = response['nextPageToken']
        addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                   , channelId
                                                                   , contentType
                                                                   , iteration
                                                                   , goS
                                                                   , keyOrder
                                                                   , order
                                                                   , pageToken
                                                                   , publishedAfter
                                                                   , publishedBefore
                                                                   , q
                                                                   , year)
        itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)
    print('  Искомых объектов', totalResults
          , ', а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))
elif stage < stageTarget:
    print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{complicatedNamePart} Temporal"')


# In[11]:


# 1.4 Цикл для прохода по значениям аргумента order, внутри которых проход по всем страницам выдачи (этап stage = 1)
stage = 1
if (len(folderFile) == 0) & (stage >= stageTarget): # eсли НЕТ файла с id и нет временных файлов с настройками и контентом
    if len(itemS) < totalResults:
    # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ включения каких-либо значений аргумента order (в т.ч. вообще БЕЗ них)

        iteration = 0 # номер итерации применения текущего метода

        print('Проход по значениям аргумента order, внутри которых проход по всем страницам выдачи')
        for order in orderS:
            addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                       , channelId
                                                                       , contentType
                                                                       , goS
                                                                       , iteration
                                                                       , keyOrder
                                                                       , order
                                                                       , None
                                                                       , publishedAfter
                                                                       , publishedBefore
                                                                       , q
                                                                       , year)
            itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)
    
            print('  Проход по всем следующим страницам с выдачей с тем же значением аргумента order:', order, '          ')
            while ('nextPageToken' in response.keys()) & (len(itemS) < totalResults) & (len(response["items"]) > 0):
            # -- второе условие -- для остановки алгоритма, если все искомые объекты найдены
                # БЕЗ какой-то из следующих страниц (в т.ч. вообще БЕЗ них)
                # третье условие -- для остановки алгоритма, если предыдущая страница выдачи содержит 0 объектов
                
                pageToken = response['nextPageToken']
                # print('pageToken', pageToken)
                addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                         , channelId
                                                                         , contentType
                                                                         , goS
                                                                         , iteration
                                                                         , keyOrder
                                                                         , order
                                                                         , pageToken
                                                                         , publishedAfter
                                                                         , publishedBefore
                                                                         , q
                                                                         , year)
                itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)
        print('  Искомых объектов', totalResults, ', а найденных С включением аргумента order:', len(itemS))
    else:
        print('Все искомые объекты найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
elif stage < stageTarget:
    print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{complicatedNamePart} Temporal"')


# In[12]:


# 1.5 Этап stage = 2
stage = 2
if (len(folderFile) == 0) & (stage >= stageTarget): # eсли НЕТ файла с id и нет временных файлов с настройками и контентом
    if len(itemS) < totalResults:
    # для остановки алгоритма, если все искомые объекты найдены БЕЗ включения каких-либо значений аргумента order (в т.ч. вообще БЕЗ них)
        print('Увы'
              , f'\nЧисло найденных объектов: {len(itemS)} -- менее числа искомых: {totalResults}')
        print('\n--- Если хотите для поиска дополнительных объектов попробовать сегментирование по годам, просто нажмите Enter'
              , '\n--- Если НЕ хотите, введите любой символ и нажмите Enter')
        if len(input()) == 0:
            print('Внутри каждого года прохожу по значениям аргумента order, внутри которых прохожу по всем страницам выдачи')
            goC = True
# ********** из чанка 1.2 + условие для goC
            while (len(itemS) < totalResults) & (goC):
                print(f'  Для года {year - 1} заход на первую страницу выдачи БЕЗ аргумента order')
                addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                           , channelId
                                                                           , contentType
                                                                           , goS
                                                                           , iteration
                                                                           , keyOrder
                                                                           , None
                                                                           , None
                                                                           , f'{year - 1}-01-01T00:00:00Z'
                                                                           , f'{year}-01-01T00:00:00Z'
                                                                           , q
                                                                           , year)
                if len(addItemS) == 0:
                    print(f'\n--- Первая страница выдачи БЕЗ аргумента order для года {year - 1} -- пуста'
                          , '\n--- Если НЕ хотите для поиска дополнительных объектов попробовать предыдущий год, просто нажмите Enter'
                          , '\n--- Если хотите, введите любой символ и нажмите Enter')
                    if len(input()) == 0:
                        goC = False
                        break
                itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)

                print(f'    Проход по всем следующим страницам с выдачей для года {year - 1} БЕЗ аргумента order')
                while 'nextPageToken' in response.keys():
                    pageToken = response['nextPageToken']
                    addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                               , channelId
                                                                               , contentType
                                                                               , goS
                                                                               , iteration
                                                                               , keyOrder
                                                                               , order
                                                                               , pageToken
                                                                               , publishedAfter
                                                                               , publishedBefore
                                                                               , q
                                                                               , year)
                    itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)

                print(f'    Искомых объектов в году {year - 1}: {totalResults},'
                      , 'а найденных БЕЗ включения каких-либо значений аргумента order:', len(itemS))          
# ********** из чанка 1.3
                if len(itemS) < totalResults:
                    print(f'  Для года {year - 1} проход по значениям аргумента order,'
                          , 'внутри которых проход по всем страницам выдачи')
                    for order in orderS:
                        addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                                   , channelId
                                                                                   , contentType
                                                                                   , goS
                                                                                   , iteration
                                                                                   , keyOrder
                                                                                   , order
                                                                                   , None
                                                                                   , publishedAfter
                                                                                   , publishedBefore
                                                                                   , q
                                                                                   , year)
                        itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)

                        print(f'    Для года {year - 1} проход по всем следующим страницам с выдачей'
                              , f'с тем же значением аргумента order:', order)
                        while ('nextPageToken' in response.keys()) & (len(itemS) < totalResults) & (len(response["items"]) > 0):
                            pageToken = response['nextPageToken']
                            addItemS, goS, iteration, keyOrder, response = bigSearch(API_keyS
                                                                                       , channelId
                                                                                       , contentType
                                                                                       , goS
                                                                                       , iteration
                                                                                       , keyOrder
                                                                                       , order
                                                                                       , pageToken
                                                                                       , publishedAfter
                                                                                       , publishedBefore
                                                                                       , q
                                                                                       , year)
                            itemS = dfsProcessing(complicatedNamePart, addItemS, itemS, goS, keyOrder, slash, stage)
                    print('    Искомых объектов', totalResults
                          , ', а найденных с добавлением сегментирования по году (год'
                          , year - 1
                          , ') и включением аргумента order:', len(itemS))
                else:
                    print('  Все искомые объекты в году', year - 1, 'найдены БЕЗ включения некоторых значений аргумента order (в т.ч. вообще БЕЗ них)')
                year -= 1
                if yearMinByUser != None: # если пользователь ограничил временнОй диапазон                
                    if (year - 1) <= yearMinByUser:
                        goC = False
                        print(f'\nЗавершил проход по заданному пользователем временнОму диапазону: {yearMinByUser}-{yearMaxByUser}')
elif stage < stageTarget:
    print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{complicatedNamePart} Temporal"')


# # 2. Сохранение в Excel выгрузки метода search

# In[13]:


# 2.0 Авторская функция для аккуратного сохранения выгрузки текущего метода в Excel
def df2fileYT(complicatedNamePart, dfIn, fileFormatChoice, method, today):
    folder = f'{today}{complicatedNamePart}'
    print('Сохраняю выгрузку метода', method, 'в директорию:', folder)
    if os.path.exists(folder):
        print('Эта директория существует')
    else:
        os.makedirs(folder)
        print('Такой директории не существовало, поэтому она создана')
    # itemS.to_excel(f'{folder}{slash}{folder} {method}.xlsx')
    
    if os.path.exists(f'{complicatedNamePart} Temporal'):
        print(f'\n--- Директорию "{complicatedNamePart} Temporal" с промежуточными результатами можете удалить вручную') 
    
    from df2file import df2file # авторский модуль для сохранения датафрейма в файл одного из форматов:
        # CSV, Excel и JSON в рамках работы с данными из YouTube
    
    # df2file(itemS) # при такой записи имя сохранаяемого файла и директория, в которую сохранить, вводятся вручную
    print('При сохранении возможно появление обширного предупреждения UserWarning: Ignoring URL.'
          , 'Оно вызвано слишком длинными URL-адресами в датафрейме и не является проблемой; его следует пролистать и перейти к диалоговому окну' )
    df2file(dfIn, f'{folder} {method}{fileFormatChoice}', folder)

if len(folderFile) == 0: # eсли НЕТ файла с id
    df2fileYT(complicatedNamePart, itemS, '.xlsx', method, today)


# # 3. Выгрузка дополнительных характеристик и контента методами playlists и playlistItems, videos, commentThreads и comments, channels


# In[14]:


# 3.0.0
if len(folderFile) == 0: # eсли НЕТ файла с id
      print(f'\nФайл "{today}{complicatedNamePart} search.xlsx" можно удалить (вручную),'
      , 'поскольку вся информация из него сохранена в основ[ой ые] файл[ы]')

print('Выгрузка метода search содержит НЕ ВСЕ доступные для выгрузки из API YouTube характеристки контента'
      , '\n--- Если хотите выгрузить дополнительные характеристики (ссылки для ознакомления с ними появятся ниже), нажмите Enter'
      , '\n--- Если НЕ хотите их выгрузить, введите любой символ и нажмите Enter. Тогда исполнение скрипта завершится')

if len(input()) > 0:
    import warnings
    warnings.filterwarnings("ignore")
    input('Скрипт исполнен. Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" \nТак и должно быть')
    sys.exit()

# 3.0.1 Этап stage = 3
stage = 3

def portionProcessing(complicatedNamePart, goS, idS, keyOrder, method, q, slash, stage, stageTarget, totalResults, year):
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
            chplviS = dfsProcessing(complicatedNamePart, addChplviS, chplviS, goS, keyOrder, slash, stage)

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
            saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year)
            goC = False
    return chplviS

# Для визуализации процесса через итерации
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


# In[16]:


# 3.1 Выгрузка дополнительных характеристик плейлистов
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
    input('--- После прочтения этой инструкции нажмите Enter')
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
            playlistS = dfsProcessing(complicatedNamePart, addPlaylistS, playlistS, goS, keyOrder, slash, stage)
    
        except googleapiclient.errors.HttpError:
            errorDescription = sys.exc_info()
            goC, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder)
        
        except IndexError:
            errorDescription = sys.exc_info()
            goC, goS = indexError(complicatedNamePart, errorDescription, method, q, slash, stageTarget, totalResults, year)

    method = 'playlistItems'
    print('\nВ скрипте используются следующие аргументы метода', method, 'API YouTube:'
          , 'part=["snippet"], playlistId, maxResults .'
          , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
          , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
          , 'https://developers.google.com/youtube/v3/docs/playlists')
    input('--- После прочтения этой инструкции нажмите Enter')
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
            playlistVideoChannelS = dfsProcessing(complicatedNamePart, addPlaylistVideoChannelS, playlistVideoChannelS, goS, keyOrder, slash, stage)

        except googleapiclient.errors.HttpError:
            errorDescription = sys.exc_info()
            goC, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder)

        except IndexError:
            errorDescription = sys.exc_info()
            goC, goS = indexError(complicatedNamePart, errorDescription, method, q, slash, stageTarget, totalResults, year)

    # Перечислить сначала id всех составляющих каждый плейлист видео через запятую и записать в ячейку,
        # затем id всех канадов, к которым относятся составляющие каждый плейлист видео, через запятую и записать в ячейку
    # display(playlistVideoChannelS) # для отладки
    for playlistId in playlistIdS:
        for column in ['snippet.resourceId.videoId', 'snippet.videoOwnerChannelId']:
            playlistVideoChannelS_snippet = playlistVideoChannelS[playlistVideoChannelS[column].notna()]
            playlistS.loc[playlistS[playlistS['id'] == playlistId].index[0], column] =\
                ', '.join(playlistVideoChannelS_snippet[playlistVideoChannelS_snippet['snippet.playlistId'] == playlistId][column].to_list())
    # display(playlistS)

    df2fileYT(complicatedNamePart, playlistS, '.xlsx', method, today)


# In[17]:


# 3.2.0 Выгрузка дополнительных характеристик видео
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
    input('--- После прочтения этой инструкции нажмите Enter')

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
            videoS = dfsProcessing(complicatedNamePart, addVideoS, videoS, goS, keyOrder, slash, stage)

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
            saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year)
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
        saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year)

    # Оформить как датафрейм id категорий из списка uniqueCategorieS и их расшифровки
    categoryNameS = pandas.json_normalize(response['items'])

    # Заменить индексы датафрейма с расшифровками значениями столбца id
    categoryNameS.index = categoryNameS['id'].to_list()

    # Добавить расшифровки категорий в новый столбец categoryName датафрейма с видео
    for row in categoryNameS.index:
        videoS.loc[videoS['snippet.categoryId'] == row, 'categoryName'] = categoryNameS['snippet.title'][row]

    columnsToJSON = [] # столбцы с JSON для сохранения в отдельный JSON
    for column in ['snippet.tags'
                   , 'topicDetails.topicCategories'
                   , 'contentDetails.regionRestriction.blocked'
                   , 'contentDetails.regionRestriction.allowed']:
        if column in videoS.columns: columnsToJSON.append(column)
    print('В выгрузке метода', method, 'есть столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
          , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')
    df2fileYT(complicatedNamePart, videoS.drop(columnsToJSON, axis=1), '.xlsx', f'{method} Other varS', today)
    columnsToJSON.append('id')
    df2fileYT(complicatedNamePart, videoS[columnsToJSON], '.json', f'{method} JSON varS', today)

    commentS = pandas.DataFrame() # не в следующем ченке, чтобы иметь возможность перезапускать его, не затирая промежуточный результат выгрузки


# In[ ]:


# 3.2.1 Выгрузка комментариев к видео
if len(videoS) > 0:
    print('\n--- Если хотите выгрузить комментарии к видео (в отдельный файл),'
          , f'содержащимся в файле "{today}{complicatedNamePart} {method}.xlsx" директории "{today}{complicatedNamePart}",'
          , 'просто нажмите Enter'
          , '\n--- Если НЕ хотите выгрузить комментарии, введите любой символ и нажмите Enter')
    if len(input()) == 0:
        def downloadComments(API_keyS, goS, id, idS, keyOrder, maxResults, method, page, pageToken, part): # id видео или комментария
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
                            response = youtube.comments().list(maxResults=maxResults, pageToken=pageToken, part=part, parentId=id).execute()
                            goC_1 = False
                        elif method == 'commentThreads':
                            response = youtube.commentThreads().list(maxResults=maxResults, pageToken=pageToken, part=part, videoId=id).execute()
                            goC_1 = False
                        else:
                            print('--- В функцию downloadComments подано неправильное имя метода выгрузки комментариев'
                                  , '\n--- Подайте, пожалуйста, правильное..')
                    addCommentS = pandas.json_normalize(response['items'])
                    pageToken = response['nextPageToken'] if 'nextPageToken' in response.keys() else None
                    # print('nextPageToken', pageToken)
                    if page != '':
                        print('  Видео №', idS.index(id) + 1, 'из', len(idS), '. Страница выдачи №', page, '          ', end='\r')
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
                    goC_0, keyOrder, problemItemId = googleapiclientError(errorDescription, keyOrder, id)
                
                except IndexError:
                    errorDescription = sys.exc_info()
                    # print('\n    ', errorDescription[1])
                    # print('Похоже, ключи закончились')
                    # goC_0 = False # нет смысла возвращяться со следующим keyOrder к прежнему id
                    # goS = False
                    goC_0, goS = indexError(errorDescription)
                               
                except TimeoutError:
                    errorDescription = sys.exc_info()
                    print(errorDescription[1])
                    time.sleep(10)
                    
            return addCommentS, errorDescription, goS, keyOrder, page, pageToken, problemItemId
        
# ********** commentS
        # commentS = pandas.DataFrame() # фрагмент вынесен в предыдущий ченк, чтобы иметь возможность перезапускать этот чанк,
        # не затирая промежуточный результат выгрузки        
        maxResults = 100
        method = 'commentThreads'
        part = 'id, replies, snippet'
        problemVideoIdS = pandas.DataFrame()
        print('В скрипте используются следующие аргументы метода', method, 'API YouTube:'
              , 'part=["snippet", "id", "replies"], maxResults, videoId .'
              , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
              , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
              , 'https://developers.google.com/youtube/v3/docs/commentThreads')
        input('--- После прочтения этой инструкции нажмите Enter')
        
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
            commentS = dfsProcessing(complicatedNamePart, addCommentS, commentS, goS, keyOrder, slash, stage)
            if errorDescription != None: problemVideoIdS.loc[problemVideoId, 'errorDescription'] = errorDescription
            while pageToken != None:
                addCommentS, errorDescription, goS, keyOrder, page, pageToken, problemItemId =\
                    downloadComments(API_keyS, goS, videoId, videoIdS, keyOrder, maxResults, method, page, pageToken, part)
                commentS = dfsProcessing(complicatedNamePart, addCommentS, commentS, goS, keyOrder, slash, stage)
                if errorDescription != None: problemVideoIdS.loc[problemVideoId, 'errorDescription'] = errorDescription
        commentS = commentS.drop(['kind', 'etag', 'id', 'snippet.channelId', 'snippet.videoId'], axis=1) # т.к. дублируются содержательно
        
        def prefixDropper(df): # избавиться от префиксов
            dfNewColumnS = []
            for column in df.columns:
                if 'snippet.topLevelComment.' in column:
                    column = column.replace('snippet.topLevelComment.', '')
                dfNewColumnS.append(column)
            df.columns = dfNewColumnS # перезаписать названия столбцов
            return df
        commentS = prefixDropper(commentS)
        df2fileYT(complicatedNamePart, commentS, '.xlsx', 'commentS', today)

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
        df2fileYT(complicatedNamePart, replieS, '.xlsx', 'replieS', today)
        
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
        commentReplieS = dfsProcessing(complicatedNamePart, replieS, commentReplieS, goS, keyOrder, slash, stage)
        method = 'comments'
        part = 'id, snippet'
        textFormat = 'plainText' # = 'html' по умолчанию
        problemCommentIdS = pandas.DataFrame()
        replieS = pandas.DataFrame() # зачем? См. этап 4.2 ниже
        print('В скрипте используются следующие аргументы метода', method, 'API YouTube:'
              , 'part=["snippet", "id"], maxResults, parentId, textFormat .'
              , 'Эти аргументы, кроме part, пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
              , 'Если хотите добавить другие аргументы метода', method, 'API YouTube, можете ознакомиться с ними по ссылке:'
              , 'https://developers.google.com/youtube/v3/docs/commentThreads')
        input('--- После прочтения этой инструкции нажмите Enter')
        print('Проход по id всех родительских (topLevel) комментариев с недостачей ответов для выгрузки этих ответов')
        commentIdS = commentReplieS['id'][commentReplieS['Недостача_ответов'] > 0]
        for commentId in tqdm(commentIdS):
            page = 0 # номер страницы выдачи
            addReplieS, errorDescription, goS, keyOrder, page, pageToken, problemCommentId =\
                downloadComments(API_keyS, goS, commentId, commentIdS, keyOrder, maxResults, method, '', None, part)
            if errorDescription != None: problemCommentIdS.loc[problemCommentId, 'errorDescription'] = errorDescription
            replieS = dfsProcessing(complicatedNamePart, addReplieS, replieS, goS, keyOrder, slash, stage)
            while pageToken != None:
                addReplieS, errorDescription, goS, keyOrder, page, pageToken, problemCommentId =\
                    downloadComments(API_keyS, goS, commentId, commentIdS, keyOrder, maxResults, method, '', pageToken, part)
                if errorDescription != None: problemCommentIdS.loc[problemCommentId, 'errorDescription'] = errorDescription
                replieS = dfsProcessing(complicatedNamePart, addReplieS, replieS, goS, keyOrder, slash, stage)
        print('Ответов выгружено', len(replieS)
              , '; проблемные родительские (topLevel) комментарии:', problemCommentIdS)
        
        # Для совместимости датафреймов добавить столбцы`snippet.totalReplyCount` и `Недостача_ответов`
        replieS.loc[:, 'snippet.totalReplyCount'] = 0
        replieS.loc[:, 'Недостача_ответов'] = 0
        
        # Удалить столбец `snippet.parentId`, т.к. и из столбца `id` всё ясно
        replieS = replieS.drop('snippet.parentId', axis=1)
        
        commentReplieS = dfsProcessing(complicatedNamePart, replieS, commentReplieS, goS, keyOrder, slash, stage)
        df2fileYT(complicatedNamePart, commentReplieS, '.xlsx', 'commentReplieS', today)


# In[ ]:


# 3.3 Выгрузка дополнительных характеристик каналов
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
    input('--- После прочтения этой инструкции нажмите Enter')
    
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
            channelS = dfsProcessing(complicatedNamePart, addСhannelS, channelS, goS, keyOrder, slash, stage)

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
            saveSettings(complicatedNamePart, method, q, slash, stageTarget, totalResults, year)
            goC = False
    columnsToJSON = [] # столбцы с JSON для сохранения в отдельный JSON
    for column in ['topicDetails.topicIds'
                   , 'topicDetails.topicCategories'
                   , 'brandingSettings.hints'
                   , 'brandingSettings.channel.featuredChannelsUrls']:
        if column in videoS.columns: columnsToJSON.append(column)
    print('В выгрузке метода', {method}, 'есть столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
          , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')
    df2fileYT(complicatedNamePart, videoS.drop(columnsToJSON, axis=1), '.xlsx', f'{method} Other varS', today)
    columnsToJSON.append('id')
    df2fileYT(complicatedNamePart, videoS[columnsToJSON], '.json', f'{method} JSON varS', today)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
input('Скрипт исполнен. Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" \nТак и должно быть')
sys.exit()


# In[ ]:


# https://stackoverflow.com/questions/30475309/get-youtube-trends-v3-country-wise-in-json -- про тренды
