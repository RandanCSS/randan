# 0.0.0 Активировать требуемые для работы скрипта модули и пакеты
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from datetime import date, datetime
        from randan.tools.df2file import df2file # авторский модуль для сохранения датафрейма в файл одного из форматов: CSV, Excel и JSON в рамках работы с данными из социальных медиа
        from randan.tools.calendarWithinYear import calendarWithinYear # авторский модуль для работы с календарём конкретного года
        from randan.tools import files2df # авторский модуль для оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
        from vk_requests.exceptions import VkAPIError # на всякий случай, хотя и без этого класса ошибки нормлаьно обрабатываются
        from tqdm import tqdm
        import os, pandas, re, shutil, time, vk_requests, warnings
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print('Пакет', module, 'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

tqdm.pandas() # для визуализации прогресса функций, применяемых к датафреймам

print('    Для исполнения скрипта не обязательны пререквизиты (предшествующие скрпиты и файлы с данными).'
      , 'Но от пользователя требуется предварительно получить API key для авторизации в API ВК (см. примерную инструкцию: https://docs.google.com/document/d/1dRqPGzLgr1wLp-_N6iuuZCmzCqrjYg1PuH7G7yomYdw ).'
      , 'Для получения API key следует создать приложение и из него скопировать сервисный ключ.'
      , 'Приложение -- это как бы аккаунт для предоставления ему разных уровней авторизации (учётных данных, или Credentials) для доступа к содержимому ВК.'
      , 'Авторизация сервисным ключом позволяет использовать некоторые методы API -- в документации API ВК ( https://dev.vk.com/ru/method ) они помечены серым кружком'
      , '(одним или в сочетании с кружками другого цвета). Его достаточно, если выполнять действия, которые были бы доступны Вам как обычному пользователю ВК:'
      , 'посмотреть открытые персональные и групповые страницы, почитать комментарии и т.п. Если же Вы хотите выполнить действия вроде удаления поста из чужого аккаунта,'
      , 'то Вам потребуется дополнительная авторизация.'
      , '\n    ВК может ограничить действие Вашего ключа или вовсе заблокировать его, если сочтёт, что Вы злоупотребляете автоматизированным доступом.'
      , '\n    Скрипт нацелен на выгрузку характеристик контента ВК методом его API newsfeed.search. Причём количество объектов выгрузки максимизируется путём её сегментирования по годам и месяцам.'
      , '\n    Для корректного исполнения скрипта просто следуйте инструкциям в возникающих по ходу его исполнения сообщениях. Скрипт исполняется и под MC OS, и под Windows.'
      , '\n    Преимущества скрипта перед выгрузкой контента из ВК вручную: гораздо быстрее, гораздо большее количество контента, его организация в формате таблицы Excel.'
      , 'Преимущества скрипта перед выгрузкой контента через непосредственно API ВК: гораздо быстрее, гораздо большее количество контента,'
      , 'не требуется тщательно изучать обширную и при этом неполную документацию методов API ВК')
input('--- После прочтения этой инструкции нажмите Enter')

# 0.0.1 Некоторые базовые настройки запроса к API YouTube
fileFormatChoice = '.xlsx' # базовый формат сохраняемых файлов; формат .json добавляется опционально через наличие columnsToJSON
folder = ''
folderFile = ''
goS = True
itemS = pandas.DataFrame()
slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
stageTarget = 0 # stageTarget принимает значения [0; 3] и относится к стадиям скрипта
temporalName = ''
yearsRange = ''

today = date.today().strftime("%Y%m%d") # запрос сегодняшней даты в формате yyyymmdd
print('\nТекущяя дата:', today, '-- она будет использована для формирования имён создаваемых директорий и файлов (во избежание путаницы в директориях и файлах при повторных запусках)\n')
# print('Сегодня год:', today[:4])
# print('Сегодня месяц:', today[4:6])
# print('Сегодня день:', today[6:])
year = int(today[:4]) # в случае отсутствия пользовательского временнОго диапазона
    # с этого года возможно сегментирование по годам вглубь веков (пока выдача не пустая)
yearMinByUser = None # в случае отсутствия пользовательского временнОго диапазона
yearMaxByUser = None # в случае отсутствия пользовательского временнОго диапазона

# 0.1 Поиск следов прошлых запусков: ключей и данных; в случае их отсутствия -- получение настроек и (опционально) данных от пользователя
# 0.1.0 Функции блока:
# для аккуратного сохранения выгрузки текущего метода в Excel в директорию, создаваемую в текущей директории
def df2fileVK(columnsToJSON, complicatedNamePart, dfIn, fileFormatChoice, method, today):
    folder = f'{today}{complicatedNamePart}'
    print('Сохраняю выгрузку метода', method)
    if os.path.exists(folder) == False:
        print('Такой директории не существовало, поэтому она создана')
        os.makedirs(folder)
    # else:
        # print('Эта директория существует')

    # df2file(itemS) # при такой записи имя сохранаяемого файла и директория, в которую сохранить, вводятся вручную
    # print('При сохранении возможно появление обширного предупреждения UserWarning: Ignoring URL.'
    #       , 'Оно вызвано слишком длинными URL-адресами в датафрейме и не является проблемой; его следует пролистать и перейти к диалоговому окну' )

    if (columnsToJSON != None) | (columnsToJSON != []):
        for column in columnsToJSON:
            if column not in itemS.columns: columnsToJSON.remove(column)
        print('В выгрузке метода', method, 'есть столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
              , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')

        df2file(dfIn.drop(columnsToJSON, axis=1), f'{folder} {method} Other varS{fileFormatChoice}', folder)
        columnsToJSON.append('id')
        df2file(dfIn[columnsToJSON], f'{folder} {method} JSON varS.json', folder)

    else: df2file(dfIn, f'{folder} {method}{fileFormatChoice}', folder)

# для сохранения следа исполнения скрипта, натолкнувшегося на ошибку, непосредственно в директорию Temporal в текущей директории
def saveSettings(columnsToJSON, complicatedNamePart, fileFormatChoice, itemS, method, q, slash, stageTarget, targetCount, today, year, yearsRange):
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
    file.write(yearsRange) # пользовательский временнОй диапазон
    file.close()

    # itemS.to_excel(f'{today}{complicatedNamePart}_Temporal{slash}{complicatedNamePart} {method}.xlsx')
    if '.' in method: df2fileVK(columnsToJSON, f'{complicatedNamePart}_Temporal', itemS, fileFormatChoice, method.split('.')[0] + method.split('.')[1].capitalize(), today)
        # чтобы избавиться от лишней точки в имени файла
    else: df2fileVK(columnsToJSON, f'{complicatedNamePart}_Temporal', itemS, fileFormatChoice, method, today)
    
    print('Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы,'
          , 'УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта')
    shutil.rmtree(rootName, ignore_errors=True)

    # для парсинга пользовательского временнОго диапазона в случае использования сохранённого следа
        # и в случае назначения пользовательского временнОго диапазона
def yearsRangeParser(yearsRange):
    yearsRange.sort()
    yearMinByUser = int(yearsRange[0])
    yearMaxByUser = int(yearsRange[-1])
    yearsRange = f'{yearMinByUser}-{yearMaxByUser}'
    return yearMaxByUser, yearMinByUser, yearsRange

rootNameS = os.listdir()
# Поиск ключей
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
            print('-- далее буд[еу]т использован[ы] эт[и] ключ[и]')

            from randan.tools.textPreprocessing import multispaceCleaner # авторский модуль для предобработки нестандартизированнрого текста
            API_keyS = multispaceCleaner(API_keyS)
            while API_keyS[-1] == ',': API_keyS = API_keyS[:-1] # избавиться от запятых в конце текста

            file = open("credentialsVK.txt", "w+") # открыть на запись
            file.write(API_keyS)
            file.close()
            break
        else:
            print('--- Вы ничего НЕ ввели. Попробуйте ещё раз..')
API_keyS = API_keyS.split(', ')
print('Количество ключей:', len(API_keyS), '\n')
keyOrder = 0

# 0.1.1 Скрипт может начаться с данных, сохранённых при прошлом исполнении скрипта, натолкнувшемся на ошибку
# 0.1.2 Поиск данных
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

        # file = open(f'{rootName}{slash}contentType.txt')
        # contentType = file.read()
        # file.close()

        # file = open(f'{rootName}{slash}channelIdForSearch.txt')
        # channelIdForSearch = file.read()
        # file.close()

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
        # print('- пользователь НЕ определил тип контента' if contentType == '' else  f'- пользователь определил тип контента как "{contentType}"')
        # if contentType == 'video':
        #     print('- пользователь НЕ выбрал конкретный канал для выгрузки видео' if channelIdForSearch == '' else  f'- пользователь выбрал канал с id "{channelIdForSearch}" для выгрузки видео')
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
            
            if yearsRange != '':
                yearsRange = yearsRange.split('-')
                yearMaxByUser, yearMinByUser, yearsRange = yearsRangeParser(yearsRange)
# 0.1.3 Данные, сохранённые при прошлом запуске скрипта, загружены;
    # их метаданные (q, contentType, yearsRange, stageTarget) будут использоваться при исполнении скрипта
            break
        elif decision == 'R': shutil.rmtree(rootName, ignore_errors=True)

# 0.1.4 Если такие данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку, не найдены, возможно, пользователь хочет подать свои данные для их дополнения
if os.path.exists(f'{rootName}{slash}{temporalName}') == False: # если itemsTemporal, в т.ч. пустой, не существует
        # и, следовательно, не существуют данные, сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку
    print('Не найдены подходящие данные, гипотетически сохранённые при прошлом запуске скрипта, натолкнувшемся на ошибку')
    print('\nВозможно, Вы располагаете файлом, в котором есть ранее выгруженные из ВК методом newsfeed.search данные, и который хотели бы дополнить?'
          , 'Или планируете первичный сбор контента?'
          , '\n--- Если планируете первичный сбор, нажмите Enter'
          , '\n--- Если располагаете файлом, укажите полный путь, включая название файла, и нажмите Enter')
    while True:
        folderFile = input()
        if len(folderFile) == 0: break
        else:
            itemS, error, folder = files2df.files2df(folderFile)
            if error != None:
                if 'No such file or directory' in error:
                    print('Путь:', folderFile, '-- не существует; попробуйте, пожалуйста, ещё раз..')
            else: break
        # display(itemS)
# 0.1.5 Теперь определены объекты: folder и folderFile (оба пустые или пользовательские), itemS (пустой или с прошлого запуска, или пользовательский), slash
# 0.1.6 Пользовательские настройки запроса к API ВК
    # # Контент: канал или видео? Или вообще плейлист?
    # while True:
    #     print('--- Если НЕ требуется определить тип контента, нажмите Enter'
    #           , ' \n--- Если требуется определить, введите символ: c -- channel, p -- playlist, v -- video -- и нажмите Enter')
    #     contentType = input()

    #     if contentType.lower() == '':
    #         contentType = ''
    #         break
    #     elif contentType.lower() == 'c':
    #         contentType = 'channel'
    #         break
    #     elif contentType.lower() == 'p':
    #         contentType = 'playlist'
    #         break
    #     elif contentType.lower() == 'v':
    #         contentType = 'video'
    #         break
    #     else:
    #         print('--- Вы ввели что-то не то; попробуйте, пожалуйста, ещё раз..')
    print('Скрипт умеет искать контент в постах открытых аккаунтов по текстовому запросу-фильтру'
          , '\n--- Введите текст запроса-фильтра, который ожидаете найти в постах, после чего нажмите Enter')
    if len(folderFile) > 0: print('ВАЖНО! В результате исполнения текущего скрипта данные из указанного Вами файла'
        , folderFile
        , 'будут дополнены актуальными данными из выдачи скрипта'
        , '(возможно появление новых объектов и новых столбцов, а также актуализация содержимого столбцов),'
        , 'поэтому, вероятно, следует ввести тот же запрос-фильтр, что и при формировании указанного Вами файла')
    q = input()

    # Ограничения временнОго диапазона
    while True:
        print('Если требуется конкретный временнОй диапазон, то можно использовать его не на текущем этапе выгрузки данных, а на следующем этапе -- предобработки датафрейма с выгруженными данными.'
              , 'Проблема в том, что без назначения временнОго диапазона метод newsfeed.search выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту.'
              , '\n--- Поэтому если всё же требуется назначить временнОй диапазон на этапе выгрузки данных, назначьте его годами (а не более мелкими единицами времени).'
              , 'Для назначения диапазона введите без кавычек минимальный год диапазона, тире, максимальный год диапазона (минимум и максимум могут совпадать) и нажмите Enter'
              , '\n--- Если НЕ требуется назначить временнОй диапазон на этапе выгрузки данных, нажмите Enter')
        yearsRange = input()
        if len(yearsRange) != 0:
            yearsRange = re.sub(r' *', '', yearsRange)
            if '-' in yearsRange:
                yearsRange = yearsRange.split('-')
                if len(yearsRange) == 2:
                    yearMaxByUser, yearMinByUser, yearsRange = yearsRangeParser(yearsRange)
                    year = yearMaxByUser
                    break
                else: print('--- Вы ввели тире, но при этом ввели НЕ два года. Попробуйте ещё раз..')
            else: print('--- Вы НЕ ввели тире. Попробуйте ещё раз..')
        else: break
# Сложная часть имени будущих директорий и файлов
complicatedNamePart = '_VK'
# complicatedNamePart += f'{"" if len(contentType) == 0 else "_"}{contentType}'
# complicatedNamePart += f'{"" if len(channelIdForSearch) == 0 else "_channelId"}{channelIdForSearch}'
complicatedNamePart += f'{"" if len(q) == 0 else "_"}{q}'
complicatedNamePart += f'{"" if len(yearsRange) == 0 else "_"}{yearMinByUser}-{yearMaxByUser}'
# print('complicatedNamePart', complicatedNamePart)


# In[ ]:


# # 1. Первичный сбор контента методом search


# In[ ]:


# 1.1 Авторская функция для метода search из API YouTube, помогающая работе с ключами
def bigSearch(API_keyS, goS, iteration, keyOrder, pause, q, start_from, start_time, end_time, vk_requests):
    while True:
        try:
            api = vk_requests.create_api(service_token=API_keyS[keyOrder])
            response = api.newsfeed.search(q=q, start_from=start_from, start_time=start_time, end_time=end_time, extended=1)
            # print('response', response)

            # Для визуализации процесса
            print('    Итерация №', iteration, ', number of items', len(response['items']), '                    ', end='\r')
            iteration += 1
            
            break
        except:
            errorDescription = sys.exc_info()
            # print('\n    ', errorDescription[1])
            if 'Too many requests per second' in str(errorDescription[1]):
                # print('  keyOrder до замены', '                    ') # для отладки
                print('\n    ', errorDescription)
                keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
                print(f'\nПохоже, ключ попал под ограничение вследствие слишком высокой частоты обращения скрипта к API; пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту')
                # print('  keyOrder после замены', keyOrder, '                    ') # для отладки
                pause += 0.25
            elif 'User authorization failed' in str(errorDescription[1]):
                print('\nПохоже, аккаунт попал под ограничение. Оно может быть снято с аккаунта сразу или спустя какое-то время.'
                      , 'Подождите или подготовьте новый ключ в другом аккаунте. И запустите скрипт с начала')
                print('\n    ', errorDescription)
                response = {'items': [], 'total_count': 0
                            , } # принудительная выдача для response без request.execute()
                goS = False # нет смысла продолжать исполнение скрипта
                break # и, следовательно, нет смысла в новых итерациях цикла                
            else:
                print('  Похоже, проблема НЕ в слишком высокой частоте обращения скрипта к API((')
                print('\n    ', errorDescription)
                goS = False # нет смысла продолжать исполнение скрипта
                break # и, следовательно, нет смысла в новых итерациях цикла
    dfAdd = pandas.json_normalize(response['items'])

    # Сменить формат представления дат, класс данных столбцов с id, создать столбец с кликабельными ссылками на контент. Здесь, а не в конце, поскольку нужна совместимость с itemS из Temporal и от пользователя
    if len(dfAdd) > 0:
        dfAdd['date'] = dfAdd['date'].apply(lambda content: datetime.fromtimestamp(content).strftime('%d.%m.%Y'))
        dfAdd['URL'] = dfAdd['from_id'].astype(str)
        dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-') == False].index, 'URL'] = 'id' + dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-') == False].index, 'URL']
        dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-')].index, 'URL'] = dfAdd.loc[dfAdd[dfAdd['URL'].str.contains('-')].index, 'URL'].str.replace('-', 'public')
        dfAdd['URL'] = 'https://vk.com' + '/' + dfAdd['URL'] + '?w=' + dfAdd['inner_type'].str.split('_').str[0] + dfAdd['owner_id'].astype(str) + '_' + dfAdd['id'].astype(str)

    return dfAdd, goS, iteration, keyOrder, pause, response

# 1.2 Авторская функция для обработки выдачи любого из методов, помогающая работе с ключами
def dfsProcessing(columnsToJSON, complicatedNamePart, fileFormatChoice, dfAdd, dfFinal, dfIn, goS, method, q, slash, stage, targetCount, today, year, yearsRange):
    df = pandas.concat([dfIn, dfAdd])        
    columnsForCheck = []
    for column in df.columns: # выдача многих методов содержит столбец id, он оптимален для проверки дублирующхся строк
        if 'id' == column:
            columnsForCheck.append(column)
    if columnsForCheck == []: # для выдач, НЕ содержащих столбец id, проверка дублирующхся строк возможна по столбцам, содержаим в имени id
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
        saveSettings(columnsToJSON, complicatedNamePart, fileFormatChoice, itemS, method, q, slash, stage, targetCount, today, year, yearsRange)
        print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit"'
              , '\nТак и должно быть'
              , '\nМодуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
        sys.exit()
    return df

# 1.3 Первое обращение к API БЕЗ аргументов start_time, end_time (этап stage = 0)
method = 'newsfeed.search'
columnsToJSON = ['attachments', 'copy_history']
iteration = 0 # номер итерации применения текущего метода
pause = 0.25
stage = 0

# if len(folderFile) == 0: # eсли НЕТ файла с id
print(f'В скрипте используются следующие аргументы метода {method} API YouTube:'
      , 'q, start_from, start_time, end_time, expand.'
      , 'Эти аргументы пользователю скрипта лучше не кастомизировать во избежание поломки скрипта.'
      , f'Если хотите добавить другие аргументы метода {method} API ВК, доступные по ссылке https://dev.vk.com/ru/method/newsfeed.search ,'
      , f'-- можете сделать это внутри метода {method} в чанке 1.1 исполняемого сейчас скрипта')
input('--- После прочтения этой инструкции нажмите Enter')

# if (len(folderFile) == 0) & (stage >= stageTarget): # eсли НЕТ файла с id и нет временного файла stage.txt с указанием пропустить этап
if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
    print('\nПервое обращение к API -- прежде всего, чтобы узнать примерное число доступных релевантных объектов')
    itemsAdditional, goS, iteration, keyOrder, pause, response = bigSearch(API_keyS, goS, iteration, keyOrder, pause, q, None, None, None, vk_requests)
    targetCount = response['total_count']
    # if len(itemS) < targetCount: # на случай достаточности
    itemS = dfsProcessing(columnsToJSON, complicatedNamePart, fileFormatChoice, itemsAdditional, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    print('  Проход по всем следующим страницам с выдачей          ')
    while 'next_from' in response.keys():
        start_from = response['next_from']
        # print('    start_from', start_from) # для отладки
        itemsAdditional, goS, iteration, keyOrder, pause, response = bigSearch(API_keyS, goS, iteration, keyOrder, pause, q, start_from, None, None, vk_requests)
        itemS = dfsProcessing(columnsToJSON, complicatedNamePart, fileFormatChoice, itemsAdditional, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
    print('  Искомых объектов', targetCount, ', а найденных БЕЗ сегментирования по годам и месяцам:', len(itemS))

# 1.4 Этап stage = 1
stage = 1
if stage >= stageTarget: # eсли нет временного файла stage.txt с указанием пропустить этап
    if len(itemS) < targetCount:
    # -- для остановки алгоритма, если все искомые объекты найдены БЕЗ сегментирования по годам и месяцам
        print('Увы, без назначения временнОго диапазона метод newsfeed.search выдаёт ограниченное количество объектов, причём наиболее приближенных к текущему моменту.'
          , 'Поэтому внутри каждого года, начиная с теущего, помесячно выгружаю контент, после чего меняю год -- вглубь веков, пока не достигну заданной пользователем левой границы временнОго диапазона,'
          , 'или года с пустой выдачей')
        while True:
            # print('Ищу текст запроса-фильтра в контенте за', year, 'год')
            calendar = calendarWithinYear(year)
            itemsYearlyAdditional = pandas.DataFrame()
            calendarColumnS = calendar.columns
            if year == int(today[:4]): calendarColumnS = calendarColumnS[:int(today[4:6])] # чтобы исключить проход по будущим месяцам текущего года
            for month in calendarColumnS:
                print('Ищу текст запроса-фильтра в контенте за',  month, 'месяц', year, 'года', '               ') # , end='\r'
                print('  Заход на первую страницу выдачи', '               ', end='\r')
                start_time = int(time.mktime(datetime(year, int(month), 1).timetuple()))
                end_time = int(time.mktime(datetime(year, int(month), int(calendar[month].dropna().index[-1])).timetuple()))
                # print('\n  Period from start_time', start_time, 'to end_time', end_time) # для отладки
                itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = bigSearch(API_keyS, goS, iteration, keyOrder, pause, q, None, start_time, end_time, vk_requests)
                itemsYearlyAdditional = dfsProcessing(columnsToJSON
                                                      , complicatedNamePart
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
                    itemsMonthlyAdditional, goS, iteration, keyOrder, pause, response = bigSearch(API_keyS, goS, iteration, keyOrder, pause, q, start_from, start_time, end_time, vk_requests)
                    itemsYearlyAdditional = dfsProcessing(columnsToJSON
                                                          , complicatedNamePart
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
            itemS = dfsProcessing(columnsToJSON, complicatedNamePart, fileFormatChoice, itemsYearlyAdditional, itemS, itemS, goS, method, q, slash, stage, targetCount, today, year, yearsRange)
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

    # # Сменить формат представления дат, класс данных столбцов с id, создать столбец с кликабельными ссылками на контент
    # itemS['date'] = itemS['date'].progress_apply(lambda content: datetime.fromtimestamp(content).strftime('%d.%m.%Y'))
    # itemS['URL'] = itemS['from_id'].astype(str)
    # itemS.loc[itemS[itemS['URL'].str.contains('-') == False].index, 'URL'] = 'id' + itemS.loc[itemS[itemS['URL'].str.contains('-') == False].index, 'URL']
    # itemS.loc[itemS[itemS['URL'].str.contains('-')].index, 'URL'] = itemS.loc[itemS[itemS['URL'].str.contains('-')].index, 'URL'].str.replace('-', 'public')
    # itemS['URL'] = 'https://vk.com' + '/' + itemS['URL'] + '?w=' + itemS['inner_type'].str.split('_').str[0] + itemS['owner_id'].astype(str) + '_' + itemS['id'].astype(str)

    # pandas.set_option('display.max_columns', None)
    display(itemS.head())
    print('Число столбцов:', itemS.shape[1], ', число строк', itemS.shape[0])

elif stage < stageTarget:
    print(f'\nЭтап {stage} пропускаю согласно настройкам из файла stage.txt в директории "{today}{complicatedNamePart}_Temporal"')


# In[ ]:


# # 2. Сохранение в Excel выгрузки метода search


# In[ ]:


# columnsToJSON = ['attachments', 'copy_history'] # столбцы с JSON для сохранения в отдельный JSON
# for column in columnsToJSON:
#     if column not in itemS.columns: columnsToJSON.remove(column)
# print('В выгрузке метода', method, 'есть столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
#       , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')
# df2fileVK(complicatedNamePart, itemS.drop(columnsToJSON, axis=1), '.xlsx', f'{method.split('.')[0] + method.split('.')[1].capitalize()} Other varS', today) # чтобы избавиться от лишней точки в имени файла
# columnsToJSON.append('id')
# df2fileVK(complicatedNamePart, itemS[columnsToJSON], '.json', f'{method.split('.')[0] + method.split('.')[1].capitalize()} JSON varS', today) # чтобы избавиться от лишней точки в имени файла

df2fileVK(columnsToJSON,complicatedNamePart, itemS, '.xlsx', method.split('.')[0] + method.split('.')[1].capitalize(), today) # чтобы избавиться от лишней точки в имени файла

# In[ ]:


print('Скрипт исполнен. Поскольку данные, сохранённые при одном из прошлых запусков скрипта в директорию Temporal, успешно использованы,'
      , 'УДАЛЯЮ её во избежание путаницы при следующих запусках скрипта')
shutil.rmtree(rootName, ignore_errors=True)

warnings.filterwarnings("ignore")
print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'
      , '\nМодуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
sys.exit()

# warnings.filterwarnings("ignore")
# print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'
#       , '\nМодуль создан при финансовой поддержке Российского научного фонда по гранту 22-28-20473')
# input()
# sys.exit()
