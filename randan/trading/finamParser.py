# coding: utf-8

'''
A proprietary module designed to facilitate the scraping and parsing of data from the finam.ru website
Авторский модуль для упрощения выгрузки данных с сайта finam.ru и их парсинга
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from datetime import date, datetime
        from io import StringIO
        from IPython.display import display

        from randan.tools import coLabAdaptor, forSelenium, textPreprocessor # авторские модули для..
            # (а) адаптации текущего скрипта к файловой системе CoLab
            # (б) упрощения некоторых оперций в selenium
            # (в) предобработки нестандартизированнрого текста

        from randan.trading import getAssets # авторский модуль для..
            # (а) выяснения, какие инструменты (акции, облигации и т.д.) есть в портфеле, на основе брокерских отчётов

        from selenium import webdriver
        from selenium.webdriver.common.by import By # для поиска элементов HTML-кода
        from selenium.webdriver.support import expected_conditions
        from selenium.webdriver.support.ui import WebDriverWait

        import os, pandas, re, selenium.common.exceptions, time, traceback
        break # выход из цикла for attempt in range(3)

    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print(
f'''Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 3
'''
              )
        check_call([sys.executable, "-m", "pip", "install", module])
        if  attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break # выход из цикла for attempt in range(3)

coLabFolder = coLabAdaptor.coLabAdaptor()

folder = coLabFolder
slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
if folder == None: folder = ''
else: folder += slash

version_main = 150

# Авторские функции..
# .. обработки облигаций, относящихся к одному Identifier
def bondsOfIdentifierProcessor(attemptsMax,  bondsFinAM_in, bondsFinAM_row, bondsOfIdentifier, columnS_target, driver, folder, pause, slash, source, sourceRow, urlInitial, version_main):
    bondsFinAM = bondsFinAM_in.copy()
    # print('bondsOfIdentifier.index :', bondsOfIdentifier.index) # для отладки

    # # Архитектура
    # # /html/body/div[2]/div[3]/div/table/tbody/tr/td[1]/div/div[1]/table/tbody/tr[7]/td[1]/span
    # elementAnchor = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div/table/tbody/tr/td[1][contains(., 'Номинальный объем:')]").text

    goS, xPath = forSelenium.tryerSleeper(attemptsMax, [2, 5], driver, pause, ['/html/body/div[', ']/div[3]/div/table/tbody/tr/td[1]'])
    if goS == False:
        print('Следует проверить xPath вручную')
        return bondsFinAM, bondsFinAM_row, goS

    trCounterOutS = bondsOfIdentifier.index[:-1] if 'Страница: ' in bondsOfIdentifier.index[-1] else bondsOfIdentifier.index
    for trCounterOut in trCounterOutS:
        trCounterOut = int(trCounterOut) + 1 # т.к. нумерация тегов строк начинается с шапки 
        # print('trCounterOut:', trCounterOut) # для отладки
        # print('bondsFinAM_row:', bondsFinAM_row) # для отладки

        xPathBond = xPath + f'/table/tbody/tr/td[1]/table/tbody/tr[2]/td/table/tbody/tr[{trCounterOut}]/td[2]/a'
        # print('xPathBond:', xPathBond) # для отладки

        bondsFinAM.loc[bondsFinAM_row, 'Эмитент'] = source['Эмитент'][sourceRow]
        if 'RatingS' in source.columns: # когда source -- это датафрейм с эмитентами
            # print("source['Issuer D Rating'][sourceRow]:", source['Issuer D Rating'][sourceRow]) # для отладки
            bondsFinAM.loc[bondsFinAM_row, 'Issuer D Rating'] = source['Issuer D Rating'][sourceRow]

        goS, xPathToNextPage = forSelenium.tryerSleeper(attemptsMax, None, driver, pause, [xPathBond, None])
        if goS == False:
            print('Следует проверить xPath вручную')
            return bondsFinAM, bondsFinAM_row, goS

        secName_finam = driver.find_element(By.XPATH, xPathBond).text
        bondsFinAM.loc[bondsFinAM_row, 'SecName FinAM'] = secName_finam
        print('\n  SecName FinAM:', secName_finam)

        bondsFinAM.loc[bondsFinAM_row, 'URL FinAM'] = driver.find_element(By.XPATH, xPathBond).get_attribute('href').replace('/default.asp', '00002')
        # display('bondsFinAM:', bondsFinAM) # для отладки

        bondsFinAM = getFeaturesByURL_FinAM(attemptsMax, bondsFinAM, bondsFinAM_row, columnS_target, driver, pause)
    
        isin = bondsFinAM.loc[bondsFinAM_row, columnS_target[0]]
        date_call, table_FinAM = getTableByURL_FinAM(driver, isin, pause)
        if len(table_FinAM) > 0:
            if os.path.exists(folder + 'Таблицы FinAM') != True: os.makedirs(folder + 'Таблицы FinAM')
            table_FinAM.to_excel(folder + 'Таблицы FinAM' + slash + f'{isin}{'' + date_call if date_call else ''}.xlsx')

        # table_FinAM = pandas.read_excel(folder + 'Таблицы FinAM' + slash + f'{isin} {date_call}.xlsx', header=[0, 1], index_col=0)
            # заготовка

        bondsFinAM_row += 1 # у некоторых облигаций без ISIN не будут заполнены и поля из описания платежей;
            # такие облигации нужны в базе, чтобы повторно не обращаться к ним

        time.sleep(pause) # для замедления перехода между page

        # Обёртка для driver.get() , чтобы не потерять промежуточные результаты
        for attempt in range(3):
            # print('attempt:', attempt) # для отладки

            try:
                driver.get(urlInitial)

                # Архитектура: /html/body/div[2]/div[1]/div[1]/a/div[1]
                WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
                    (By.XPATH, f"//div[@data-id='logo-sign']")
                    )) # проверка, что страница не пустая, а хотя бы с логотипом

                break # выход из цикла for attempt in range(3)

            except Exception as excptn:
                print('attempt:', attempt) # для отладки

                print('Exception 1:', excptn)
                print(traceback.format_exc()) # показ точной строчки кода с ошибкой

                # Закрыть или обнулить драйвер
                forSelenium.driverCloser(driver)

                # Воссоздать драйвер и подготовить для следующей итерации цикла for attempt in range(3) или обращения вне цика
                driver = forSelenium.driverCreator(version_main, headless=False, use_subprocess=True)
                driver.set_page_load_timeout((1 + attempt) * 100 * pause)

                # Если число попыток истекло, а результат так и не достигнут
                if attempt == 2:
                    print('Число попыток истекло, а результат так и не достигнут')
                    goS = False
                    forSelenium.driverCloser(driver)
                    return bondsFinAM, bondsFinAM_row, goS

        pageSource = driver.page_source
        if 'Я согласен' in pageSource: driver.find_element(By.XPATH, "//button[text()='Я согласен']").click()

    forSelenium.driverCloser(driver)
    return bondsFinAM, bondsFinAM_row, goS

def finamParser(attemptsMax,
                bondsFinAM,
                bondS_FinAM_RB, # для проверки, есть ли облигация с некоторым SecName FinAM уже в bondS_FinAM_RB
                conumnName,
                momentCurrent,
                pause,
                source, # список ISIN или эмитентов
                version_main):

    columnS_target = ['ISIN код:', 'Рег. номер:', 'Описание купонов']

    bondsFinAM = bondsFinAM.rename(columns={
        'ISIN': columnS_target[0],
        'REGNUMBER FinAM': columnS_target[1],
        'Описание платежей': columnS_target[2]
        })

    bondsFinAM_row = 0 # далее bondsFinAM_row увеличивается на 1 при каждом исполнении функции bondsOfIdentifierProcessor
    bondsOfIdentifier_Excluded = pandas.DataFrame()
    driver = forSelenium.driverCreator(version_main, headless=False, use_subprocess=True)

    for counter in range(len(source)): # counter совпадает с длиной датафрейма bondsFinAM , если source -- список ISIN , не не совпадает, если source -- список эмитентов
        sourceRow = source.index[counter]
        identifier = source[conumnName][sourceRow]

        page = 0 # каждая страница выдачи поиска FinAM содержит не более 30 облигаций

        while True: # проход по page , для каждой page своя таблица bondsOfIdentifier

            # Ввод названия эмитента
            if conumnName == 'Эмитент':
                urlInitial = 'https://bonds.finam.ru/issue/search/default.asp?page=' + str(page) + '&status=4&srchString=' + quote(identifier.encode('windows-1251'))

            if conumnName == 'ISIN':
                urlInitial = 'https://bonds.finam.ru/issue/search/default.asp?emitterCustomName=' + identifier

            # print('urlInitial:', urlInitial) # для отладки

            # Обёртка для driver.get() , чтобы не потерять промежуточные результаты
            for attempt in range(3):
                # print('attempt:', attempt) # для отладки

                try:
                    driver.get(urlInitial)

                    # Архитектура: /html/body/div[2]/div[1]/div[1]/a/div[1]
                    WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
                        (By.XPATH, f"//div[@data-id='logo-sign']")
                        )) # проверка, что страница не пустая, а хотя бы с логотипом

                    break # выход из цикла for attempt in range(3)

                except Exception as excptn:
                    print('attempt:', attempt) # для отладки

                    print('Exception 1:', excptn)
                    print(traceback.format_exc()) # показ точной строчки кода с ошибкой

                    # Закрыть или обнулить драйвер
                    forSelenium.driverCloser(driver)

                    # Воссоздать драйвер и подготовить для следующей итерации цикла for attempt in range(3) или обращения вне цика
                    driver = forSelenium.driverCreator(version_main, headless=False, use_subprocess=True)
                    driver.set_page_load_timeout((1 + attempt) * 100 * pause)

                    # Если число попыток истекло, а результат так и не достигнут
                    if attempt == 2:
                        print('Число попыток истекло, а результат так и не достигнут')
                        forSelenium.driverCloser(driver)
                        print('return 1') # для отладки
                        return bondsFinAM, counter

            # Предупреждение про Cookie закрыть
            # Архитектура: /html/body/div[3]/button
            pageSource = driver.page_source
            if 'Я согласен' in pageSource: driver.find_element(By.XPATH, "//button[text()='Я согласен']").click()

            if 'По Вашему запросу ничего не найдено' in driver.find_element("tag name", "body").text:
                break # выход из цикла while True ; по запросу для эмитента ничего не найдено => сразу break

            else:
                print('\n\nИдентификатор:', source[conumnName][sourceRow]) # для отладки
                if conumnName == 'Эмитент': print('page:', page)

                textAnchor = 'выпусков:'
                # print('textAnchor:', textAnchor) # для отладки

                # Найти фрагмент с текстом
                # Архитектура
                # /html/body/div[2]/div[3]/div/table/tbody/tr/td[1]/table/tbody/tr/td[1]/table/tbody/tr[1]/td[1]/table/tbody/tr/td/table/tbody/tr/td[5]
                try:
                    # Найти фрагмент с текстом  
                    elementAnchor =  WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
                        (By.XPATH, f"//td[contains(., '{textAnchor}')]")
                        ))

                    # Найти ближайшую родительскую таблицу
                    table = elementAnchor.find_element(By.XPATH, "./ancestor::table")
                    print("✅ Нашёл нужную таблицу!")

                except Exception:
                    print(f"❌ Ошибка: {Exception}")

                    # Диагностика
                    print("Проверяю наличие фразы на странице...")
                    if textAnchor in driver.page_source:
                        print("Текст есть в коде страницы, но не найден через XPath")
                    else:
                        print("Текста нет в коде страницы")

                # Получить HTML и распарсить
                table_html = table.get_attribute('outerHTML')
                bondsOfIdentifier = pandas.read_html(table_html)[0]
                # display('bondsOfIdentifier:', bondsOfIdentifier) # для отладки

                sectionText = '№'
                textClosing = 'Простой поиск'
                column = getAssets.columnFinder(bondsOfIdentifier, sectionText)
                if column != None:
                    boundLarger, boundSmaller = getAssets.sectionFinder(bondsOfIdentifier, False, sectionText, textClosing)
                    bondsOfIdentifier = bondsOfIdentifier.loc[boundSmaller:boundLarger, :]
                    bondsOfIdentifier.columns = bondsOfIdentifier.loc[boundSmaller, :]
                    bondsOfIdentifier = bondsOfIdentifier[bondsOfIdentifier[sectionText] != textClosing]
                    bondsOfIdentifier = bondsOfIdentifier[bondsOfIdentifier[sectionText].notna()]
                    bondsOfIdentifier = bondsOfIdentifier.drop(boundSmaller)
                    bondsOfIdentifier.index = bondsOfIdentifier[sectionText]
                    bondsOfIdentifier = bondsOfIdentifier.drop(sectionText, axis=1)

                    # Чтобы на каждой page номера строк начинались с 1
                    bondsOfIdentifierIndexNew = [str(i) for i in range(1, len(bondsOfIdentifier))]
                    bondsOfIdentifierIndexNew.append(bondsOfIdentifier.index[-1] if 'Страница: ' in bondsOfIdentifier.index[-1] else str(len(bondsOfIdentifier)))
                    bondsOfIdentifier.index = bondsOfIdentifierIndexNew

                # display('bondsOfIdentifier со всеми облигациями на странице выдачи:', bondsOfIdentifier) # для отладки

                bondsOfIdentifierInitialLenth = len(bondsOfIdentifier)

                if conumnName == 'Эмитент':

                    # Если облигация с некоторым SecName FinAM уже есть в bondS_FinAM_RB
                    bondsOfIdentifier = bondsOfIdentifier[bondsOfIdentifier['Выпуск'].isin(bondS_FinAM_RB['SecName FinAM'].tolist()) != True]

                    # Название эмитента должно быть в SecName облигации
                    bondsOfIdentifier_Excluded_Additional = bondsOfIdentifier[bondsOfIdentifier['Выпуск'].apply(textPreprocessor.simbolsCleaner).str.contains(identifier, case=False) == False]
                    if len(bondsOfIdentifier_Excluded_Additional) > 0:
                        if 'Страница: ' in bondsOfIdentifier.index[-1]: bondsOfIdentifier_Excluded_Additional = pandas.DataFrame()
                        else: bondsOfIdentifier_Excluded_Additional['Эмитент'] = identifier

                    else: bondsOfIdentifier_Excluded_Additional = pandas.DataFrame()

                    display('bondsOfIdentifier_Excluded_Additional:', bondsOfIdentifier_Excluded_Additional) # для отладки

                    if len(bondsOfIdentifier) > 0:
                        if 'Страница: ' in bondsOfIdentifier.index[-1]: # придётся обработать верхние строчки таблицы отдельно от нижней
                            bondsOfIdentifier_rowsUpper = bondsOfIdentifier.iloc[:-1, :]
                            bondsOfIdentifier_rowsUpper =\
                                bondsOfIdentifier_rowsUpper[bondsOfIdentifier_rowsUpper['Выпуск'].apply(textPreprocessor.simbolsCleaner).str.contains(identifier, case=False)]

                            bondsOfIdentifier_rowLower = bondsOfIdentifier.drop(bondsOfIdentifier.index[:-1])
                            bondsOfIdentifier = pandas.concat([bondsOfIdentifier_rowsUpper, bondsOfIdentifier_rowLower])

                        else: # вся таблица обрабатывается целиком
                            bondsOfIdentifier = bondsOfIdentifier[bondsOfIdentifier['Выпуск'].apply(textPreprocessor.simbolsCleaner).str.contains(identifier, case=False)]

                display('bondsOfIdentifier с облигациями, не выгруженными ранее:', bondsOfIdentifier) # для отладки

                if len(bondsOfIdentifier) == 0:
                    print('Таблица пуста и в ней нет указания переходить на следующую страницу')
                    break # выход из цикла while True

                elif len(bondsOfIdentifier) == 1:
                    print('Таблица имеет одну строку; она указывает на переход? Или содержательная?')
                    if 'Страница: ' in bondsOfIdentifier.index[-1]:
                        print('  Есть указание на переход, но оно может быть ложным')
                        if bondsOfIdentifierInitialLenth > 1: # исходная таблица имела содержательные строки, поэтому следует попробовать перейти на следующую страницу
                            page += 1
                            print('Перехожу на page', page)
                        else: break # выход из цикла while True ; исходная таблица НЕ имела содержательные строки, поэтому НЕ следует попробовать переходить на следующую страницу
                    else:
                        print('  Нет указания на переход')
                        bondsFinAM, bondsFinAM_row, goS = bondsOfIdentifierProcessor(attemptsMax,
                                                                                     bondsFinAM,
                                                                                     bondsFinAM_row,
                                                                                     bondsOfIdentifier,
                                                                                     columnS_target,
                                                                                     driver,
                                                                                     folder,
                                                                                     pause,
                                                                                     slash, 
                                                                                     source,
                                                                                     sourceRow,
                                                                                     urlInitial,
                                                                                     version_main)

                        if goS != True:
                            forSelenium.driverCloser(driver)
                            print('return 2') # для отладки
                            return bondsFinAM, counter

                        display('bondsFinAM 1:', bondsFinAM.tail()) # для отладки
                        break # выход из цикла while True ; НЕ указывает на переход; тут перед break можно вставить функцию выгрузки данных с циклом for
                else:
                    print('Таблица имеет более одной строки')
                    if 'Страница: ' in bondsOfIdentifier.index[-1]:
                        print('  Эта таблица точно не последняя, поскольку есть указание переходить на следующую страницу и она имеет и содержательные строки')
                        bondsFinAM, bondsFinAM_row, goS = bondsOfIdentifierProcessor(attemptsMax,
                                                                                     bondsFinAM,
                                                                                     bondsFinAM_row,
                                                                                     bondsOfIdentifier,
                                                                                     columnS_target,
                                                                                     driver,
                                                                                     folder,
                                                                                     pause,
                                                                                     slash, 
                                                                                     source,
                                                                                     sourceRow,
                                                                                     urlInitial,
                                                                                     version_main)

                        if goS != True:
                            forSelenium.driverCloser(driver)
                            print('return 3') # для отладки
                            return bondsFinAM, counter

                        display('bondsFinAM 2:', bondsFinAM.tail()) # для отладки
                        page += 1
                        print('Перехожу на page', page)

                    else:
                        print('  Нет указания на переход')
                        bondsFinAM, bondsFinAM_row, goS = bondsOfIdentifierProcessor(attemptsMax,
                                                                                     bondsFinAM,
                                                                                     bondsFinAM_row,
                                                                                     bondsOfIdentifier,
                                                                                     columnS_target,
                                                                                     driver,
                                                                                     folder,
                                                                                     pause,
                                                                                     slash, 
                                                                                     source,
                                                                                     sourceRow,
                                                                                     urlInitial,
                                                                                     version_main)

                        if goS != True:
                            forSelenium.driverCloser(driver)
                            print('return 4') # для отладки
                            return bondsFinAM, counter

                        display('bondsFinAM 3:', bondsFinAM.tail()) # для отладки
                        break  # выход из цикла while True ; эта таблица точно последняя; тут перед break можно вставить функцию выгрузки данных с циклом for

            time.sleep(pause) # для замедления перехода между page

            if conumnName == 'Эмитент':
                bondsOfIdentifier_Excluded = pandas.concat([bondsOfIdentifier_Excluded, bondsOfIdentifier_Excluded_Additional])

        bondsFinAM = bondsFinAM.drop_duplicates('SecName FinAM', keep='first')
            # потому что в "Словаре эмитентов" более общие наименования одного и того же эмитента находятся в более правых столбцах
                # и, как следствие, при объединении столбцов в один -- попадают в более нижние ячейки

        bondsFinAM_columns = ['Эмитент',
                              'ISIN', 
                              'Амортизация FinAm',
                              'Описание платежей',
                              'Спред',
                              'URL FinAM',
                              'SecName FinAM',
                              'REGNUMBER FinAM']

        if 'Issuer D Rating' in bondsFinAM.columns: bondsFinAM_columns.append('Issuer D Rating')

        # print('columnS_target:', columnS_target) # для отладки

        bondsFinAM.rename(columns={
            columnS_target[0]: 'ISIN',
            columnS_target[1]: 'REGNUMBER FinAM',
            columnS_target[2]: 'Описание платежей'
            })[bondsFinAM_columns].to_excel(
            folder + 'Замеры рейтингов' + slash + momentCurrent.strftime("%Y%m%d_%H%M") + '_bondsFinAM.xlsx', index=False
                )

        if conumnName == 'ISIN': time.sleep(pause) # для замедления перехода между эмитентами или ISIN; почему-то именно во втором случае сервер банит

    display('bondsOfIdentifier_Excluded:', bondsOfIdentifier_Excluded) # для отладки
    forSelenium.driverCloser(driver)
    print('return 5') # для отладки
    return bondsFinAM.rename(columns={
        columnS_target[0]: 'ISIN',
        columnS_target[1]: 'REGNUMBER FinAM',
        columnS_target[2]: 'Описание платежей'
        }), counter

def getFeaturesByURL_FinAM(attemptsMax, bondsFinAM_in, bondsFinAM_row, columnS_target, driver, pause):
    bondsFinAM = bondsFinAM_in.copy()

    driver.set_page_load_timeout(100) # включить ограниченный таймаут загрузки

    # Выбор правильного URL FinAM
    for attempt in range(1):
        # print('attempt:', attempt) # для отладки

        driver.get(bondsFinAM['URL FinAM'][bondsFinAM_row])

        pageSource = driver.page_source

        # Предупреждение про Cookie закрыть
        # Архитектура: /html/body/div[3]/button
        if 'Я согласен' in pageSource: driver.find_element(By.XPATH, "//button[text()='Я согласен']").click()

        # Проверка, что страница не содержит error404-banner -- это случается, когда по облигации не предусмотрены платежи
        # Архитектура: /html/body/div[2]/div/table/tbody/tr/td/div[2]/div/div[1]
        if 'error404-banner_text' in pageSource:
            bondsFinAM.loc[bondsFinAM_row, 'URL FinAM'] = bondsFinAM.loc[bondsFinAM_row, 'URL FinAM'].replace('00002', '')
            print('  По облигации не предусмотрены платежи, поэтому потребовалась корректировка URL FinAM')

        else: break # выход из цикла for attempt in range(1)

    for textTarget in columnS_target:
        # print('textTarget:', textTarget) # для отладки
        if textTarget in driver.find_element("tag name", "body").text:
            if textTarget == 'Описание купонов':
                attempt = 0
                while attempt < attemptsMax:
                    try:
                        textFetched = forSelenium.pathRelative(driver, 
                                                               f"//b[contains(., '{textTarget}')]",
                                                               "./following-sibling::table",
                                                               1,
                                                               None,
                                                               textTarget).text
                        break # выход из цикла while attempt < attemptsMax

                    except selenium.common.exceptions.StaleElementReferenceException:
                        time.sleep(pause)
                        attempt += 1

                textFetched = textPreprocessor.multispaceCleaner(textFetched.replace(textTarget, '').replace('\n', ' '))
                bondsFinAM.loc[bondsFinAM_row, 'Амортизация FinAm'] = 1 if ('погаш' in textFetched.lower()) & ('част' in textFetched.lower()) else 0

                spread = spreadExtract(textFetched)
                # spread = re.findall(r'Преми.+|Спред.+|RUONIA.+', textFetched, re.IGNORECASE)
                # print('spread:', spread) # для отладки
                # if spread != []:
                #     spread = spread[0]
                #     spread = re.findall(r'\d+,?\d*', spread)
                #     if spread != []: bondsFinAM.loc[bondsFinAM_row, 'Спред'] = float(spread[0].replace(',', '.'))
                #     else: bondsFinAM.loc[bondsFinAM_row, 'Спред'] = 0

                # else: bondsFinAM.loc[bondsFinAM_row, 'Спред'] = 0

            else:
                attempt = 0
                while attempt < attemptsMax:
                    try:
                        textFetched = WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
                            (By.XPATH, f"//tbody//td//tbody//td[contains(., '{textTarget}')]")
                            )).text
                        break # выход из цикла while attempt < attemptsMax

                    except selenium.common.exceptions.StaleElementReferenceException:
                        time.sleep(pause)
                        attempt += 1

                textFetched = textPreprocessor.multispaceCleaner(textFetched.replace(textTarget, '').replace('\n', ' '))

            textFetched = textFetched.strip()
            # print('textFetched:', textFetched) # для отладки
            bondsFinAM.loc[bondsFinAM_row, textTarget] = textFetched

        else:
            print(f"  Параметр '{textTarget}' не отображён для облигации по ссылке {bondsFinAM['URL FinAM'][bondsFinAM_row]}")

    return bondsFinAM

def getTableByURL_FinAM(driver, isin, pause):
    # Подождать загрузку таблицы (любой элемент с классом "light")
    table_element = WebDriverWait(driver, pause).until(
        expected_conditions.presence_of_element_located((By.CLASS_NAME, "light"))
        )

    # Получить HTML таблицы
    table_html = table_element.get_attribute('outerHTML')

    # Парсить посредством pandas с отключением преобразования чисел
    tableS_table_FinAM = pandas.read_html(StringIO(table_html), decimal=',', thousands=None)

    if tableS_table_FinAM:

        table_FinAM = tableS_table_FinAM[0]

        if len(table_FinAM) > 0:

            # Удалить строки с описанием купонов
            table_FinAM = table_FinAM[~table_FinAM.apply(lambda row: row.astype(str).str.contains('Описание купонов').any(), axis=1)]

            table_FinAM.columns = pandas.MultiIndex.from_tuples(
                [(col[0], col[1].replace('\xa0', ' ')) for col in table_FinAM.columns]
                )

            table_FinAM_columnS = table_FinAM.columns

            for table_FinAM_column in table_FinAM_columnS:
                if table_FinAM[table_FinAM_column].dtype == 'object':  # Только строковые столбцы
                    table_FinAM[table_FinAM_column] = table_FinAM[table_FinAM_column].str.replace('RUR', '', regex=False).str.strip()
                    table_FinAM[table_FinAM_column] = table_FinAM[table_FinAM_column].str.replace(',', '.')
                    table_FinAM[table_FinAM_column] = pandas.to_numeric(table_FinAM[table_FinAM_column], errors='ignore')

            # display('table_FinAM 1:', table_FinAM) # для отладки

            for counter in range(len(table_FinAM_columnS)):
                # print(table_FinAM_columnS[counter]) # для отладки
                if table_FinAM_columnS[counter] == (          'Погашение',        'Размер (ден)'):
                    # print('Столбец найден') # для отладки
                    break

            # print('counter:', counter) # для отладки

            table_FinAM = table_FinAM[table_FinAM_columnS[:counter + 1]]
            # display('table_FinAM 2:', table_FinAM) # для отладки

            table_FinAM = table_FinAM.iloc[:table_FinAM[table_FinAM.isna().all(axis=1)].index[0], :]
            # display('table_FinAM 3:', table_FinAM) # для отладки

            table_FinAM[(   'Купоны',        'Ставка')] = table_FinAM[(   'Купоны',        'Ставка')].str.replace('%', '').astype(float)

            # Преобразовать столбец "Дата" в формат datetime
            table_FinAM[(             'Купоны',                'Дата')] =\
                pandas.to_datetime(table_FinAM[(             'Купоны',                'Дата')], format='%d.%m.%Y')

            date_call = table_FinAM.loc[
                table_FinAM[table_FinAM[(   'Купоны',        'Ставка')].notna()].index[-1],
                (   'Купоны',          'Дата')
                ].date().strftime("%Y%m%d")

            # print('date_call:', date_call) # для отладки

        else: # заглушка, если не if len(table_FinAM) > 0
            date_call = None
            table_FinAM = pandas.DataFrame()

    else: # заглушка, если не if tableS_table_FinAM
        date_call = None
        table_FinAM = pandas.DataFrame()

    return date_call, table_FinAM

# .. извлечения значения спреда из текста описания платежей
def spreadExtract(textFetched):

    if not textFetched or not isinstance(textFetched, str):
        return 0
    
    text = textFetched
    
    # Исключить ложные срабатывания
    if re.search(r'ставка\s+[\d,]+\s*%', text, re.IGNORECASE) and not re.search(r'(спред|преми|надбавк|margin|б\.п\.|процентных\s+пункт)', text, re.IGNORECASE):
        return 0
    
    patterns = [
        # 1. Базисные пункты: 100 б.п. = 1.00%
        (r'(\d+)\s+б\.п\.', 'bp'),
        
        # 2. Процентные пункты: 1,30 п.п. = 1.30%
        (r'(\d+[,.]?\d*)\s+процентных\s+пункт[а-я]*', 'pp'),
        
        # 3. Процентные пункты (сокращение)
        (r'(\d+[,.]?\d*)\s+п\.п\.', 'pp'),
        
        # 4. RUONIA + X,XX%
        (r'RUONIA\s*\+\s*(\d+[,.]?\d*)\s*%', 'percent'),
        
        # 5. RUONIA + X,XX (без %)
        (r'RUONIA\s*\+\s*(\d+[,.]?\d*)', 'percent'),
        
        # 6. Ключевая ставка + X,XX%
        (r'ключев[а-я]+\s+ставк[а-я]+\s*\+\s*(\d+[,.]?\d*)\s*%', 'percent'),
        
        # 7. Ключевая ставка + X,XX (без %)
        (r'ключев[а-я]+\s+ставк[а-я]+\s*\+\s*(\d+[,.]?\d*)', 'percent'),
        
        # 8. Премия X,XX%
        (r'преми[яю]\s+(\d+[,.]?\d*)\s*%', 'percent'),
        
        # 9. Исходный паттерн
        (r'(Преми.+|Спред.+|Надбавк.+|Margin.+)', 'fragment'),
    ]
    
    for pattern, p_type in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            match = matches[0]
            
            if isinstance(match, str):
                numbers = re.findall(r'(\d+[,.]?\d*)', match)
                if numbers:
                    value_str = numbers[0].replace(',', '.')
                    try:
                        value = float(value_str)
                        if 'месяц' in match.lower() and value == 3:
                            return 0
                        
                        # Базисные пункты: 100 б.п. = 1.00%
                        if p_type == 'bp' or 'б.п.' in match.lower():
                            return value / 100
                        
                        # Процентные пункты: 1,30 п.п. = 1.30% (НЕ ДЕЛИМ!)
                        if p_type == 'pp' or 'процентных пункт' in match.lower():
                            return value
                        
                        return value
                    except ValueError:
                        continue
            else:
                value_str = str(match).replace(',', '.')
                try:
                    value = float(value_str)
                    
                    # Базисные пункты: 100 б.п. = 1.00%
                    if p_type == 'bp':
                        return value / 100
                    
                    # Процентные пункты: 1,30 п.п. = 1.30% (НЕ ДЕЛИМ!)
                    if p_type == 'pp':
                        return value
                    
                    return value
                except ValueError:
                    continue
    
    return 0
