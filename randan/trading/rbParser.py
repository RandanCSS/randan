#!/usr/bin/env python
# coding: utf-8

'''
A module designed to facilitate the scraping and parsing of data from the rusnonds.ru website
Модуль для упрощения выгрузки данных с сайта rusnonds.ru и их парсинга
'''

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from randan.tools import forSelenium # авторский модуль для
            # (а) упрощения некоторых оперций в selenium

        from selenium.webdriver.common.by import By
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

        check_call([sys.executable, '-m', 'pip', 'install', module])
        if  attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )

# Функции для..
# .. выгрузки с RB характеристик облигаций с ISIN или с SecName или REGNUMBER
def isin_restoration(bondS_FinAM_RB, bondsRB, column_target_FinAM, columnS_target_RB, driver, errorS, folder, momentCurrent, pause):

    if 'ISIN код' not in bondsRB.columns: bondsRB = bondsRB.rename(columns={'ISIN': 'ISIN код'})

    # for counter in range(len(errorS[:1])): # для отладки
    for counter in range(len(errorS)):
        errorS_row = errorS[counter]
        print('\nerrorS_row:', errorS_row) # для отладки

        resultS_container_previous = None

        # На всякий случай, очистить окно поиска
        # Архитектура: /html/body/div[1]/div/div/div/header/section/div[1]/div/div[4]/div/form/svg
        try:
            driver.find_element(By.CSS_SELECTOR, "svg.fa-xmark").click()
            print('Очистка svg.fa-xmark') # для отладки
            time.sleep(pause)

        except Exception as excptn:
            # print('Exception 1:', excptn)
            # print(traceback.format_exc()) # показ точной строчки кода с ошибкой

            form_input = driver.find_element(By.XPATH, "//form//input[@class='input']")
            form_input.click()
            time.sleep(pause)
            form_input.clear() # очистить окно поиска
            print('Очистка clear()') # для отладки

        identifier_finam = bondS_FinAM_RB[column_target_FinAM][errorS_row]
        print('identifier_finam:', identifier_finam) # для отладки

        # Архитектура: /html/body/div[1]/div/div/div/header/section/div[1]/div/div[4]/div/form/input
        driver.find_element(By.XPATH, "//form//input[@class='input']").click() # клик на окно поиска
        time.sleep(pause)
        driver.find_element(By.XPATH, "//form//input[@class='input']").send_keys(identifier_finam) # ввести в него SecName облигации

        try: # либо "лупа"
            # Архитектура: /html/body/div[1]/div/div/div/header/section/div[1]/div/div[4]/div/form/svg
            driver.find_element(By.CSS_SELECTOR, "svg.fa-magnifying-glass").click() # клик на поиск по введённому SecName облигации
            time.sleep(pause)

        except selenium.common.exceptions.ElementNotInteractableException: # либо ничего
            pass  

        tryer = 1
        while tryer < 20: # чтобы скрипт на скорости не прорускал клик (цикл while tryer)
            try: # чтобы скрипт на скорости не прорускал клик
                print('tryer:', tryer, '     ') # для отладки , end='\r'

                # Блоки с результатами поиска
                # Архитектура: /html/body/div[1]/div/div/div/header/section/div[1]/div/div[4]/div/form/div/div[2]
                resultS_container = driver.find_elements(By.XPATH, "//div[@class='results-container']")

                # Полная загрузка страницы означает, что в блоках result_container есть текст (не пустота ''), в котором и можно проверить 
                resultS_container_text = ''
                for result_container in resultS_container:
                    resultS_container_text += result_container.text
                    # print('resultS_container_text:', resultS_container_text) # для отладки

                if len(resultS_container_text) > 0: # полная загрузка страницы состоялась
                    if 'Не найдено' in resultS_container_text: # и в блоках result_container есть 'Не найдено'
                        print('Облигация с искомым', column_target_FinAM, 'не найдена')
                        # Архитектура: /html/body/div[1]/div/div/div/header/section/div[1]/div/div[4]/div/form/svg
                        driver.find_element(By.CSS_SELECTOR, "svg.fa-xmark").click() # очистить окно поиска
                        time.sleep(pause)

                        print('break 1') # для отладки
                        break # для выхода из цикла while

                    else: # и в блоках result_container нет 'Не найдено'
                        bondS_FinAM_RB, bondsRB, result_container_notEmpty, success =\
                            resultS_container_processor(bondS_FinAM_RB,
                                                        bondsRB,
                                                        column_target_FinAM,
                                                        columnS_target_RB,
                                                        driver,
                                                        errorS_row,
                                                        folder,
                                                        momentCurrent,
                                                        resultS_container)
                        # print('resultS_container_previous:', resultS_container_previous) # для отладки

                        if success:
                            driver.get('https://rusbonds.ru')
                                # попытка избавиться от залипания окна поиска на успешном запросе

                            time.sleep(pause)

                            print('break 4') # для отладки
                            break # для выхода из цикла while

                        else:
                            if resultS_container_previous == resultS_container: 
                                print('break 5') # для отладки
                                break # для выхода из цикла while

                            else:
                                resultS_container_previous = resultS_container
                                    # для будущей проверки resultS_container_previous == resultS_container

                                # Прокрутка на высоту видимой части окна (clientHeight) и заход на новую итерацию while
                                try: driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;",
                                                           result_container_notEmpty.find_element(By.XPATH, f".//div[@class='results ps']"))
                                except selenium.common.exceptions.NoSuchElementException:\
                                    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;",
                                        result_container_notEmpty.find_element(By.XPATH, f".//div[@class='results ps ps--active-y']"))

                                print('Скролю') # для отладки

            except Exception as excptn: # чтобы скрипт на скорости не прорускал клик (цикл while tryer)
                # print('Exception:', excptn)
                # print(traceback.format_exc()) # показ точной строчки кода с ошибкой

                time.sleep(pause * tryer)
                tryer += 1

    if 'ISIN' not in bondsRB.columns: bondsRB = bondsRB.rename(columns={'ISIN код': 'ISIN'})

    return bondS_FinAM_RB, bondsRB, driver

# .. унификации написания идентификаторов эмитентов
def issuerIdentifierNormalizer(issuerIdentidier):
    issuerIdentidier = issuerIdentidier.replace('P', 'Р') # для автодора; латиница на кириллицу

    if 'автодор' in issuerIdentidier.lower():
        issuerIdentidier = re.sub(r'\s+ГК\b', '', issuerIdentidier) # для автодора

    if 'секьюритиз' in issuerIdentidier.lower():
        issuerIdentidier = issuerIdentidier.replace('Секьюритиз', 'Секьюритизация ').replace('-об', '')
            # для СФО СБ Секьюритизация

    return issuerIdentidier

# .. для авторизации на сайте rusnonds.ru
def loginerRB(attemptsMax, boundarieS, driver, pause, xPathS):
    # Вызов окна ввода логина и пароля
    # print('xPathS[0]:', xPathS[0]) # для отладки
    goS, xPath_loginPrompt = forSelenium.tryerSleeper(attemptsMax, boundarieS[0], driver, pause, xPathS[0])
    print('xPath_loginPrompt:', xPath_loginPrompt) # для отладки
    if goS == False:
        print('Следует проверить xPath вручную')
        warnings.filterwarnings("ignore")
        input()
        sys.exit()
    driver.find_element(By.XPATH, xPath_loginPrompt).click()

    # Ввод логина и пароля
    # print('xPathS[1]:', xPathS[1]) # для отладки
    if xPathS[1] == None: xPath_credentialsEntry = xPath_loginPrompt
    else:
        goS, xPath_credentialsEntry = forSelenium.tryerSleeper(attemptsMax, boundarieS[1], driver, pause, xPathS[1])
        print('xPath_credentialsEntry:', xPath_credentialsEntry) # для отладки
        if goS == False:
            print('Следует проверить xPath вручную')
            warnings.filterwarnings("ignore")
            input()
            sys.exit()
        # return xPath_credentialsEntry, xPath_credentialsEntry
    driver.find_element(By.XPATH, xPath_credentialsEntry + '/div[2]/div/div/div[1]/input').send_keys('alexey.n.rotmistrov@gmail.com')
    driver.find_element(By.XPATH, xPath_credentialsEntry + '/div[3]/div/div/div[1]').click()
    driver.find_element(By.XPATH, xPath_credentialsEntry + '/div[3]/div/div/div[1]/input').send_keys('05T05t2022')
    driver.find_element(By.XPATH, xPath_credentialsEntry + '/div[5]/button[2]/span').click()

    print(
'''--- Сейчас в браузере появится Captcha; обработайте её вручную
--- Расположите два окна: с этим скриптом и управляемое им окно браузера -- так, чтобы они оба были видны; нажмите Enter'''
          )
    
    input()
    return xPath_loginPrompt, xPath_credentialsEntry

# .. для обработки блоков resultS_container
def resultS_container_processor(bondS_FinAM_RB,
                                bondsRB,
                                column_target_FinAM,
                                columnS_target_RB,
                                driver,
                                errorS_row,
                                folder,
                                momentCurrent,
                                resultS_container):
    success = True
    for result_container in resultS_container:
        if result_container.text != '':
            # print('result_container.text:', result_container.text) # для отладки
            result_container_notEmpty = result_container

            # if column_target_FinAM != 'ISIN':

            secName_finam = bondS_FinAM_RB['SecName FinAM'][errorS_row]
            print('secName_finam:', secName_finam)

            secName_finam_pattern = re.compile(rf"\b{re.escape(secName_finam)}\b", re.IGNORECASE)
                # чтобы не спутать облигации, имеющие похожие SecName

            result_container_notEmpty_text = result_container_notEmpty.text.strip()
            result_container_notEmpty_text = issuerIdentifierNormalizer(result_container_notEmpty_text)
            print('result_container_notEmpty_text:', result_container_notEmpty_text) # для отладки

            if (secName_finam_pattern.search(result_container_notEmpty_text) is not None and column_target_FinAM != 'ISIN') or \
               (result_container_notEmpty_text != '' and column_target_FinAM == 'ISIN'):
                    # если column_target_FinAM == 'ISIN' , то не требуется проверка совпадения текстов

                print('Искомый блок result_container_notEmpty (с искомым secName_finam внутри) найден')

                # div с class='result' является дочерним относительно div class='results-container')
                resultS = result_container_notEmpty.find_elements(By.XPATH, f".//div[@class='result']")
                for result in resultS:

                    result_text = result.text.strip()
                    result_text = issuerIdentifierNormalizer(result_text)
                    print('result_text:', result_text) # для отладки

                    if (secName_finam_pattern.search(result_text) is not None and column_target_FinAM != 'ISIN') or \
                       (result_text != '' and column_target_FinAM == 'ISIN'):
                            # если column_target_FinAM == 'ISIN' , то не требуется проверка совпадения текстов
                                # поскольку для ISIN выдача однозначна (гипотеза) 

                        print('Искомый блок result (с искомым secName_finam внутри) найден')

                        # Теперь кликнуть на блок result
                        if ('В обращении' in result_text) | ('Размещается' in result_text):
                            print('Облигация в обращении') # для отладки
                            result.click()

                            bondsRB_row = len(bondsRB) if column_target_FinAM != 'ISIN' else errorS_row
                                # если column_target_FinAM != 'ISIN' , то добавляется новая запись
                                # если column_target_FinAM == 'ISIN' , то редактируется (дополняется) имеющаяся запись

                            bondsRB = getFeaturesByURL_RB(columnS_target_RB, bondsRB, driver, bondsRB_row)
                            bondsRB.loc[bondsRB_row, 'URL RB'] = driver.current_url
                            bondsRB.loc[bondsRB_row, 'Момент обращения к RB'] = momentCurrent.strftime("%Y%m%d_%H%M")

                            if 'ISIN' not in bondsRB.columns:
                                bondsRB.rename(columns={'ISIN код': 'ISIN'}).to_excel(
                                    folder + momentCurrent.strftime("%Y%m%d_%H%M") + '_bondsRB Temporal.xlsx',
                                    index=False
                                    ) # промежуточное сохранение

                            else: bondsRB.to_excel(
                                folder + momentCurrent.strftime("%Y%m%d_%H%M") + '_bondsRB Temporal.xlsx',
                                index=False
                                ) # промежуточное сохранение

                            if column_target_FinAM != 'ISIN':
                                bondS_FinAM_RB.loc[errorS_row, 'ISIN'] = bondsRB.loc[bondsRB_row, 'ISIN код']
                                    # записать искомый и найденный ISIN в bondS_FinAM_RB
                                    # если column_target_FinAM == 'ISIN' , то bondS_FinAM_RB НЕ редактируется

                                bondS_FinAM_RB.loc[errorS_row, 'Момент обращения к FinAM'] = momentCurrent.strftime("%Y%m%d_%H%M")

                                bondS_FinAM_RB.to_excel(
                                    folder + momentCurrent.strftime("%Y%m%d_%H%M") + '_bondS_FinAM_RB Temporal.xlsx',
                                    index=False
                                    ) # промежуточное сохранение

                        else:
                            print('Облигация вышла из обращения') # для отладки

                            bondsRB.loc[errorS_row, 'Не в обращении'] = 1

                            if column_target_FinAM != 'ISIN': bondS_FinAM_RB.loc[errorS_row, 'Не в обращении'] = 1
                                # если column_target_FinAM == 'ISIN' , то bondS_FinAM_RB НЕ редактируется

                        print('break 2') # для отладки
                        break # для выхода из внутреннего цикла for
                    elif result == resultS[-1]: print('Искомый блок result не найден')
                print('break 3') # для отладки
                break # для выхода из внешнего цикла for

        if result_container == resultS_container[-1]:
            print('Искомый блок result_container_notEmpty не найден')
            success = False
    return bondS_FinAM_RB, bondsRB, result_container_notEmpty, success
