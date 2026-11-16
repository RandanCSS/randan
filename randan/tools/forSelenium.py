#!/usr/bin/env python
# coding: utf-8

'''
A module for facilitating the use of selenium
Модуль для упрощения некоторых операций в selenium
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
        from selenium import webdriver
        from selenium.webdriver.common.by import By # для поиска элементов HTML-кода
        from selenium.webdriver.support import expected_conditions
        from selenium.webdriver.support.ui import WebDriverWait
        import re, selenium.common.exceptions, time, undetected_chromedriver
        break # выход из цикла while True

    except (ImportError, ModuleNotFoundError) as excptn_1:
        module = (excptn_1.name or '').split('.')[0]
        if attempt > MAX_ATTEMPTS:
            print(
f'Пакет {module} НЕ удалось импортировать за {MAX_ATTEMPTS} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт')

            raise

        print(
f'Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас. Попытка № {attempt} из {MAX_ATTEMPTS}')

        try: check_call([sys.executable, '-m', 'pip', 'install', module, '--quiet', '--disable-pip-version-check'])
        except CalledProcessError as excptn_2:
            print(f"Не удалось установить {module}. {type(excptn_2).__name__}: {str(excptn_2).split('Stacktrace:')[0].strip()}")
            raise

        attempt += 1

# 1. Функции
def blockSearch(attemptsMax, driver, text, xPathS):
    block = None 
    trCounter = 1
    while True:
        # print('xPath:', xPathS[0] + str(trCounter) + xPathS[1]) # для отладки
        try:
            block = driver.find_element(By.XPATH, xPathS[0] + str(trCounter) + xPathS[1]).text
            if text in block: break
            trCounter += 1
        except:
            # print('trCounter:', trCounter) # для отладки
            trCounter += 1
            if trCounter > attemptsMax: break # против бесконечного цикла при пустом блоке страницы
    return block if text in block else None

def driverCloser(driver):
    if driver is not None:
        try:
            driver.quit()
        except Exception as excptn: # не удалось закрыть драйвер
            print('Exception:', excptn)
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой
        driver = None # обнулить драйвер

def driverCreator(version_main, headless=False, use_subprocess=True):
    options = undetected_chromedriver.ChromeOptions()
    options.add_argument('--disable-backgrounding-occluded-windows')
    options.add_argument('--disable-background-timer-throttling')
    if headless: options.add_argument('--headless')
    return undetected_chromedriver.Chrome(options=options, use_subprocess=True, version_main=version_main)

def pathRelative(driver, pathAnchor, pathTarget, pause, textAnchor, textTarget):
    if pathAnchor == None: pathAnchor = pathTarget
    if textAnchor == None: textAnchor = textTarget

    # print('pathAnchor:', pathAnchor) # для отладки
    # print('pathTarget:', pathTarget) # для отладки
    # print('textAnchor:', textAnchor) # для отладки
    # print('textTarget:', textTarget) # для отладки

    if WebDriverWait(driver, pause).until(lambda driver: len(re.findall(textAnchor, driver.find_element(By.TAG_NAME, "body").text, re.IGNORECASE)) == 1): # проверить уникальность
        elementAnchor = WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located((By.XPATH, pathAnchor))) # найти якорь
        # print('elementAnchor:', elementAnchor) # для отладки
        elementTarget = elementAnchor if pathAnchor == pathTarget else WebDriverWait(elementAnchor, pause).until(expected_conditions.presence_of_element_located((By.XPATH, pathTarget))) # от якоря к кнопке
        # print('elementTarget:', elementTarget) # для отладки
        return elementTarget

def tryerSleeper(attemptsMax, boundarieS, driver, pause, xPathS):
    goS = True
    goC = True

    attempt = 1
    while (attempt < attemptsMax) & goC:
        if boundarieS != None:
            for i in range(boundarieS[0], boundarieS[1]): # цикл на случай вариативности дивов
                xPath = xPathS[0] + str(i) + xPathS[1]
                try:
                    driver.find_element(By.XPATH, xPath)
                    goC = False
                    break
                except selenium.common.exceptions.NoSuchElementException:
                    errorDescription = sys.exc_info()
                    print(f'Попытка tryerSleeper № {attempt} . Ошибка:', errorDescription, '          ', end='\r') # , end='\r'
                    time.sleep(pause)
                    attempt += 1
                    if attempt == attemptsMax: goS = False
        else:
            xPath = xPathS[0]
            try:
                driver.find_element(By.XPATH, xPath)
                goC = False
                break
            except selenium.common.exceptions.NoSuchElementException:
                errorDescription = sys.exc_info()
                print(f'Попытка tryerSleeper № {attempt} . Ошибка:', errorDescription, '          ', end='\r') # , end='\r')
                time.sleep(pause)
                attempt += 1
                if attempt == attemptsMax: goS = False
    print('                                                                                                                                                                                                                                                ', end='\r')
    return goS, xPath
