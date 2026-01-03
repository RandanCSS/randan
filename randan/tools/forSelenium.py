#!/usr/bin/env python
# coding: utf-8

'''
A proprietary module for facilitating the use of selenium
Авторский модуль для упрощения некоторых оперций в selenium
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By # для поиска элементов HTML-кода
        from selenium.webdriver.support import expected_conditions
        from selenium.webdriver.support.ui import WebDriverWait
        import re, selenium.common.exceptions, time
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        print(
f'''Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 10
'''
              )
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break
            
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
    
def pathRelative(driver, pathAnchor, pathTarget, pause, textAnchor, textTarget):
    if (pathAnchor == None) & (textAnchor == None):
        pathAnchor = pathTarget
        textAnchor = textTarget

    if len(re.findall(textAnchor, driver.find_element("tag name", "body").text, re.IGNORECASE)) == 1: # проверить уникальность
        elementAnchor = WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located((By.XPATH, pathAnchor))) # найти якорь
        # print('elementAnchor:', elementAnchor) # для отладки
        elementTarget = elementAnchor if pathAnchor == pathTarget else elementAnchor.find_element(By.XPATH, pathTarget) # от якоря к кнопке
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
