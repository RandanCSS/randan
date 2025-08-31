#!/usr/bin/env python
# coding: utf-8

'''
A module for facilitating the use of selenium
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
        from pandas import DataFrame
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
            
def blockSearch(attemptsMax, text, xPathS):
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
