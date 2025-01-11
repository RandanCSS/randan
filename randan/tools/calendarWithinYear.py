#!/usr/bin/env python
# coding: utf-8

'''
A module for working on a calendar within a specific year
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from pandas import DataFrame
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        print('Пакет', module, 'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

def calendarWithinYear(year):
    # Календарь день-месяц для лучшей визуализации дальнейшего процесса
    calendar = DataFrame()
    mnth_n31 = [2, 4, 6, 9, 11] # месяцы с числом дней менее 31
    for mnth in range(1, 13):
        if mnth == 2: # обработка февраля
            for day in range(1, 29):
                calendar.loc[str(day) if len(str(day)) == 2 else '0' + str(day), '02'] = 1
            if  year % 4 == 0: # обработка високосных лет
                calendar.loc['29', '02'] = 1
        elif (mnth in mnth_n31) & (mnth != 2): # месяцы с числом дней менее 31, но не февраль
            for day in range(1, 31):
                calendar.loc[str(day) if len(str(day)) == 2 else '0' + str(day)
                             , str(mnth) if len(str(mnth)) == 2 else '0' + str(mnth)] = 1
        else: # месяцы с числом дней 31
            for day in range(1, 32):
                calendar.loc[str(day) if len(str(day)) == 2 else '0' + str(day)
                             , str(mnth) if len(str(mnth)) == 2 else '0' + str(mnth)] = 1
    return calendar
