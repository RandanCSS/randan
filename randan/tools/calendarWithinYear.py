#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for working on a calendar within a specific year
(RU) Модуль для работы с календарём в рамках некоторого года
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
        from pandas import DataFrame
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

def calendarWithinYear(year):
    """
    Функция "Календарь день-месяц" для лучшей визуализации дальнейшего процесса

    Parameters
    ----------
    year : int
    """
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

def yearsRangeParser(yearsRange):
    yearsRange = yearsRange.split('-') # мутация
    yearsRange.sort()
    yearMinByUser = int(yearsRange[0])
    yearMaxByUser = int(yearsRange[-1])
    yearsRange = f'{yearMinByUser}-{yearMaxByUser}' # обратная мутация
    return yearMaxByUser, yearMinByUser, yearsRange
