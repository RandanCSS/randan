#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for coloring console output
(RU) Модуль для раскрашивания вывода на экран
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
        from contextlib import contextmanager
        from functools import wraps
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

def _colorize(code):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sys.stdout.write(code)
            try: return func(*args, **kwargs)
            finally: sys.stdout.write("\033[0m")
        return wrapper
    return decorator

gray_prints  = _colorize("\033[90m")
green_prints = _colorize("\033[32m")
red_prints   = _colorize("\033[31m")
