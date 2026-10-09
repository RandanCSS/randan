#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and manages the web scraping workflow
(RU) Модуль для упрощения скрапинга
'''

# 0. Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# 0.0 В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call, CalledProcessError

# --- остальные модули и пакеты
MAX_ATTEMPTS = 3
attempt = 1

while True:
    try:
        from randan.tools import textPreprocessor # модуль для предобработки нестандартизированного текста
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

def argument_key_comparison(argument, key, params):
    if (key in params.keys()) & (argument != None):
        if params[key] != argument:
            print(f'!!   Вы подали {key} и как аргумент, и через словарь params , причём Вы подали разные значения туда и туда; будет использовано значение, поданное в params !!\n')
            argument = params[key]
    elif (key in params.keys()) & (argument == None): argument = params[key]
    elif (key not in params.keys()) & (argument != None): pass # отдельный аргумент определён, поэтому запрос к пользователю не поступит
    else: pass # НИ ключ params , НИ отдельный аргумент НЕ определены, поэтому запрос к пользователю поступит
    return argument

def containerExport(folderFile, container):
    file = open(folderFile, 'w+') # открыть на запись
    file.write(container)
    file.close()

def containerImport(folderFile, containerType):
    # file = open(f'{rootName}{slash}{containerName}.txt')
    file = open(folderFile)
    container = file.read()
    file.close()
    if container:
        if type(container) != containerType: container = containerType(container) # мало ли какой тип окажется при импорте

    return container

# .. работы с ключами
def credentialsProcessor(access_token, api_name, instruction_url, nameS_in_folderCurrent):
    if not access_token:
        if f'credentials{api_name}.txt' in nameS_in_folderCurrent:
            file = open(f'credentials{api_name}.txt')
            api_keyS = file.read().strip()
            file.close()
            print(f"Проверяю наличие файла credentials{api_name}.txt с ключ{'ами' if ',' in api_keyS else 'ом'}, гипотетически сохранённым{'и' if ',' in api_keyS else ''} при первом запуске скрипта")
            print(f"Нашёл файл credentials{api_name}.txt; далее буду использовать ключ{'и' if ',' in api_keyS else ''} из него:", api_keyS)

        else:
            print(
f'''--- НЕ нашёл файл credentials{api_name}.txt . Введите в окно Ваш API key для авторизации в API {api_name} 
(примерная инструкция, как создать API key, доступна по ссылке {instruction_url} ). Для подстраховки от ограничения действия API key желательно создать несколько ключей (три -- отлично) и ввести их без кавычек через запятую с пробелом
--- После ввода нажмите Enter'''
            )

            while True:
                api_keyS = input()
                if len(api_keyS) > 0:
                    print(f"-- далее буд{'у' if ',' in api_keyS else 'е'}т использован{'ы' if ',' in api_keyS else ''} эт{'и' if ',' in api_keyS else 'от'} ключ{'и' if ',' in api_keyS else ''}")

                    api_keyS = textPreprocessor.multispaceCleaner(api_keyS)
                    if len(api_keyS) > 0:
                        while api_keyS[-1] == ',': api_keyS = api_keyS[:-1] # избавиться от запятых в конце текста

                    containerExport(f'credentials{api_name}.txt', api_keyS)
                    break

                else:
                    print('--- Вы ничего НЕ ввели. Попробуйте ещё раз..')
        api_keyS = api_keyS.replace(' ', '').replace(',', ', ').split(', ')

    else: api_keyS = [access_token]
    print('Количество ключей:', len(api_keyS), '\n')
    return api_keyS
