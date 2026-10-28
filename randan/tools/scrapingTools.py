#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and manages the web scraping workflow
(RU) Модуль для упрощения скрапинга
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
        from randan.tools import coLabAdaptor, files2df, textPreprocessor # модуи для
            # (а) адаптации текущего скрипта к файловой системе CoLab
            # (б) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа
            # (в) предобработки неформализованного текста

        break # выход из цикла while True

    except (ImportError, ModuleNotFoundError) as excptn_1:
        module = excptn_1.name.split('.')[0]
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

def argument_key_comparison(argument, key, params): # приоритет аргумента над ключом словаря params
    if key in params.keys(): # key подан через словарь params (в т.ч. со значением None)
        if argument is not None and argument != params[key]: # key подан И как аргумент (т.е. НЕ со значением None), при этом поданы разные значения туда и туда
            print(f'!! Вы подали {key} и как аргумент, и через словарь params , причём Вы подали разные значения туда и туда; будет использовано значение, поданное как аргумент !!')
            return argument

        else: # key НЕ подан как аргумент или его значение такое же, как через словарь params
            return params[key]

    else: # key НЕ подан через словарь params (в т.ч. со значением None)
        return argument # используется key как аргумент (в т.ч. со значением None)

def call_for_user_items(message_adjustable):
    print(
f'''
Возможно, Вы располагаете файлом, {message_adjustable}, и который хотели бы дополнить? Или планируете первичный сбор контента?
--- Если планируете первичный сбор, нажмите Enter
--- Если располагаете файлом формата XLSX, укажите полный путь, включая название файла, и нажмите Enter.
Затем при необходимости сможете добавить к нему другие располагаемые файлы''')

    while True:
        folder_file = input()
        if len(folder_file) == 0:
            folder = None # для унификации
            folder_file = None # для унификации
            break

        else:
            itemS, error, folder = files2df.files2df(folder_file)
            if error:
                if 'No such file or directory' in error: print('Путь:', folder_file, '-- не существует; попробуйте, пожалуйста, ещё раз..')
                else:
                    print('error из files2df:', error)
                    break
            else: break
        # display('itemS:', itemS)
    return folder, folder_file, itemS

def containerExport(folder_file, container):
    with open(folder_file, 'w', encoding='utf-8') as file: file.write(container)

def containerImport(folder_file, containerType): # containerType: float, int, str
    with open(folder_file, encoding='utf-8') as file: container = file.read()
    if container and not isinstance(container, containerType): container = containerType(container) # мало ли какой тип окажется при импорте
    return container

# .. работы с ключами
def credentialsProcessor(access_token, api_name, folder, instruction_url, nameS_in_folderCurrent, silentMode=False):
    folder = coLabAdaptor.folderCoLab_folderIn_comparison(folder)
    if not access_token:
        if f'credentials{api_name}.txt' in nameS_in_folderCurrent:
            with open(folder + f'credentials{api_name}.txt', encoding='utf-8') as file: api_keyS = file.read().strip()
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
                if api_keyS:
                    print(f"-- далее буд{'у' if ',' in api_keyS else 'е'}т использован{'ы' if ',' in api_keyS else ''} эт{'и' if ',' in api_keyS else 'от'} ключ{'и' if ',' in api_keyS else ''}")

                    api_keyS = textPreprocessor.multispaceCleaner(api_keyS)
                    while len(api_keyS) > 0 and api_keyS[-1] == ',': api_keyS = api_keyS[:-1] # избавиться от запятых в конце текста
                    if api_keyS: containerExport(f'credentials{api_name}.txt', api_keyS)
                    break

                else:
                    print('--- Вы ничего НЕ ввели. Попробуйте ещё раз..')

        api_keyS = [key.strip() for key in api_keyS.replace(' ', '').split(',') if key.strip()]
        # api_keyS = api_keyS.replace(' ', '').replace(',', ', ').split(', ')

    else: api_keyS = [access_token]
    if not silentMode: print('Количество ключей:', len(api_keyS), '\n')
    return api_keyS

# .. выгрузки itemS_json itemS_xlsx из temporalNameS
def get_itemsTemporal(temporalNameS):
    itemS_json = None
    itemS_xlsx = None
    for temporalName in temporalNameS:
        # print('temporalName:', temporalName) # для отладки
        if temporalName.endswith('.json') and 'VK' in temporalName:
            itemS_json = pandas.read_json(f'{folder_in_folderCurrent}{slash}{temporalName}')

        if temporalName.endswith('.xlsx') and 'VK' in temporalName:
            itemS_xlsx = pandas.read_excel(f'{folder_in_folderCurrent}{slash}{temporalName}', index_col=0)

    if itemS_json is not None and itemS_xlsx is not None:
        itemS = itemS_json.merge(itemS_xlsx, how='outer', on='id')
        folderWithTemporal = folder_in_folderCurrent
        break # выход из for folder_in_folderCurrent in nameS_in_folderCurrent

    elif itemS_json is None and itemS_xlsx is not None:
        itemS = itemS_xlsx.copy() # for temporalName in temporalNameS завершился, но файл itemS_json не импортировался
        folderWithTemporal = folder_in_folderCurrent
        break # выход из for folder_in_folderCurrent in nameS_in_folderCurrent

    # -- данные, сохранённые при прошлом запуске скрипта, загружены и цикл for name_in_folderCurrent in nameS_in_folderCurrent завершён

    elif itemS_xlsx is None and itemS_json is None:
        print(
f"СТРАННО: хотя директория '{folder_in_folderCurrent}' похожа на искомую Temporal , но в ней ни одного файла с данными, сохранёнными при прошлом запуске скрипта, не найдено; рекомендую Вам удалить её вручную, а я продолжаю поиск")
    return folderWithTemporal, itemS
