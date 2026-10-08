#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and manages the web scraping workflow of VK
(RU) Модуль для упрощения скрапинга VK
'''

# 0. Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from datetime import datetime, timezone
        from randan.tools import varPreprocessor # модуль для предобработки переменных номинального, порядкового, интервального и более высокого типа шкалы
        import numpy, pandas, traceback
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
        if attempt == 3: print(
f'''Пакет {module} НЕ удалось импортировать за {attempt} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
        )

# 1. Функции для..
# .. обработки столбцов выдачи
def dfColumnsProcessor(df_in, fields, response):
    df = df_in.copy()
    df['date'] = pandas.to_datetime(df['date'], unit='s', utc=True).dt.strftime('%Y.%m.%d')
    # df['date'] = df['date'].apply(lambda content: datetime.fromtimestamp(content, tz=timezone.utc).strftime('%Y.%m.%d'))
        # сменить формат представления дат, класс данных столбцов с id, создать столбец с кликабельными ссылками на контент;
            # здесь, а не в конце, поскольку нужна совместимость с itemS из Temporal и от пользователя

    if 'inner_type' in df.columns: # скажем, у комментариев к постам нет своего URL и в их датафрейме нет inner_type
        df['URL'] = df['from_id'].astype(str)

        mask = df['URL'].str.contains('-', na=False)
        df.loc[~mask, 'URL'] = 'id' + df.loc[~mask, 'URL']
        df.loc[mask, 'URL'] = df.loc[mask, 'URL'].str.replace('-', 'public')
        # df.loc[df[df['URL'].str.contains('-') == False].index, 'URL'] = 'id' + df.loc[df[df['URL'].str.contains('-') == False].index, 'URL']
        # df.loc[df[df['URL'].str.contains('-')].index, 'URL'] = df.loc[df[df['URL'].str.contains('-')].index, 'URL'].str.replace('-', 'public')

        df['URL'] =\
            'https://vk.com' + '/' + df['URL'] + '?w=' + df['inner_type'].str.split('_').str[0] + df['owner_id'].astype(str) + '_' + df['id'].astype(str)

    if fields is not None:
        for fieldS_column in ['groups', 'profiles']:
            if fieldS_column in response.keys():
                if response[fieldS_column]: # например, когда в основном df нет групповых или, наоборот, персональных аккаунтов,
                        # тогда fieldS_column есть, но с пустым содержимым

                    # print('fieldS_column:', fieldS_column) # для отладки

                    df = fieldsProcessor(df_in=df, fieldS_column=fieldS_column, response=response)

    return df

def errorProcessor(API_keyS, keyOrder, pause, response, tryer):
    goC = True
    goS = True

    if 'error' in response.keys():
        VK_ERROR_HANDLERS = {
            5:  'User authorization failed',          # авторизация
            6:  'Too many requests per second',       # лимит частоты
            9:  'Flood control',                      # флуд-контроль
            10: 'Internal server error',              # серверная ошибка
            14: 'Captcha needed',                     # капча
            15: 'Access denied',                      # нет прав
            100: 'One of the parameters specified was missing or invalid',
        }
        
        if VK_ERROR_HANDLERS[5] in response['error']['error_msg'] or response['error']['error_code'] == 5:
            print(
'''
Похоже, аккаунт попал под ограничение. Оно может быть снято с аккаунта сразу или спустя какое-то время.
Подождите или подготовьте новый ключ в другом аккаунте. И запустите скрипт с начала'''
                  )
            response = {'items': [], 'total_count': 0} # принудительная выдача для response
            goS = False # нет смысла продолжать исполнение скрипта
            goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        elif VK_ERROR_HANDLERS[6] in response['error']['error_msg'] or response['error']['error_code'] == 6:
            # print('  keyOrder до замены', '                    ') # для отладки

            keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
            print(
f'''
Похоже, ключ попал под ограничение вследствие слишком высокой частоты обращения скрипта к API;
пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту'''
                  )
            # print('  keyOrder после замены', keyOrder, '                    ') # для отладки

            pause += 0.25
            tryer += 1
            if tryer >= len(API_keyS):
                print(
'''
Попробовал все располагаемые ключи; все они заблокированны или неактивны(
Попробуйте обновить сервисный ключ в Вашем приложении API ВК,
после чего замените старые ключи новым в файле credentialsVK.txt и перезапустите этот скрипт'''
                )

                response = {'items': [], 'total_count': 0} # принудительная выдача для response
                goS = False # нет смысла продолжать исполнение скрипта
                goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        elif VK_ERROR_HANDLERS[10] in response['error']['error_msg'] or response['error']['error_code'] == 10:
            print('\nПохоже, ошибка на сервере ВК; подождите и запустите скрипт с начала')
            response = {'items': [], 'total_count': 0} # принудительная выдача для response
            goS = False # нет смысла продолжать исполнение скрипта
            goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        elif 'Application is blocked' in response['error']['error_msg']:
            # print('  keyOrder до замены', '                    ') # для отладки

            keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
            print(
f'''
Похоже, ключ попал под ограничение вследствие блокировки приложения, к которому он относится; пробую перейти к следующему ключу (№ {keyOrder})'''
            )

            # print('  keyOrder после замены', keyOrder, '                    ') # для отладки

            tryer += 1
            if tryer >= len(API_keyS):
                print(
'''
Попробовал все располагаемые ключи; все они заблокированны или неактивны(
Попробуйте обновить сервисный ключ в Вашем приложении API ВК,
после чего замените старые ключи новым в файле credentialsVK.txt и перезапустите этот скрипт'''
                )

                response = {'items': [], 'total_count': 0} # принудительная выдача для response
                goS = False # нет смысла продолжать исполнение скрипта
                goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        elif 'Unknown application: could not get application' in response['error']['error_msg']:
            # print('  keyOrder до замены', '                    ') # для отладки

            keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
            print(f'\nПохоже, Ваше ВК-приложение попало под ограничение; пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту')
            # print('  keyOrder после замены', keyOrder, '                    ') # для отладки

            pause += 0.25
            tryer += 1
            if tryer >= len(API_keyS):
                print(
'''
Попробовал все располагаемые ключи; все они заблокированны или неактивны(
Попробуйте обновить сервисный ключ в Вашем приложении API ВК,
после чего замените старые ключи новым в файле credentialsVK.txt и перезапустите этот скрипт'''
                )

                response = {'items': [], 'total_count': 0} # принудительная выдача для response
                goS = False # нет смысла продолжать исполнение скрипта
                goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        else:
            print('  Похоже, проблема НЕ в слишком высокой частоте обращения скрипта к API((')
            print('  ', response['error']['error_msg'])
            response = {'items': [], 'total_count': 0} # принудительная выдача для response
            goS = False # нет смысла продолжать исполнение скрипта
            goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

    return goC, goS, keyOrder, pause, response, tryer

# .. обработки выдачи аргумента fields
def fieldsProcessor(df_in, fieldS_column, response): # fieldS_column -- group или profile
    df = df_in.copy()

    idColumnS = [] # столбцы, чьи названия включают 'id'
    for column in df.columns:
    # for column in df.columns[1:]: # для отладки
        if 'id' in column:
            # print('column:', column) # для отладки
            idColumnS.append(column)

    columnsToJSON = varPreprocessor.jsonChecker(df)
    idColumnS.extend(columnsToJSON)

    fieldS_df = pandas.json_normalize(response[fieldS_column])
    if 'id' not in fieldS_df.columns: return df
    idS = fieldS_df['id'].to_list()
    
    idS_copyStr = ' '.join(map(str, idS)) # список в текстовый объект, чтобы ниже подать его внутрь столбца idColumnS_concatinated, созданного конкатенацией idColumnS
    # if idS:
    #     if len(idS) > 0:
    #         idS_copy = idS.copy()
    #         for idCopy in idS_copy: idS_copyStr += str(idCopy) + ' '
    #         idS_copyStr = idS_copyStr[:-2]
    #         # print('idS_copyStr':, idS_copyStr) # для отладки

    def fieldsIdsChecker(cellContent): # функция, приминяемая ниже посредством apply , чтобы ускорить процесс (по сравнению с циклом по ячейкам)
        if not cellContent or pandas.isna(cellContent): return ''
        idS_copy = cellContent.split('idS_copy')[1].split(' ')
        cellContent_list = cellContent.split('idS_copy')[0].split(' ')
        idS_toItemS = []
        for idCopy in idS_copy:
            for cellContent_item in cellContent_list:
                if idCopy == cellContent_item: idS_toItemS.append(int(idCopy))

        if len(idS_toItemS) > 0:
            # print('idS_toItemS не пустой список:', idS_toItemS) # для отладки

            try: return fieldS_df[fieldS_df['id'].isin(idS_toItemS)].to_dict('records')
            except Exception as excptn:
                print('Exception в fieldsIdsChecker') # для отладки
                print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
                print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой
                # print('dict:', fieldS_df[fieldS_df['id'].isin(idS_toItemS)].to_dict('records')) # для отладки
                return ''

        else:
            # print('idS_toItemS пустой список:', idS_toItemS) # для отладки
            return '' # заглушка

    df['idColumnS_concatinated'] = '' # столбец для конкатенации столбцов, чьи названия включают 'id'
    for idColumn in idColumnS: df['idColumnS_concatinated'] += ' ' + df[idColumn].astype(str)
    df['idColumnS_concatinated'] += 'idS_copy' + idS_copyStr # 'idS_copy' -- разделитель
    df[fieldS_column] = df['idColumnS_concatinated'].apply(fieldsIdsChecker)
    df = df.drop('idColumnS_concatinated', axis=1)
    
    try: df[fieldS_column] = df[fieldS_column].replace('N/A', numpy.nan)
    except Exception as excptn:
        print('Exception в fieldsProcessor') # для отладки
        print(f"{type(excptn).__name__}: {str(excptn).split('Stacktrace:')[0].strip()}") # для отладки
        print(traceback.format_exc().split('Stacktrace:')[0].strip()) # показ точной строчки кода с ошибкой

    return df
