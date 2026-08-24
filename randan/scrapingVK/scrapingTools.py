# .. обработки столбцов выдачи
def dfColumnsProcessor(df_in, response):
    df = df_in.copy()
    df['date'] = df['date'].apply(lambda content: datetime.fromtimestamp(content).strftime('%Y.%m.%d'))
        # сменить формат представления дат, класс данных столбцов с id, создать столбец с кликабельными ссылками на контент;
            # здесь, а не в конце, поскольку нужна совместимость с itemS из Temporal и от пользователя

    df['URL'] = df['from_id'].astype(str)
    df.loc[df[df['URL'].str.contains('-') == False].index, 'URL'] = 'id' + df.loc[df[df['URL'].str.contains('-') == False].index, 'URL']
    df.loc[df[df['URL'].str.contains('-')].index, 'URL'] = df.loc[df[df['URL'].str.contains('-')].index, 'URL'].str.replace('-', 'public')
    df['URL'] =\
        'https://vk.com' + '/' + df['URL'] + '?w=' + df['inner_type'].str.split('_').str[0] + df['owner_id'].astype(str) + '_' + df['id'].astype(str)

    if fields != None:
        for fieldsColumn in ['groups', 'profiles']:
            if fieldsColumn in response.keys():
                if response[fieldsColumn] != []: # например, когда в основном df нет групповых или, наоборот, персональных аккаунтов,
                        # тогда fieldsColumn есть, но с пустым содержимым

                    # print('fieldsColumn:', fieldsColumn) # для отладки

                    df = fieldsProcessor(dfIn=df, fieldsColumn=fieldsColumn, response=response)

    return df

def errorProcessor(API_keyS, keyOrder, pause, response, tryer):
    goC = True
    goS = True

    if 'error' in response.keys():
        if 'Application is blocked' in response['error']['error_msg']:
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
                # response = {'items': [], 'total_count': 0} # принудительная выдача для response
                goS = False # нет смысла продолжать исполнение скрипта
                goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        elif 'Too many requests per second' in response['error']['error_msg']:
            # print('  keyOrder до замены', '                    ') # для отладки

            keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
            print(
f'''
Похоже, ключ попал под ограничение вследствие слишком высокой частоты обращения скрипта к API;
пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту'''
                  )
            # print('  keyOrder после замены', keyOrder, '                    ') # для отладки

            pause += 0.25

        elif 'Unknown application: could not get application' in response['error']['error_msg']:
            # print('  keyOrder до замены', '                    ') # для отладки

            keyOrder = keyOrder + 1 if keyOrder < (len(API_keyS) - 1) else 0 # смена ключа, если есть на что менять
            print('\nПохоже, Ваше ВК-приложение попало под ограничение; пробую перейти к следующему ключу (№ {keyOrder}) и снизить частоту')
            # print('  keyOrder после замены', keyOrder, '                    ') # для отладки
            pause += 0.25

        elif 'Internal server error: Unknown error, try later' in response['error']['error_msg']:
            print('\nПохоже, ошибка на сервере ВК; подождите и запустите скрипт с начала')
            response = {'items': [], 'total_count': 0} # принудительная выдача для response
            goS = False # нет смысла продолжать исполнение скрипта
            goC = False # и, следовательно, нет смысла в новых итерациях цикла while goC

        elif 'User authorization failed' in response['error']['error_msg']:
            print(
'''
Похоже, аккаунт попал под ограничение. Оно может быть снято с аккаунта сразу или спустя какое-то время.
Подождите или подготовьте новый ключ в другом аккаунте. И запустите скрипт с начала'''
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
