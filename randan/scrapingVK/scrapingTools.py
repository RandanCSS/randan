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
