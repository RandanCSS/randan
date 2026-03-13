def groups_profiles_parser(columnName, dfIn, folder):
    JSONS = []
    for cellContent in dfIn[columnName].dropna():
        JSONS.extend(cellContent)
    df = pandas.json_normalize(JSONS).drop_duplicates('id').reset_index(drop=True)
    display('df:', df)


    # Проверка всех столбцов на наличие в их ячейках словарей и списков
    columnsToJSON = list(df.columns) # все столбцы датафрейма df записать в объект columnsToJSON ,
        # причём отнести тип этого объекта к списку методом списков list()

    for column in df.columns: # цикл для прохода по всем столбцам датафрейма df
        # Если в столбце не встречаются ячейки со словарями или списками, то..
        if df[column].apply(lambda содержимоеПроверяемойЯчейки: True if (type(содержимоеПроверяемойЯчейки) == dict) | (type(содержимоеПроверяемойЯчейки) == list) else False).sum() == 0:
            columnsToJSON.remove(column) # .. то этот столбец исключается из "подозреваемых" методм .remove() класса списков

    print('columnsToJSON:', columnsToJSON)


    # Экспортировать датафрейм в новые файлы формата .xlsx и .json и расположить их внутри директории
    if len(columnsToJSON) > 0: # проверка, что список columnsToJSON содержит более, чем 0 элементов
        print('Обнаружены столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
              , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')
    
        # Три следующих столбца не содержат внутри своих ячеек JSON-объекты, но эти столбцы обеспечат соединение файлов формата .xlsx и .json , чтобы строки таблиц не перепутались
        columnsToJSON.append('id')
        if 'from_id' in df.columns: columnsToJSON.append('from_id')
        if 'owner_id' in df.columns: columnsToJSON.append('owner_id')

        df2file.df2file(df, columnName + '.json', folder) # экспортировать датафрейм в файл .json,
            # причём предварительно оставить в нём посредтвом индексирования только столбцы columnsToJSON
            # и расположить его внутри директории folder
    
        # Теперь эти три столбца удаляются из списка columnsToJSON , посколку в дальнейшем столбцы, содержащиеся в списке columnsToJSON , будут удалены из датафрейма при его сохранении в файл формата .xlsx
        columnsToJSON.remove('id')
        if 'from_id' in columnsToJSON: columnsToJSON.remove('from_id')
        if 'owner_id' in columnsToJSON: columnsToJSON.remove('owner_id')
    
        df2file.df2file(df.drop(columnsToJSON, axis=1), columnName + '.xlsx', folder) # экспортировать датафрейм в файл .xlsx, причём предварительно удалить из него столбцы columnsToJSON
            # посредством метода .drop() пакета pandas и расположить его внутри директории folder
    
    else: df2file.df2file(df, columnName + '.xlsx', folder) # экспортировать датафрейм в файл .xlsx со ВСЕМИ столбцами и расположить его внутри директории folder
            # и расположить его внутри директории
