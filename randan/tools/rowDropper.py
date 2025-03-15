df = df.drop([307, 1037, 1339, 1420, 1433])
columnsToJSON = list(df.columns) # все столбцы датафрейма `df` записать в объект columnsToJSON , причём отнести класс этого объекта к списку методом класса списков list()

for column in df.columns:
    if df[column].apply(lambda содержимоеПроверяемойЯчейки: True if (type(содержимоеПроверяемойЯчейки) == dict) | (type(содержимоеПроверяемойЯчейки) == list) else False).sum() == 0:
        columnsToJSON.remove(column)

if len(columnsToJSON) > 0: # проверка, что список columnsToJSON содержит более, чем 0 элементов
    print('Обнаружены столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
          , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')

    df.drop(columnsToJSON, axis=1).to_excel(r"YT_раздельный сбор\20250311_0230_YT_раздельный сбор_videos_Other_varS.xlsx")
    df[columnsToJSON].to_json(r"YT_раздельный сбор\20250311_0230_YT_раздельный сбор_videos_JSON_varS.json")
