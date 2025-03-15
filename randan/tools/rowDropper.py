from randan.tools import varPreprocessor
df = df.drop([904, 2173, 2392, 2822, 3252, 3558, 3865, 3925])
# Проверка всех столбцов на наличие в их ячейках JSON-формата
from randan.tools import varPreprocessor
columnsToJSON = varPreprocessor.jsonChecker(df)

if len(columnsToJSON) > 0:
    print('Обнаружены столбцы, содержащие внутри своих ячеек JSON-объекты; Excel не поддерживает JSON-формат;'
          , 'чтобы формат JSON не потерялся, сохраняю эти столбцы в файл формата НЕ XLSX, а JSON. Остальные же столбцы сохраняю в файл формата XLSX')

    columnsToJSON.append('id')
    if 'from_id' in df.columns: columnsToJSON.append('from_id')
    if 'owner_id' in df.columns: columnsToJSON.append('owner_id')

    df[columnsToJSON].to_json(r"YT_раздельный сбор\20250311_0230_YT_раздельный сбор_videos_JSON_varS.json")

    columnsToJSON.remove('id')
    if 'from_id' in columnsToJSON: columnsToJSON.remove('from_id')
    if 'owner_id' in columnsToJSON: columnsToJSON.remove('owner_id')

    df.drop(columnsToJSON, axis=1).to_excel(r"YT_раздельный сбор\20250311_0230_YT_раздельный сбор_videos_Other_varS.xlsx")
