# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from bs4 import BeautifulSoup
        from tqdm import tqdm
        from randan.tools import coLabAdaptor # авторский модуль для адаптации текущего скрипта к файловой системе CoLab
        import os, pandas, re, requests, warnings
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[0]
        print(
f'''Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 10
'''
              )
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break

coLabFolder = coLabAdaptor.coLabAdaptor()
tqdm.pandas() # для визуализации прогресса функций, применяемых к датафреймам

# 1. Авторские функции
    
def getColumnNameS(text):
    columnS = BeautifulSoup(text, features='xml').find_all('column')
    # print('columnS:', columnS) # для отладки
    columnNameS = []
    for column in columnS:
        columnNameS.append(column.get('name'))
    return columnNameS

def pseudojson2df(index, url):
    df = pandas.DataFrame()
    headerS = {'Cookie': 'yashr=7199406881722422993; yabs-sid=1989516261722422994; gdpr=0; _ym_uid=172242390960576307; _ym_d=1722423909; yandex_login=aleksei.rotmistrov; yandexuid=1251707911713359739; yuidss=1251707911713359739; ymex=2038826180.yrts.1723466180; skid=98896631723495108; yabs-dsp=mts_banner.bjhrRmhRcmRTYWFqa2szTTdWRHB2UQ==; my=YwA=; amcuid=9218374081731017878; yandex_gid=213; is_gdpr=0; is_gdpr_b=COXEFxCCoAIoAg==; i=uleIuerZ29JaTX59z5G/+HKk9fEmnUoKXjW/KGLZiTQaKYElKHEzfPDCABcpPVUVz6h+GEzjHO3ElrWjkRmIAGlp+lY=; Session_id=3:1733758173.5.0.1730723942669:tOnmRptGBpQAvmusaCECKg:471c.1.2:1|454550616.-1.2.3:1730723942|3:10299425.956559.MUCs35YHhfnyWe6-GuWX5wjaRxs; sessar=1.1196.CiDL7YFrdyEcpmiO9V7a1ylcpw6ej8qZiLU8_AgTxsNW_w.AGdGhxY1_HuHtpuOQLQHoSH6QAM9RilP9yNVtHdZlXc; sessionid2=3:1733758173.5.0.1730723942669:tOnmRptGBpQAvmusaCECKg:471c.1.2:1|454550616.-1.2.3:1730723942|3:10299425.956559.fakesign0000000000000000000; _ym_isad=2; yabs-vdrf=A0; yp=1736202372.atds.1#1735233224.hdrc.0#2049118172.pcs.0#1765295017.swntab.0#1746491882.szm.1_875%3A1280x720%3A1676x760#2046083942.udn.cDphbGVrc2VpLnJvdG1pc3Ryb3Y%3D#1738499314.vhstfltr_onb.1%3A1730723314256#1734471330.ygu.1#1734622172.dlp.2; ys=udn.cDphbGVrc2VpLnJvdG1pc3Ryb3Y%3D#wprid.1733759017472236-15201027122329898332-balancer-l7leveler-kubr-yp-sas-248-BAL#c_chck.3523091273; bh=EkAiTWljcm9zb2Z0IEVkZ2UiO3Y9IjEzMSIsIkNocm9taXVtIjt2PSIxMzEiLCJOb3RfQSBCcmFuZCI7dj0iMjQiGgUieDg2IiIPIjEzMS4wLjI5MDMuODYiKgI/MDICIiI6CSJXaW5kb3dzIkIIIjEwLjAuMCJKBCI2NCJSXCJNaWNyb3NvZnQgRWRnZSI7dj0iMTMxLjAuMjkwMy44NiIsIkNocm9taXVtIjt2PSIxMzEuMC42Nzc4LjEwOSIsIk5vdF9BIEJyYW5kIjt2PSIyNC4wLjAuMCIiWgI/MA=='
        , 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'}
    text = re.findall(r'<data.+?/data>', requests.get(url, headers=headerS).text, re.DOTALL)[index]
    # print('text:', text) # для отладки
    columnNameS = getColumnNameS(text)
    rowS = BeautifulSoup(text, features='xml').find_all('row')
    # print('rowS:', rowS) # для отладки
    i = 0
    for row in tqdm(rowS):
        i += 1
        for column in columnNameS:
            df.loc[i, column] = row.get(column)
    # display('df:', df)
    return df

def getBonds(
             path=coLabFolder,
             returnDfs=False
             ):
    """
    Функция для выгрузки характеристик торгуемых на МосБирже облигаций, причём не Д (дефолтные) и не ПИР (повышенный инвестиционный риск). Дополнительно выгружается словарь полей БД МосБиржи

    Parameters
    ----------
         path : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию, не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

    returnDfs : bool -- в случае True функция возвращает итоговые датафреймы boardS, bondS и columnsDescriptionsSelected
    """
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if path == None: path = ''
    else: path += slash

    # Формирование файла с режимами торгов
    
    # 1.0 Если нет файла с режимами торгов
    
    # Опасный блок, т.к. не все boardS требуются в дальнейшем
    # print(
# '''--- Если НЕ хотите обновить файл с режимами торгов, просто нажмите Enter
# --- Если хотите, то введите пробел, после чего нажмите Enter'''
    #       )
    # if len(input()) != 0:
    
    if os.path.exists(path + 'boardS.xlsx') == False:
        print('Файл с режимами торгов создаётся')
        rowS = BeautifulSoup(requests.get('https://iss.moex.com/iss/engines/stock/markets/bonds/boards',
                                          headers=headerS).text, features='xml').find_all('row')
        # print('rowS:', rowS)
        colS = ['board_group_id', 'boardid', 'is_traded', 'title']
        boardS = pandas.DataFrame(columns=colS)
        i = 0
        for row in tqdm(rowS):
            i += 1
            for col in colS:
                boardS.loc[i, col] = row.get(col)
        # display('boardS:', boardS)
        boardS.to_excel(path + 'boardS.xlsx', index=False)
    else:
        print('Файл с режимами торгов существует')
        boardS = pandas.read_excel(path + 'boardS.xlsx')
    
    # 1.1 Нужны именно облигации, причём торгуемые, не Д (дефолтные) и не ПИР
    boardS = boardS[(boardS['is_traded'] == 1) & (boardS['title'].str.contains('Облигации ', case=False))\
        & (boardS['title'].str.contains('ПИР ') != True) & (boardS['title'].str.contains('Д ') != True)]
    boardS
    
    # Формирование файла с доступными облигациями
    
    # 2.1 Формирование словаря полей БД МосБиржи
    if os.path.exists(path + 'columnsDescriptionS.xlsx') == False:
        print('Файл со словарём полей БД МосБиржи создаётся')
        columnsDescriptionS = pandas.DataFrame()
        for index in [2, 8]:
            columnsDescriptionS_additional = pseudojson2df(index, 'https://iss.moex.com/iss/engines/stock/markets/bonds/')
            columnsDescriptionS_additional.loc[:, 'data id'] = index
            columnsDescriptionS = pandas.concat([columnsDescriptionS, columnsDescriptionS_additional], ignore_index=True)
        columnsDescriptionS = columnsDescriptionS.drop_duplicates(['id', 'name'], ignore_index=True)
        # display('columnsDescriptionS:', columnsDescriptionS) # для отладки
        columnsDescriptionS.to_excel(path + 'columnsDescriptionS.xlsx', index=False)
    else:
        print('Файл со словарём полей БД МосБиржи существует')
        columnsDescriptionS = pandas.read_excel(path + 'columnsDescriptionS.xlsx')
    
    # 2.2 Формирование файла с доступными облигациями в интересующих режимах торгов
    print(
'''--- Если НЕ хотите обновить файл с доступными облигациями в интересующих режимах торгов, просто нажмите Enter
--- Если хотите, то введите пробел, после чего нажмите Enter'''
          )
    if len(input()) != 0:
        print('Файл с доступными облигациями в интересующих режимах торгов создаётся')
        securitieS = pandas.DataFrame()
        marketdata_yieldS = pandas.DataFrame()
        for board in boardS['boardid']:
            print(board)
            urlGlobal = f'https://iss.moex.com/iss/engines/stock/markets/bonds/boards/{board}/securities'
            securitieS_additional = pseudojson2df(0, urlGlobal)
            securitieS = pandas.concat([securitieS, securitieS_additional], ignore_index=True)
            marketdata_yieldS_additional = pseudojson2df(-1, urlGlobal)
            marketdata_yieldS = pandas.concat([marketdata_yieldS, marketdata_yieldS_additional], ignore_index=True)

        if os.path.exists(path + 'columnsDescriptionsSelected.xlsx'): columnsDescriptionS = pandas.read_excel(path + 'columnsDescriptionsSelected.xlsx')
        else:
            columnsDescriptionS = pandas.read_excel(path + 'columnsDescriptionS.xlsx')
            # display(columnsDescriptionS) # для отладки
            # print('BOARDID' in columnsDescriptionS.columns) # для отладки
            columnsDescriptionS = columnsDescriptionS[columnsDescriptionS['name'] !='BOARDID']
        columnsDescriptionS = columnsDescriptionS[columnsDescriptionS['name'].notna()]
        columnsDescriptionS = columnsDescriptionS['name'].tolist()
        columnsDescriptionS.append('URL')
    
        bondS = securitieS.merge(marketdata_yieldS, on='SECID')
        bondS = bondS.groupby('SECID', as_index=False).first()
        bondS['URL'] = 'https://www.moex.com/ru/issue.aspx?code=' + bondS['ISIN']
        bondS = bondS[columnsDescriptionsSelected]
        bondS.to_excel(path + 'bondS.xlsx', index=False)
        # display(bondS)
    else:
        print('Файл с доступными облигациями в интересующих режимах торгов НЕ обновлялся')
        bondS = pandas.read_excel(path + 'bondS.xlsx')

    if returnDfs: return boardS, bondS, columnsDescriptionS
    warnings.filterwarnings("ignore")
    print('Скрипт исполнен. Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть')
    input()
    sys.exit()
