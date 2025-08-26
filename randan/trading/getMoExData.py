# 0. Активировать требуемые для работы скрипта модули и пакеты
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        import os
        import re
        import warnings

        import pandas
        import requests
        from bs4 import BeautifulSoup
        from tqdm import tqdm

        from randan.tools import (
            coLabAdaptor,
        )  # авторский модуль для адаптации текущего скрипта к файловой системе CoLab

        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = (
            str(errorDescription[1]).replace("No module named '", "").replace("'", "")
        )  # .replace('_', '')
        if "." in module:
            module = module.split(".")[0]
        print(
            f"""Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас
Попытка № {attempt} из 10
"""
        )
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if attempt == 10:
            print(
                f"""Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
"""
            )
            break

coLabFolder = coLabAdaptor.coLabAdaptor()


# 1. Авторские функции
# выгрузки имён полей БД МосБиржи
def getColumnNameS(text):
    columnS = BeautifulSoup(text, features="xml").find_all("column")
    # print('columnS:', columnS) # для отладки
    columnNameS = []
    for column in columnS:
        columnNameS.append(column.get("name"))
    return columnNameS

    # выгрузки данных из БД МосБиржи


def pseudojson2df(headerS, index, url):
    df = pandas.DataFrame()
    text = re.findall(
        r"<data.+?/data>", requests.get(url, headers=headerS).text, re.DOTALL
    )[index]
    # print('text:', text) # для отладки
    columnNameS = getColumnNameS(text)
    rowS = BeautifulSoup(text, features="xml").find_all("row")
    # print('rowS:', rowS) # для отладки
    i = 0
    for row in tqdm(rowS):
        i += 1
        for column in columnNameS:
            df.loc[i, column] = row.get(column)
    # display('df:', df)
    return df


# 2. Авторская функция исполнения скрипта


def getMoExData(market="bonds", path=coLabFolder, returnDfs=False):
    """
    Функция умеет выгружать характеристики торгуемых на МосБирже облигаций, причём не дефолтные (далее -- Д) и не повышенного инвестиционного риска (далее -- ПИР). Дополнительно выгружается словарь полей БД МосБиржи. Также она умеет выгружать фьючерсы

    Parameters
    ----------
       market : str -- если интересуют облигации, подходит значение по умолчанию 'bonds' , если фьючерсы, впишите 'forts'

         path : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

    returnDfs : bool -- в случае True функция возвращает итоговые датафреймы boardS, columnsDescriptionS и securitieS строго в такой последовательности
    """
    slash = "\\" if os.name == "nt" else "/"  # выбор слэша в зависимости от ОС
    if path == None:
        path = ""
    else:
        path += slash

        # Формирование файла с режимами торгов

        # 1.0 Если нет файла с режимами торгов
    print("Создаю файл с режимами торгов")
    if market == "bonds":
        url = f"https://iss.moex.com/iss/engines/stock/markets/{market}"
    if market == "forts":
        url = f"https://iss.moex.com/iss/engines/futures/markets/{market}"
    headerS = {
        "Cookie": "yashr=7199406881722422993; yabs-sid=1989516261722422994; gdpr=0; _ym_uid=172242390960576307; _ym_d=1722423909; yandex_login=aleksei.rotmistrov; yandexuid=1251707911713359739; yuidss=1251707911713359739; ymex=2038826180.yrts.1723466180; skid=98896631723495108; yabs-dsp=mts_banner.bjhrRmhRcmRTYWFqa2szTTdWRHB2UQ==; my=YwA=; amcuid=9218374081731017878; yandex_gid=213; is_gdpr=0; is_gdpr_b=COXEFxCCoAIoAg==; i=uleIuerZ29JaTX59z5G/+HKk9fEmnUoKXjW/KGLZiTQaKYElKHEzfPDCABcpPVUVz6h+GEzjHO3ElrWjkRmIAGlp+lY=; Session_id=3:1733758173.5.0.1730723942669:tOnmRptGBpQAvmusaCECKg:471c.1.2:1|454550616.-1.2.3:1730723942|3:10299425.956559.MUCs35YHhfnyWe6-GuWX5wjaRxs; sessar=1.1196.CiDL7YFrdyEcpmiO9V7a1ylcpw6ej8qZiLU8_AgTxsNW_w.AGdGhxY1_HuHtpuOQLQHoSH6QAM9RilP9yNVtHdZlXc; sessionid2=3:1733758173.5.0.1730723942669:tOnmRptGBpQAvmusaCECKg:471c.1.2:1|454550616.-1.2.3:1730723942|3:10299425.956559.fakesign0000000000000000000; _ym_isad=2; yabs-vdrf=A0; yp=1736202372.atds.1#1735233224.hdrc.0#2049118172.pcs.0#1765295017.swntab.0#1746491882.szm.1_875%3A1280x720%3A1676x760#2046083942.udn.cDphbGVrc2VpLnJvdG1pc3Ryb3Y%3D#1738499314.vhstfltr_onb.1%3A1730723314256#1734471330.ygu.1#1734622172.dlp.2; ys=udn.cDphbGVrc2VpLnJvdG1pc3Ryb3Y%3D#wprid.1733759017472236-15201027122329898332-balancer-l7leveler-kubr-yp-sas-248-BAL#c_chck.3523091273; bh=EkAiTWljcm9zb2Z0IEVkZ2UiO3Y9IjEzMSIsIkNocm9taXVtIjt2PSIxMzEiLCJOb3RfQSBCcmFuZCI7dj0iMjQiGgUieDg2IiIPIjEzMS4wLjI5MDMuODYiKgI/MDICIiI6CSJXaW5kb3dzIkIIIjEwLjAuMCJKBCI2NCJSXCJNaWNyb3NvZnQgRWRnZSI7dj0iMTMxLjAuMjkwMy44NiIsIkNocm9taXVtIjt2PSIxMzEuMC42Nzc4LjEwOSIsIk5vdF9BIEJyYW5kIjt2PSIyNC4wLjAuMCIiWgI/MA==",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    }
    boardS = pseudojson2df(headerS, 0, url)
    # display('boardS:', boardS) # для отладки

    # 1.1 Если облигации: нужны именно торгуемые и не Д , и не ПИР
    if market == "bonds":
        boardS["is_traded"] = boardS["is_traded"].astype(int)
        boardS = boardS[
            (boardS["is_traded"] == 1)
            & (boardS["title"].str.contains("облигации ", case=False))
            & (boardS["title"].str.contains("ПИР ") != True)
            & (boardS["title"].str.contains("Д ") != True)
        ]
    # display('boardS:', boardS) # для отладки

    # Формирование файла с доступными облигациями

    # 2.1 Формирование словаря полей БД МосБиржи
    print("Создаю файл со словарём полей БД МосБиржи")
    columnsDescriptionS = pandas.DataFrame()
    if market == "bonds":
        indeceS = [2, 8]
    if market == "forts":
        indeceS = [2, 3]
    for index in indeceS:
        columnsDescriptionS_additional = pseudojson2df(headerS, index, url)
        columnsDescriptionS_additional.loc[:, "data id"] = index
        columnsDescriptionS = pandas.concat(
            [columnsDescriptionS, columnsDescriptionS_additional], ignore_index=True
        )
    columnsDescriptionS = columnsDescriptionS.drop_duplicates(
        ["id", "name"], ignore_index=True
    )
    # display('columnsDescriptionS:', columnsDescriptionS) # для отладки
    columnsDescriptionS.to_excel(market + "ColumnsDescriptionS.xlsx", index=False)

    # 2.2 Формирование файла с доступными облигациями в интересующих режимах торгов
    decision = ""
    if os.path.exists(path + market + "SecuritieS.xlsx"):
        print(
            """--- Если НЕ хотите обновить файл с доступными инструментами в интересующих режимах торгов, просто нажмите Enter
--- Если хотите, то нажмите пробел и затем Enter"""
        )
        decision = input()

    if (os.path.exists(path + market + "SecuritieS.xlsx") != True) | len(decision) != 0:
        print("Создаю файл с доступными инструментами в интересующих режимах торгов")
        securitieS = pandas.DataFrame()
        marketdata_yieldS = pandas.DataFrame()
        marketdata = pandas.DataFrame()
        # display("boardS['boardid']:", boardS['boardid']) # для отладки        
        for board in boardS['boardid']:
        # for board in ['TQOB']: # для отладки   
            print('board:', board)
            securitieS_additional = pseudojson2df(headerS, 0, url + f'/boards/{board}/securities')
            securitieS = pandas.concat([securitieS, securitieS_additional], ignore_index=True)

            if market == 'bonds':
                marketdata_yieldS_additional = pseudojson2df(headerS, -1, url + f'/boards/{board}/securities')
                marketdata_yieldS = pandas.concat([marketdata_yieldS, marketdata_yieldS_additional], ignore_index=True)
                # print('marketdata_yieldS.columns:', marketdata_yieldS.columns) # для отладки   

            if market == 'forts':
                marketdata_additional = pseudojson2df(headerS, 1, url + f'/boards/{board}/securities')
                # display('marketdata_additional:', marketdata_additional) # для отладки
                marketdata = pandas.concat([marketdata, marketdata_additional], ignore_index=True)
                # print('marketdata.columns:', marketdata.columns) # для отладки   

        if os.path.exists(path + market + 'ColumnsDescriptionsSelected.xlsx'):
            columnsDescriptionS = pandas.read_excel(path + market + 'ColumnsDescriptionsSelected.xlsx')
        else:
            columnsDescriptionS = pandas.read_excel(
                path + market + "ColumnsDescriptionS.xlsx"
            )
            # display('columnsDescriptionS:', columnsDescriptionS) # для отладки

            columnsDescriptionS = columnsDescriptionS[columnsDescriptionS['name'] !='BOARDID']

        columnsDescriptionS = columnsDescriptionS[columnsDescriptionS['name'].notna()]
        columnsDescriptionS = columnsDescriptionS['name'].drop_duplicates().tolist()
        if market == 'bonds': columnsDescriptionS.append('URL')
    
        if market == 'bonds':
            securitieS = securitieS.merge(marketdata_yieldS, on='SECID', suffixes=("", "_drop"), how="left")
            securitieS = securitieS[[column for column in securitieS.columns if not column.endswith("_drop")]]
            # print('securitieS.columns:', securitieS.columns) # для отладки

        if market == 'forts':
            securitieS = securitieS.merge(marketdata, on='SECID', suffixes=("", "_drop"), how="left")
            securitieS = securitieS[[column for column in securitieS.columns if not column.endswith("_drop")]]
            # print('securitieS.columns:', securitieS.columns) # для отладки

        securitieS = securitieS.groupby('SECID', as_index=False).first()
        # print('securitieS.columns:', securitieS.columns) # для отладки
        if market == 'bonds': securitieS['URL'] = 'https://www.moex.com/ru/issue.aspx?code=' + securitieS['ISIN']
        # print('securitieS.columns:', securitieS.columns) # для отладки
        securitieS = securitieS[columnsDescriptionS]
        securitieS.to_excel(path + market + "SecuritieS.xlsx", index=False)
        # display(securitieS) # для отладки
    else:
        print(
            "Файл с доступными инструментами в интересующих режимах торгов НЕ обновлялся"
        )
        securitieS = pandas.read_excel(path + market + "SecuritieS.xlsx")

    if returnDfs:
        return boardS, columnsDescriptionS, securitieS
    warnings.filterwarnings("ignore")
    print(
        'Скрипт исполнен. Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'
    )
    input()
    sys.exit()


# https://iss.moex.com/iss/reference/
# https://iss.moex.com/iss/engines/stock/markets/qnv
# https://iss.moex.com/iss/engines/stock/markets/bonds/boards/tqcb/securities
