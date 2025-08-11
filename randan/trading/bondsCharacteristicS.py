# 0.0 Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        from datetime import date
        from randan.trading import getMoExData # авторский модуль для выгрузки характеристик торгуемых на МосБирже облигаций
        from randan.tools import coLabAdaptor # авторский модуль для адаптации текущего скрипта к файловой системе CoLab
        import os, pandas, warnings
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

# 1. Авторская функция исполнения скрипта

def bondsCharacteristicsProcessor(
                                  bondsIn,
                                  path=coLabFolder,
                                  returnDfs=False
                                  ):
    """
    Функция для выяснения, какие облигации есть в портфеле, на основе брокерских отчётов

    Parameters
    ----------
      bondsIn : DataFrame -- датафрейм с облигациями, характеристики которых требуется получить; должен содержать хотя бы столбец ISIN
         path : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию, не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

    returnDfs : bool -- в случае True функция возвращает итоговые датафрейм bondS
    """
        
# 1.0 Настройки
    bondS = bondsIn.copy()
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if path == None: path = ''
    else: path += slash

    warnings.filterwarnings("ignore")

# 1.1 Добавить характеристики облигаций из БД МосБиржи
    boardS, columnsDescriptionS, сharacteristicsFromMoEx = getMoExData.getMoExData(market='bonds', returnDfs=True)
    # print('сharacteristicsFromMoEx.columns:', сharacteristicsFromMoEx.columns) # для отладки
    bondS = bondS.merge(сharacteristicsFromMoEx, on='ISIN', suffixes=("_drop", ""), how="left")
    bondS = bondS[[column for column in bondS.columns if not column.endswith("_drop")]]
    # print('bondS.columns:', bondS.columns) # для отладки
    bondS = bondS[bondS['SECNAME'].notna()] # убрать акции
    # display(bondS) # для отладки

# 1.2 Фильтры по датам
    # bondS = bondS[bondS['MATDATE'] != '0000-00-00'] # исключаются "вечные" облигации; обычно они субординорованные
    # bondS = bondS[bondS['NEXTCOUPON'] != '0000-00-00']
    # bondS = bondS[bondS['SETTLEDATE'] != '0000-00-00']
    # display(bondS) # для отладки

    # Сколько дней до купона?
    # bondS = bondS[bondS['NEXTCOUPON'] != bondS['SETTLEDATE']] # исключить облигации, по которым купон уже на след.день
    bondS['До купона'] = bondS['NEXTCOUPON'].astype('datetime64[ns]') - bondS['SETTLEDATE'].astype('datetime64[ns]')
    bondS['До купона'] = bondS['До купона'].astype(str)
    bondS['До купона'] = bondS['До купона'].str.split(' ').str[0]
    bondS['До купона'] = bondS['До купона'].astype(int)
    # display(bondS) # для отладки
    # display(bondS[bondS['ISIN'] == 'RU000A10B347']) # для отладки
    # display(сharacteristicsFromMoEx[сharacteristicsFromMoEx['ISIN'] == 'RU000A10B347']) # для отладки
    
    # Сколько дней до возможности погасить?
    bondS_offer = bondS[bondS['BUYBACKDATE'] != '0000-00-00'] # облигации С офертой
    offerS = bondS_offer.index
    # display(bondS) # для отладки

    # Сколько дней до оферты
    # Вычесть из даты оферты след.день 
    bondS_offer['До возможности погасить'] = (bondS_offer['BUYBACKDATE'] + '--' + bondS_offer['SETTLEDATE']).apply(lambda text:\
        str(date(int(text.split('--')[0].split('-')[0]), int(text.split('--')[0].split('-')[1]), int(text.split('--')[0].split('-')[2]))\
            - date(int(text.split('--')[1].split('-')[0]), int(text.split('--')[1].split('-')[1]), int(text.split('--')[1].split('-')[2]))
            ).split(' ')[0]
        )

    bondS_offer.loc[bondS_offer[bondS_offer['До возможности погасить'] == '0:00:00'].index, 'До возможности погасить'] = 0
    bondS_offer['До возможности погасить'] = bondS_offer['До возможности погасить'].astype(int)
    bondS_offer['Оферта'] = 'Есть'
    # display(bondS_offer) # для отладки

    # Облигации БЕЗ оферты
    bondS_other = bondS.drop(offerS)
    # display(bondS_other) # для отладки
    
    # До погашения
    # Вычесть из даты погашения след.день 
    bondS_other['До возможности погасить'] = (bondS_other['MATDATE'] + '--' + bondS_other['SETTLEDATE']).apply(lambda text:\
        str(date(int(text.split('--')[0].split('-')[0]), int(text.split('--')[0].split('-')[1]), int(text.split('--')[0].split('-')[2]))\
            - date(int(text.split('--')[1].split('-')[0]), int(text.split('--')[1].split('-')[1]), int(text.split('--')[1].split('-')[2]))
            ).split(' ')[0]
        )

    bondS_offer.loc[bondS_offer[bondS_offer['До возможности погасить'] == '0:00:00'].index, 'До возможности погасить'] = 0
    bondS_other['До возможности погасить'] = bondS_other['До возможности погасить'].astype(int)
    bondS_other['Оферта'] = 'Нет'
    # display(bondS_other) # для отладки
    
    bondS = pandas.concat([bondS_offer, bondS_other])
    # display(bondS) # для отладки

# 1.3 Предобрабока столбцов с финансовой информацией
    for column in ['ACCRUEDINT', 'COUPONPERCENT', 'FACEVALUE', 'PRICE']:
        bondS = bondS[bondS[column] != '']
        bondS[column] = bondS[column].astype(float)

    # Умножение FACEVALUE и ACCRUEDINT на цену валюты в рублях
    boardS, columnsDescriptionS, exchangesRaw = getMoExData.getMoExData(market='forts', returnDfs=True)
    exchangesRaw = exchangesRaw[['SHORTNAME', 'LAST', 'SETTLEPRICE']]
    exchangesRaw.columns = ['Unnamed: 0', 'Цена послед.', 'Цена закр.']
    # display(futureS) # для отладки
    
    # Из QUIK
    # exchangesRaw = pandas.read_excel(r'C:\Users\Alexey\Dropbox\QUIK_УралСиб_Driver\Текущие_торги.xlsx', usecols='A, D, F')
    # display(exchangesRaw) # для отладки
    
    currencieS = list(bondS['FACEUNIT'].unique()) # валюта номинала
    print('currencieS:', currencieS) # для отладки
    currencieS.remove('SUR')
    exchangeS = pandas.DataFrame()
    
    for currency in currencieS:
    # for currency in currencieS[0:1]: # для отладки
        exchangesAdditional = exchangesRaw[exchangesRaw['Unnamed: 0'].str.contains(currency, case=False)]
        # display('exchangesAdditional:', exchangesAdditional) # для отладки
        if len(exchangesAdditional) > 1: exchangesAdditional = exchangesAdditional.iloc[[0], :] # чтобы не брать пару USD|CNY
        # display('exchangesAdditional:', exchangesAdditional) # для отладки
        exchangesAdditional['Валюта'] = currency
        exchangeS = pandas.concat([exchangeS, exchangesAdditional])

    for column in ['Цена послед.', 'Цена закр.']:
        exchangeS[column] = exchangeS[column].astype(float)
        
    display('exchangeS:', exchangeS[['Цена послед.', 'Цена закр.', 'Валюта']]) # для отладки
        
    exchangeS.loc[exchangeS['Цена послед.'] == 0, 'Цена послед.'] = exchangeS.loc[exchangeS['Цена послед.'] == 0, 'Цена закр.'] # на случай нулей в столбце 'Цена послед.'
    exchangeS = exchangeS.drop(['Unnamed: 0', 'Цена закр.'], axis=1)
    
    # Поскольку исходно CHF в паре с USD
    if (exchangeS['Валюта'] == 'CHF').sum() > 0:
        exchangeS.loc[exchangeS['Валюта'] == 'CHF', 'Цена послед.'] =\
            exchangeS.loc[exchangeS['Валюта'] == 'USD', 'Цена послед.'][exchangeS[exchangeS['Валюта'] == 'USD'].index[0]]\
            / exchangeS.loc[exchangeS['Валюта'] == 'CHF', 'Цена послед.'][exchangeS[exchangeS['Валюта'] == 'CHF'].index[0]]
    
    exchangeS = exchangeS.sort_values('Валюта').reset_index(drop=True)
    # display('exchangeS:', exchangeS) # для отладки
    
    for currency in currencieS:
        currencyExchangeValue = exchangeS.loc[exchangeS['Валюта'] == currency, 'Цена послед.'][exchangeS[exchangeS['Валюта'] == currency].index[0]]
        # print('currencyExchangeValue:', currencyExchangeValue) # для отладки        
        # print('type(currencyExchangeValue):', type(currencyExchangeValue)) # для отладки        
        bondS.loc[bondS['FACEUNIT'] == currency, 'FACEVALUE'] *= currencyExchangeValue
        bondS.loc[bondS['CURRENCYID'] == currency, 'ACCRUEDINT'] *= currencyExchangeValue # валюта расчётов

# 1.4 Расчёт доходности в день до возможности погасить 
    # Если купить примерно на 1000 единиц валюты, то придётся заплатить
    bondS['Полная цена покупки'] = 1000 + 1000 / bondS['FACEVALUE'] * bondS['ACCRUEDINT']
    bondS['Полная цена покупки'] = bondS['Полная цена покупки'].astype(float).round(2)
    
    # На 1000 единиц валюты к погашению будет начислен купоный доход
    bondS['Купоный доход к погашению'] = 1000 * bondS['COUPONPERCENT'] / 36500 * bondS['До возможности погасить']
    bondS['Купоный доход к погашению'] = bondS['Купоный доход к погашению'].astype(float).round(2)
    
    # Плюс 1000 единиц валюты изменятся к погашению в связи с приведением цены к номиналу
    bondS['Маржа к погашению'] = 1000 * (100 - bondS['PRICE']) / 100
    bondS['Маржа к погашению'] = bondS['Маржа к погашению'].astype(float).round(2)
    
    # Подневная доходность к погашению = суммарный доход к полной цене покупки
    bondS['% доходности в день к погашению'] =\
        (
        100 * (1000 + bondS['Купоный доход к погашению'] + bondS['Маржа к погашению'])\
        / bondS['Полная цена покупки'] - 100\
        )/ bondS['До возможности погасить']
    bondS['% доходности в день к погашению'] = bondS['% доходности в день к погашению'].astype(float).round(4)

    # !!! Стоимость!!!
    bondS['Стоимость'] = bondS['Лотов'] * bondS['PRICE'] * 10 * bondS['FACEVALUE'] / 1000

# 1.5 Рейтинг и другие важные характеристики из bondsRatingS
    if os.path.exists(path + 'Замеры рейтингов'):
        # print("Директория 'Замеры рейтингов' существует") # для отладки
        fileNameS_inDirectory = os.listdir('Замеры рейтингов')
        fileNameS_inDirectory.sort(reverse=True)        
        fileNameS_forUse = []
        for fileName in fileNameS_inDirectory:
            if 'bondsRatingS_' in fileName:
                # print(f"Работаю с файлом '{fileName}'") # для отладки
                fileNameS_forUse.append(fileName)
                if len(fileNameS_forUse) == 2: break
        print(f"\nРаботаю с файлом bondsRatingS_previous:'{fileName}'")
        bondsRatingS = pandas.read_excel('Замеры рейтингов' + slash + fileNameS_forUse[0], index_col=0)
        # display('bondsRatingS:', bondsRatingS) # для отладки
        bondsRatingS_previous = pandas.read_excel('Замеры рейтингов' + slash + fileNameS_forUse[1], index_col=0)
        # display('bondsRatingS_previous:', bondsRatingS_previous) # для отладки
        # display('bondS:', bondS) # для отладки
        bondsRatingS = bondsRatingS[bondsRatingS['ISIN'].isin(bondS['ISIN'])]
        # display('bondsRatingS:', bondsRatingS) # для отладки
        bondsRatingS = bondsRatingS.merge(bondsRatingS_previous[['ISIN', 'rating']], on='ISIN', suffixes=("", "_previous"), how="left")
        # display('bondsRatingS:', bondsRatingS) # для отладки
        bondsRatingS.loc[bondsRatingS['rating'] != bondsRatingS['rating_previous'], 'С прошлого замера'] = 'Рейтинг изменился'
        print('\n Изменение лейтинга с прошлого замера:')
        display(bondsRatingS[bondsRatingS['С прошлого замера'] == 'Рейтинг изменился'][['ISIN', 'rating', 'rating_previous']])

        # display('bondS:', bondS) # для отладки
        # display('bondsRatingS:', bondsRatingS) # для отладки
        bondS = bondS.merge(bondsRatingS, on='ISIN', suffixes=("_drop", ""), how="left")
        bondS = bondS[[column for column in bondS.columns if not column.endswith("_drop")]]
    else:
        print("Найдите и запустите скрипт bondsRatingS")
        bondsRatingS = pandas.DataFrame()

# 1.6 Интегральная переменная Специфика
    bondS.loc[(bondS['Сектор рынка'] == 'Гос') | (bondS['SECNAME'].str.contains('ОФЗ|Россия', case=False)), 'Сектор рынка'] = 'Гос'
    bondS.loc[(bondS['Сектор рынка'].str.contains('Гос|Корп|Мун', case=False) != True), 'Сектор рынка'] = '?Корп'
    bondS['Специфика'] = bondS['FACEUNIT'].str[:2]
    for column in ['Сектор рынка', 'Амортизация', 'Тип купона', 'Тип текущего купона', 'Структурный параметр', 'Субординированность']:
        bondS[column] = bondS[column].fillna('--')
        bondS['Специфика'] += ' ' + bondS[column].str[:2]
    display("bondS['Специфика']:", bondS['Специфика'].value_counts())

    if returnDfs: return bondS
