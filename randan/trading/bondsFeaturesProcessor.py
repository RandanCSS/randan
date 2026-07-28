# Авторский модуль для гармонизации и обработки характеристик облигаций

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from datetime import date
        from randan.trading import getMoExData, issuerProcessor, ratingProcessor # авторские модули для
            # (а) выгрузки характеристик торгуемых на МосБирже облигаций,
            # (б) операций с эмитентами торгуемых на МосБирже облигаций

        from randan.tools import coLabAdaptor, files2df # авторские модули для
            # (а) адаптации текущего скрипта к файловой системе CoLab,
            # (б) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа

        import os, pandas, warnings
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
        check_call([sys.executable, "-m", "pip", "install", module])
        if  attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )
            break

coLabFolder = coLabAdaptor.coLabAdaptor()

# 1. Авторская функция исполнения скрипта

def bondsFeaturesProcessor(
                           bondsIn,
                           issuerS,
                           pause,
                           path=coLabFolder,
                           returnDfs=False
                           ):
    """
    Функция для выяснения, какие облигации есть в портфеле, на основе брокерских отчётов

    Parameters
    ----------
      bondsIn : DataFrame -- датафрейм с облигациями, характеристики которых требуется получить; должен содержать хотя бы столбец ISIN
      issuerS : DataFrame -- датафрейм со Словарём эмитентов
        pause : int -- период засыпания исполнения функций selenium
         path : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию, не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

    returnDfs : bool -- в случае True функция возвращает итоговые датафрейм bondS
    """
        
# 1.0 Настройки
    bondS = bondsIn.copy()
    bondS = bondS.drop_duplicates('ISIN', keep='last', ignore_index=True)
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС

    if path == None: path = ''
    else: path += slash

    warnings.filterwarnings("ignore")

# 1.1 Добавить характеристики облигаций из БД МосБиржи
    boardS, columnsDescriptionS, securitieS = getMoExData.getMoExData(market='bonds', returnDfs=True)
    bondS = bondS.merge(securitieS, how='left', on='ISIN', suffixes=('_drop', ''))
    bondS = bondS[[column for column in bondS.columns if not column.endswith('_drop')]]
    # print('bondS.columns:', bondS.columns) # для отладки

# Добавить столбцы Эмитент и Bond D Rating и Issuer D Rating; при необходимости, дозаполнить последние
    bondS = issuerProcessor.issuerNameProcessor(bondS, issuerS)
        # теперь в bondS у каждой облигации указан эмитент с названием, соотнесённым со Словарём эмитентов
            # (все эти эмитенты представлены в issuerS)

# 1.2 Импорт словарей (актуального и прошлого) эмитентов с рейтингом из Акуальные эмитенты.xlsx в issuerS_withActualRating
    # print('path:', path) # для отладки
    if os.path.exists(path + 'Замеры рейтингов'):
        fileUptodateName_0 = files2df.getFileUptodateName('_Акуальные эмитенты', None, path + 'Замеры рейтингов')
        # print('fileUptodateName_0:', fileUptodateName_0) # для отладки

        issuerS_withActualRating = pandas.read_excel(path + 'Замеры рейтингов' + slash + fileUptodateName_0)
        # display('issuerS_withActualRating:', issuerS_withActualRating) # для отладки

        fileUptodateName_1 = files2df.getFileUptodateName('_Акуальные эмитенты', [fileUptodateName_0], path + 'Замеры рейтингов')
        # print('fileUptodateName_1:', fileUptodateName_1) # для отладки

        issuerS_withActualRating_previous = pandas.read_excel(path + 'Замеры рейтингов' + slash + fileUptodateName_1)
        # display('issuerS_withActualRating_previous:', issuerS_withActualRating_previous) # для отладки

        issuerS_withActualRating = issuerS_withActualRating.merge(issuerS_withActualRating_previous[['Эмитент', 'Issuer D Rating']], on='Эмитент', suffixes=("", " Previous"))
        # display('issuerS_withActualRating:', issuerS_withActualRating) # для отладки

    else:
        print("Найдите и запустите скрипт bondsRatingS")
        issuerS_withActualRating = pandas.DataFrame()

    issuerS_withActualRating = issuerS_withActualRating[issuerS_withActualRating['Эмитент'].isin(bondS['Эмитент'].unique())]
        # убрать эмитенты, не относящиеся к рассматриваемым облигациям

    issuerS_withActualRating.loc[issuerS_withActualRating['Issuer D Rating'] != issuerS_withActualRating['Issuer D Rating Previous'], 'С прошлого замера'] = 'Рейтинг изменился'

    print('\n Изменения рейтинга с прошлого замера:')
    print('  Повышение')
    issuerS_withActualRating_up = issuerS_withActualRating[
        (issuerS_withActualRating['Issuer D Rating'].notna()) & (issuerS_withActualRating['Issuer D Rating Previous'].notna()) &\
        (issuerS_withActualRating['Issuer D Rating'] > issuerS_withActualRating['Issuer D Rating Previous'])
        ]

    display(issuerS_withActualRating_up)

    print('  Понижение')
    issuerS_withActualRating_down = issuerS_withActualRating[
        (issuerS_withActualRating['Issuer D Rating'].notna()) & (issuerS_withActualRating['Issuer D Rating Previous'].notna()) &\
        (issuerS_withActualRating['Issuer D Rating'] < issuerS_withActualRating['Issuer D Rating Previous'])
        ]

    display(issuerS_withActualRating_down)

    issuerS_withActualRating_change = pandas.concat([issuerS_withActualRating_up, issuerS_withActualRating_down])

    bondS = bondS.merge(issuerS_withActualRating, how="left", on='Эмитент', suffixes=("_drop", ""))
    bondS = bondS[[column for column in bondS.columns if not column.endswith("_drop")]]

    # display('bondS:', bondS) # для отладки

# 1.3 "Шеринг" рейтинга из issuerS_withActualRating# и выгрузка с сайта moex.ru
    for issuer_withActualRating in issuerS_withActualRating['Эмитент']:
        print('issuer_withActualRating:', issuer_withActualRating) # для отладки
        issuerS_withActualRating_index = issuerS_withActualRating[issuerS_withActualRating['Эмитент'] == issuer_withActualRating].index
        bondS.loc[bondS['Эмитент'] == issuer_withActualRating, 'Issuer D Rating'] = issuerS_withActualRating.loc[issuerS_withActualRating_index[0], 'Issuer D Rating']

#     # Разделение облигаций по субординированности, поскольку для несубординированных обычно рейтинг эмитента и облигации совпадают
#         # и для них можно выводить Bond D Rating и Issuer D Rating друг из друга.
#         # А для субординированных облигаций рейтинг следует брать с сайтов
#     bondS_subordinated = bondS[bondS['Субординированность'] == 'Да']
#     display('bondS_subordinated 1:', bondS_subordinated) # для отладки

#     bondS_unsubordinated = bondS[bondS['Субординированность'] == 'Нет']
#     display('bondS_unsubordinated 1:', bondS_unsubordinated) # для отладки

#     # Поиск эмитентов, у облигаций которых в столбце Bond D Rating (а) ни у одной нет рейтинга,
#         # (б) у некоторых есть рейтинг и у некоторых его нет и (в) у всех есть рейтинг
#     issuerS_fromBonds_fullRating, issuerS_fromBonds_noRating, issuerS_fromBonds_partialRating =\
#         ratingProcessor.bondS_ofIssuer_ratingChecker(bondS_unsubordinated)

#     # print('issuerS_fromBonds_fullRating:', issuerS_fromBonds_fullRating) # для отладки
#     print('issuerS_fromBonds_noRating:', issuerS_fromBonds_noRating) # для отладки
#     print('issuerS_fromBonds_partialRating:', issuerS_fromBonds_partialRating) # для отладки

#     # Выгрузить НЕрасполагаемый рейтинг с сайта moex.ru для столбцов Issuer D Rating и Bond D Rating
#         # для НЕсубординированных облигаций
#     if len(issuerS_fromBonds_noRating) > 0:
#         bondS_unsubordinated_noRating = bondS_unsubordinated[
#             (bondS_unsubordinated['Эмитент'].isin(issuerS_fromBonds_noRating)) & (bondS['ISIN'].notna()) & (bondS['Эмитент'].notna())
#             ]

#         bondS_unsubordinated_noRating, bondS_unsubordinated_noRating_rowS_processed =\
#             ratingProcessor.ratingMoExForBondsWithoutRating(bondS_unsubordinated_noRating, pause)
#                 # NB: Почему-то некоторые облигации не имеют SECNAME

#         # display('bondS_unsubordinated_noRating:', bondS_unsubordinated_noRating) # для отладки

#         bondS_rowS_processed.extend(bondS_unsubordinated_noRating_rowS_processed)

#         bondS_unsubordinated = bondS_unsubordinated.merge(bondS_unsubordinated_noRating[['URL RB', 'Bond D Rating', 'Issuer D Rating']],
#                                                           how="left",
#                                                           on='URL RB',
#                                                           suffixes=("", "_drop"))

#         bondS_unsubordinated['Bond D Rating'] = bondS_unsubordinated['Bond D Rating_drop'].combine_first(bondS_unsubordinated['Bond D Rating'])
#             # замена старых значений новыми только там, где новые не NaN

#         bondS_unsubordinated['Issuer D Rating'] = bondS_unsubordinated['Issuer D Rating_drop'].combine_first(bondS_unsubordinated['Issuer D Rating'])
#             # замена старых значений новыми только там, где новые не NaN

#         bondS_unsubordinated = bondS_unsubordinated.drop(['Bond D Rating_drop', 'Issuer D Rating_drop'], axis=1)
#         display('bondS_unsubordinated 2:', bondS_unsubordinated) # для отладки

#     # Выгрузить НЕрасполагаемый рейтинг с сайта moex.ru для столбцов Issuer D Rating и Bond D Rating
#         # для субординированных облигаций
#     bondS_subordinated = bondS_subordinated[
#         (bondS['ISIN'].notna()) & (bondS['Эмитент'].notna())
#         ]

#     bondS_subordinated, bondS_subordinated_rowS_processed =\
#         ratingProcessor.ratingMoExForBondsWithoutRating(bondS_subordinated, pause, subordinated=True)
#         # NB: Почему-то некоторые облигации не имеют SECNAME

#     display('bondS_subordinated 2:', bondS_subordinated) # для отладки

#     bondS_rowS_processed.extend(bondS_subordinated_rowS_processed)
#     bondS = pandas.concat([bondS_unsubordinated, bondS_subordinated])

# 1.4 Фильтры по датам
    for column in ['MATDATE', 'NEXTCOUPON']:
        bondS = bondS[bondS[column].notna()]
        bondS.loc[bondS[column] == '0000-00-00', column] =\
            bondS.loc[bondS[column] == '0000-00-00', 'SETTLEDATE'] # иначе к столбцу не применяется .astype('datetime64[ns]')

    # Сколько дней до купона?
    bondS = bondS[bondS['MATDATE'] != bondS['SETTLEDATE']] # исключить облигации, по которым погашение уже на след.день
    # bondS = bondS[bondS['NEXTCOUPON'] != bondS['SETTLEDATE']] # исключить облигации, по которым купон уже на след.день
    # display(bondS['MATDATE'].sort_values()) # для отладки
    # display(bondS[['MATDATE', 'NEXTCOUPON', 'SETTLEDATE']].head(50)) # для отладки
    # display(bondS[['MATDATE', 'NEXTCOUPON', 'SETTLEDATE']].tail(50)) # для отладки

    bondS['До купона'] = bondS['NEXTCOUPON'].astype('datetime64[ns]') - bondS['SETTLEDATE'].astype('datetime64[ns]')
    bondS['До купона'] = bondS['До купона'].astype(str)
    bondS['До купона'] = bondS['До купона'].str.split(' ').str[0]
    bondS['До купона'] = bondS['До купона'].astype(int)
    # display(bondS) # для отладки
    
    # Сколько дней до возможности погасить?
    bondS_offer = bondS[bondS['BUYBACKDATE'] != '0000-00-00'] # облигации С офертой
    offerS = bondS_offer.index
    # display(bondS) # для отладки

    # Сколько дней до оферты
    # Вычесть из даты оферты след.день
    bondS_offer['До возможности погасить'] = (bondS_offer['BUYBACKDATE'].astype(str) + '--' + bondS_offer['SETTLEDATE'].astype(str)).apply(lambda text:\
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
    bondS_other['До возможности погасить'] = (bondS_other['MATDATE'].astype(str) + '--' + bondS_other['SETTLEDATE'].astype(str)).apply(lambda text:\
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

# 1.5 Предобрабока столбцов с финансовой информацией
    for column in ['ACCRUEDINT', 'COUPONPERCENT', 'FACEVALUE', 'PRICE']:
        bondS.loc[(bondS[column].notna()) & (bondS[column] != ''), column] = bondS.loc[(bondS[column].notna()) & (bondS[column] != ''), column].astype(float)

    # Умножение FACEVALUE и ACCRUEDINT на цену валюты в рублях
    boardS, columnsDescriptionS, exchangesRaw = getMoExData.getMoExData(market='forts', returnDfs=True)
    exchangesRaw = exchangesRaw[['SHORTNAME', 'LAST', 'SETTLEPRICE']]
    exchangesRaw.columns = ['Unnamed: 0', 'Цена послед.', 'Цена закр.']
    # display(exchangesRaw) # для отладки
    
    # Из QUIK
    # exchangesRaw = pandas(r'C:\Users\Alexey\Dropbox\QUIK_УралСиб_Driver\Текущие_торги.xlsx', usecols='A, D, F')
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

    # display(bondS) # для отладки

# 1.6 Расчёт доходности в день до возможности погасить 
    # Если купить примерно на 1000 единиц валюты, то придётся заплатить
    bondS['Полная цена покупки'] = 1000 + 1000 / bondS['FACEVALUE'] * bondS['ACCRUEDINT']
    bondS['Полная цена покупки'] = bondS['Полная цена покупки'].astype(float).round(2)
    
    # На 1000 единиц валюты к погашению будет начислен купоный доход
    bondS.loc[(bondS['COUPONPERCENT'].isna()) | (bondS['COUPONPERCENT'] == ''), 'Сводный купон'] =\
        bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'Купон RB']
    bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'Сводный купон'] =\
        bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'COUPONPERCENT']
    
    bondS['Купоный доход к погашению'] = 1000 * bondS['Сводный купон'] / 36500 * bondS['До возможности погасить']
    bondS['Купоный доход к погашению'] = bondS['Купоный доход к погашению'].astype(float).round(2)
    
    # Плюс 1000 единиц валюты изменятся к погашению в связи с приведением цены к номиналу
    bondS['Маржа к погашению'] = 1000 * (100 - bondS['PRICE']) / 100
    bondS['Маржа к погашению'] = bondS['Маржа к погашению'].astype(float).round(2)
    
    # Подневная доходность к погашению = суммарный доход к полной цене покупки
    bondS['% доходности в день к погашению'] =\
        (
        100 * (1000 + bondS['Купоный доход к погашению'] + bondS['Маржа к погашению'])\
        / bondS['Полная цена покупки'] - 100\
        ) / bondS['До возможности погасить']
    bondS['% доходности в день к погашению'] = bondS['% доходности в день к погашению'].astype(float).round(4)

    # !!! Стоимость!!!
    if 'Лотов' in bondS.columns: bondS['Стоимость'] = bondS['Лотов'] * bondS['PRICE'] * 10 * bondS['FACEVALUE'] / 1000
        # если поданы на вход облигации из портфеля (уже купленные)
    # display(bondS) # для отладки

# 1.7 Интегральная переменная Специфика
    bondS.loc[(bondS['Сектор рынка'] == 'Гос') | (bondS['SECNAME'].str.contains('ОФЗ|Россия', case=False)), 'Сектор рынка'] = 'Гос'
    bondS.loc[(bondS['Сектор рынка'].str.contains('Гос|Корп|Мун', case=False) != True), 'Сектор рынка'] = 'Корп'
    bondS.loc[(bondS['COUPONPERCENT'].isna()) | (bondS['COUPONPERCENT'] == ''), 'Купон определён'] = 0
    bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'Купон определён'] = 1
    # display("bondS['Купон определён']:", bondS['Купон определён'].value_counts()) # для отладки
    
    bondS['Специфика'] = bondS['FACEUNIT'].str[:2]
    for column in ['Сектор рынка', 'Амортизация FinAm' if 'Амортизация FinAm' in bondS.columns else 'Амортизация T', 'Купон определён']:
        bondS[column] = bondS[column].fillna('--')
        bondS['Специфика'] += ' ' + bondS[column].astype(str).str[:1]

    display(bondS['Специфика'].value_counts().sort_index())

    if returnDfs: return bondS, bondS_rowS_processed, issuerS_withActualRating, issuerS_withActualRating_change
