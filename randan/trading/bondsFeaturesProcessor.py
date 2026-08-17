# Модуль для гармонизации и обработки характеристик облигаций

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from datetime import date, datetime, timedelta
        from IPython.display import display

        from randan.trading import finamParser, getMoExData, issuerProcessor, ratingProcessor # авторские модули для
            # (а) упрощения выгрузки данных с сайта finam.ru и их парсинга
            # (б) выгрузки характеристик торгуемых на МосБирже облигаций
            # (в) операций с эмитентами торгуемых на МосБирже облигаций
            # (г) упрощения некоторых оперций в selenium

        from randan.tools import cellsLeftMerger, coLabAdaptor, files2df # авторские модули для
            # (а) упрощения операции левостороннего присоединения датафрейма-донора к датафрейму-реципиенту по специальному столбцу
            # (б) адаптации текущего скрипта к файловой системе CoLab
            # (в) оформления в датафрейм таблиц из файлов формата CSV, Excel и JSON в рамках работы с данными из социальных медиа

        from selenium.webdriver.common.by import By # для поиска элементов HTML-кода
        from tqdm import tqdm
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
        if attempt == 3:
            print(
f'''Пакет {module} НЕ прединсталлирован; он требуется для работы скрипта, но инсталлировать его не удаётся,
поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт
'''
                  )

coLabFolder = coLabAdaptor.coLabAdaptor()

# 1. Вспомогательные функции для..
# .. расчёта доходностей облигации (бескупонной, без реинвестирования и с реинвестированием -- по формулам простого и сложного процентов)
def bondYieldCalculator(bond_df_in, bond_df_index, df_current, driver_CB, momentCurrent):
    bond_df = bond_df_in.copy()
    # display('bond_df 1 в bondYieldCalculator :', bond_df) # для отладки

    if len(df_current) > 1:
        if sum(df_current['Ставка'].notna()) == 0: # в качестве прокси взять ставку ЦБ

            table = driver_CB.find_element(By.XPATH,
                                           "//div[@class='table-caption gray' and text()='% годовых']/following-sibling::div[@class='table']//table")
                # найти таблицу, которая идёт после заголовка "% годовых"

            rowTagert = table.find_elements(By.XPATH, ".//tr")[1] # первая строка после заголовка
            rate_CB = rowTagert.find_elements(By.XPATH, ".//td")[1].text # вторая ячейка слева в этой строке
            rate_CB = float(rate_CB.replace(',', '.'))
            print('rate_CB:', rate_CB) # для отладки

            df_current.loc[df_current.index[0], 'Ставка'] = rate_CB

    if len(df_current) > 1 & (sum(df_current['Ставка'].notna()) == 1): df_current['Ставка'] = df_current.loc[df_current.index[0], 'Ставка']
        # экстраполяция, исходя из допущения неизменности ставки на весь период до оферты | погашения
            # -- иначе несравнимы облигации с плавающим текущим купоном и с фиксированным текущим купоном

    # df_current['Ставка'] = 0 # альтернатива для отладки
    # display('df_current 1:', df_current) # для отладки

    bond_df.loc[bond_df_index, 'Амортизация Current'] = 0
    if sum((df_current['% от Номинала'].notna()) & (df_current['Размер (ден)'].notna())) > 0:
        print('Похоже, есть амортизации в рамках периода от сегоднящней даты до оферты | погашения') # для отладки

        if df_current.loc[df_current.index[-1], '% от Номинала'] != 100:
            print('И это не итоговое погашение') # для отладки
            bond_df.loc[bond_df_index, 'Амортизация Current'] = 1

        else: print('Но это итоговое погашение') # для отладки

    # display('bond_df 2 в bondYieldCalculator :', bond_df) # для отладки
    price_coefficient = bond_df['PRICE'][bond_df_index]
    print('price_coefficient:', price_coefficient)
    print(
'''Если price_coefficient > 100, то рассчитываемая для следующих купонных периодов сумма для реинвестирования
уменьшается за счёт потери от разницы между рыночной ценой и остаточным номиналом'''
        )

    # Погашаемый % от номинала, нормированный в рамках периода от сегодняшней даты до оферты | погашения
    df_current.loc[df_current['% от Номинала'].isna(), '% от Номинала'] = 0 # чтобы считать '% от Номинала НРМРВНН'
    df_current.loc[df_current.index[-1], '% от Номинала'] = 100 - df_current.loc[df_current.index[:-1], '% от Номинала'].sum()
        # чтобы при оферте считать полное погашение

    df_current.loc[:, '% от Номинала НРМРВНН'] = 100 * df_current.loc[:, '% от Номинала'] / df_current['% от Номинала'].sum()
    # df_current.loc[:, '% от Номинала НРМРВНН'] = 0 # альтернатива для отладки
    # df_current.loc[df_current.index[-1], '% от Номинала НРМРВНН'] = 100 # альтернатива для отладки
    # display('df_current 2:', df_current) # для отладки

    # Добавить строчку № -1 , в которую внести сегодяншнюю дату и рыночную цену (в %) покупки облигации
    df_current.loc[-1, 'Дата'] = momentCurrent.date()
    df_current.loc[-1, '% от Номинала НРМРВНН'] = 0
    df_current.loc[-1, 'Остаточный номинал на конец периода'] = bond_df['FACEVALUE'][bond_df_index]
    df_current.loc[-1, 'Остаточный номинал на конец периода РИ'] = bond_df['FACEVALUE'][bond_df_index] # РИ -- реинвестирование

    df_current = df_current.sort_index()
    df_current['Дата'] = pandas.to_datetime(df_current['Дата'])

    price = (bond_df['FACEVALUE'] * (bond_df['PRICE'] / 100))[bond_df_index] # остаточный номинал * роночную цену (в %)
        # остаточный номинал = обращающееся тело долга

    price_dirty = price + bond_df['ACCRUEDINT'][bond_df_index] # + НКД
    df_current.loc[-1, 'Цена покупки'] = price_dirty
    df_current['Цена покупки'] = df_current['Цена покупки'].astype(float).round(2)
    # display('df_current 3:', df_current) # для отладки

    for df_current_row in range(0, len(df_current) - 1):
    # for df_current_row in range(0, 2): # для отладки
        # print('df_current_row:', df_current_row) # для отладки

    # Без реинвестирования

        # Купон начисляется на тело, находящееся в строчке выше;
            # амортизация вычитается из тела, находящегося в строчке выше
            # на основе самого исходного тела. Отличие от РИ

        df_current.loc[df_current_row, 'Погашение в купонный период'] =\
            df_current.loc[-1, 'Остаточный номинал на конец периода'] *\
            (df_current.loc[df_current_row, '% от Номинала НРМРВНН'] / 100)

        df_current.loc[df_current_row, 'Дней в купонном периоде'] =\
            (df_current.loc[df_current_row, 'Дата'] - df_current.loc[df_current_row - 1, 'Дата']).days

        df_current.loc[df_current_row, 'Купонный доход за купонный период'] =\
            df_current.loc[df_current_row - 1, 'Остаточный номинал на конец периода'] *\
            df_current.loc[df_current_row, 'Ставка'] / 36500 *\
            df_current.loc[df_current_row, 'Дней в купонном периоде']

        if df_current_row == 0: df_current.loc[df_current_row, 'Купонный доход за купонный период'] += bond_df.loc[bond_df_index, 'ACCRUEDINT']
            # учесть НКД

        df_current.loc[df_current_row, 'Полный доход за купонный период'] =\
            df_current.loc[df_current_row, 'Купонный доход за купонный период'] + df_current.loc[df_current_row, 'Погашение в купонный период']

        # На основе тела, находящегося в строчке выше, а также амортизации в текущей строчке формируется тела, находящееся в текущей строчке

        df_current.loc[df_current_row, 'Остаточный номинал на конец периода'] =\
            df_current.loc[df_current_row - 1, 'Остаточный номинал на конец периода'] -\
            df_current.loc[df_current_row, 'Погашение в купонный период']

    # С реинвестированием

        # Купон начисляется на тело, находящееся в строчке выше;
            # амортизация вычитается из тела, находящегося в строчке выше,
            # на основе тела, находящегося в строчке выше и пересчитываемого на каждой итерации нормированного процента. Отличие от не РИ

        df_current.loc[df_current_row, '% от Номинала НРМРВНН РИ'] =\
            (100 * df_current.loc[df_current_row:, '% от Номинала НРМРВНН'] /\
            df_current.loc[df_current_row: ,'% от Номинала НРМРВНН'].sum())\
                [df_current_row] # пересчёт % нормированного для всех остающихся строк df_current и использование значения из верхней

        df_current.loc[df_current_row, 'Погашение в купонный период РИ'] =\
            df_current.loc[df_current_row - 1, 'Остаточный номинал на конец периода РИ'] *\
            (df_current.loc[df_current_row, '% от Номинала НРМРВНН РИ'] / 100)

        df_current.loc[df_current_row, 'Купонный доход за купонный период РИ'] =\
            df_current.loc[df_current_row - 1, 'Остаточный номинал на конец периода РИ'] *\
            df_current.loc[df_current_row, 'Ставка'] / 36500 *\
            df_current.loc[df_current_row, 'Дней в купонном периоде']

        if df_current_row == 0: df_current.loc[df_current_row, 'Купонный доход за купонный период РИ'] += bond_df.loc[bond_df_index, 'ACCRUEDINT']
            # учесть НКД

        if df_current_row == 0: # на этой итерации Сумму РИ формируется впервые, поэтому нет купона на Сумму РИ
            df_current.loc[df_current_row, 'Дней в купонном периоде для РИ'] = 0
            df_current.loc[df_current_row, 'Купонный доход на Сумму для РИ за купонный период'] = 0

        if df_current_row > 0:

            df_current.loc[df_current_row, 'Дней в купонном периоде для РИ'] =\
                df_current.loc[df_current_row, 'Дней в купонном периоде'] - 3
                    # 3 дня теряются на получение инветсором выплаты и реинвесирование (в ту же облигацию)

            df_current.loc[df_current_row, 'Купонный доход на Сумму для РИ за купонный период'] =\
                df_current.loc[df_current_row - 1, 'Сумма для РИ'] *\
                df_current.loc[df_current_row, 'Ставка'] / 36500 *\
                df_current.loc[df_current_row, 'Дней в купонном периоде для РИ']
                    # 3 дня теряются на получение инветсором выплаты и реинвесирование (в ту же облигацию)

        df_current.loc[df_current_row, 'Полный доход за купонный период РИ'] =\
            df_current.loc[df_current_row, 'Купонный доход за купонный период РИ'] +\
            df_current.loc[df_current_row, 'Погашение в купонный период РИ'] +\
            df_current.loc[df_current_row, 'Купонный доход на Сумму для РИ за купонный период']

        df_current.loc[df_current_row, 'Сумма для РИ'] =\
            df_current.loc[df_current_row, 'Полный доход за купонный период РИ'] * (100 / price_coefficient) # экстраполяция
                # ситуации отклонения рыночной цены от остающегося в обращении номинала,
                    # характерной для первой покупки, на все покупки реинвестирования

        # На основе тела, находящегося в строчке выше, а также амортизации в текущей строчке формируется тела, находящееся в текущей строчке

        if df_current_row < df_current.index[-1]:
            df_current.loc[df_current_row, 'Остаточный номинал на конец периода РИ'] =\
                df_current.loc[df_current_row - 1, 'Остаточный номинал на конец периода РИ'] -\
                df_current.loc[df_current_row, 'Погашение в купонный период РИ'] +\
                df_current.loc[df_current_row, 'Сумма для РИ']

        else: # на последней итерации нет смысла в РИ
            df_current.loc[df_current_row, 'Остаточный номинал на конец периода РИ'] =\
                df_current.loc[df_current_row - 1, 'Остаточный номинал на конец периода РИ'] -\
                df_current.loc[df_current_row, 'Погашение в купонный период РИ']

    display('head:', df_current.head()) # для отладки
    display('tail:', df_current.tail()) # для отладки

    bond_df.loc[bond_df_index, 'Купонный доход к погашению'] = df_current['Купонный доход за купонный период'].sum()

    print("bond_df.loc[bond_df_index, 'Купонный доход к погашению']:",
            bond_df.loc[bond_df_index, 'Купонный доход к погашению']) # для отладки


    period_total = df_current['Дней в купонном периоде'].sum()
    print('period_total:', period_total) # для отладки


    income_redemption = df_current['Погашение в купонный период'].sum()
    print('income_redemption:', income_redemption) # для отладки

    bond_df.loc[bond_df_index, 'Доходность бескупонная, годовых'] = 365 * ((income_redemption - price_dirty) / price_dirty) / period_total

    print("bond_df.loc[bond_df_index, 'Доходность бескупонная, годовых']:",
          bond_df.loc[bond_df_index, 'Доходность бескупонная, годовых']) # для отладки


    income_noReinvestment = df_current['Полный доход за купонный период'].sum()
    print('income_noReinvestment:', income_noReinvestment) # для отладки

    bond_df.loc[bond_df_index, 'Доходность без реинвестирования, годовых'] =\
        365 * ((income_noReinvestment - price_dirty) / price_dirty) / period_total

    print("bond_df.loc[bond_df_index, 'Доходность без реинвестирования, годовых']:",
          bond_df.loc[bond_df_index, 'Доходность без реинвестирования, годовых']) # для отладки


    income_Reinvestment = df_current.loc[df_current.index[-1], 'Полный доход за купонный период РИ']
    print('income_Reinvestment:', income_Reinvestment) # для отладки

    bond_df.loc[bond_df_index, 'Доходность с реинвестированием, годовых, простой процент'] =\
        365 * ((income_Reinvestment - price_dirty) / price_dirty) / period_total

    bond_df.loc[bond_df_index, 'Доходность с реинвестированием, годовых, сложный процент'] =\
        (income_Reinvestment / price_dirty) ** (365 / period_total) - 1

    print("bond_df.loc[bond_df_index, 'Доходность с реинвестированием, годовых, простой процент']:",
          bond_df.loc[bond_df_index, 'Доходность с реинвестированием, годовых, простой процент']) # для отладки

    print("bond_df.loc[bond_df_index, 'Доходность с реинвестированием, годовых, сложный процент']:",
          bond_df.loc[bond_df_index, 'Доходность с реинвестированием, годовых, сложный процент']) # для отладки

    df_current = df_current[:].round(2)
    return bond_df, df_current

# .. расчёта эффекта валютных курсов для иновалютных облигаций
def currencyEffectProcessor(bondS_in, currencieS):
    bondS = bondS_in.copy()

# <Умножение FACEVALUE и ACCRUEDINT для иновалютных облигаций на цену соответствующей валюты в рублях>

    # Импорт курсов инвалют
    boardS, columnsDescriptionS, exchangesRaw = getMoExData.getMoExData(market='forts', returnDfs=True)
    exchangesRaw = exchangesRaw[['SHORTNAME', 'LAST', 'SETTLEPRICE']]
    exchangesRaw.columns = ['Unnamed: 0', 'Цена послед.', 'Цена закр.']
    # display(exchangesRaw) # для отладки

    # Список валют иновалютных облигаций и запись их курсов в exchangeS
    exchangeS = pandas.DataFrame()
    for currency in currencieS:
    # for currency in currencieS[0:1]: # для отладки
        exchangesAdditional = exchangesRaw[exchangesRaw['Unnamed: 0'].str.contains(currency, case=False)]
        # display('exchangesAdditional:', exchangesAdditional) # для отладки

        if len(exchangesAdditional) > 1: exchangesAdditional = exchangesAdditional.iloc[[0], :] # чтобы не брать пару USD|CNY
        # display('exchangesAdditional:', exchangesAdditional) # для отладки

        exchangesAdditional['Валюта'] = currency
        exchangeS = pandas.concat([exchangeS, exchangesAdditional])

    # Предобрабока столбцов с финансовой информацией в exchangeS
    for column in ['Цена послед.', 'Цена закр.']:
        exchangeS[column] = exchangeS[column].astype(float)

    display('exchangeS:', exchangeS[['Цена послед.', 'Цена закр.', 'Валюта']]) # для отладки

    exchangeS.loc[exchangeS['Цена послед.'] == 0, 'Цена послед.'] = exchangeS.loc[exchangeS['Цена послед.'] == 0, 'Цена закр.']
        # на случай нулей в столбце 'Цена послед.'

    exchangeS = exchangeS.drop(['Unnamed: 0', 'Цена закр.'], axis=1)

    # Поскольку исходно CHF в паре с USD
    if (exchangeS['Валюта'] == 'CHF').sum() > 0:
        exchangeS.loc[exchangeS['Валюта'] == 'CHF', 'Цена послед.'] =\
            exchangeS.loc[exchangeS['Валюта'] == 'USD', 'Цена послед.'][exchangeS[exchangeS['Валюта'] == 'USD'].index[0]] /\
            exchangeS.loc[exchangeS['Валюта'] == 'CHF', 'Цена послед.'][exchangeS[exchangeS['Валюта'] == 'CHF'].index[0]]

    exchangeS = exchangeS.sort_values('Валюта').reset_index(drop=True)
    # display('exchangeS:', exchangeS) # для отладки

    for currency in currencieS:
        currencyExchangeValue = exchangeS.loc[exchangeS['Валюта'] == currency, 'Цена послед.'][exchangeS[exchangeS['Валюта'] == currency].index[0]]
        # print('currencyExchangeValue:', currencyExchangeValue) # для отладки
        # print('type(currencyExchangeValue):', type(currencyExchangeValue)) # для отладки

        bondS.loc[bondS['FACEUNIT'] == currency, 'FACEVALUE'] *= currencyExchangeValue # 'FACEUNIT' -- внутренняя валюта облигации
        bondS.loc[bondS['CURRENCYID'] == currency, 'ACCRUEDINT'] *= currencyExchangeValue # 'CURRENCYID' -- валюта расчётов

    # display(bondS) # для отладки
# <\Умножение FACEVALUE и ACCRUEDINT для иновалютных облигаций на цену соответствующей валюты в рублях>

    return bondS

# 2. Основная функция
def bondsFeaturesProcessor(attemptsMax,
                           bondsIn,
                           driver,
                           driver_CB,
                           driver_TB,
                           issuerS,
                           momentCurrent,
                           pause,
                           version_main,
                           folder=coLabFolder,
                           returnDfs=False):
    """
    Функция для выяснения, какие облигации есть в портфеле, на основе брокерских отчётов

    Parameters
    ----------
      bondsIn : DataFrame -- датафрейм с облигациями, характеристики которых требуется получить; должен содержать хотя бы столбец ISIN
      issuerS : DataFrame -- датафрейм со Словарём эмитентов
        pause : int -- период засыпания исполнения функций selenium
       folder : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться;
                     по умолчанию, не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

    returnDfs : bool -- в случае True функция возвращает итоговые датафрейм bondS
    """

# 2.0 Настройки
    bondS = bondsIn.copy()
    bondS = bondS.drop_duplicates('ISIN', keep='last', ignore_index=True)

    # Блок, поскольку folder многократно используется внутри функции в формулах
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if folder == None: folder = ''
    else: folder += slash

    warnings.filterwarnings("ignore")

# 2.1 Добавить характеристики облигаций из БД МосБиржи
    boardS, columnsDescriptionS, securitieS = getMoExData.getMoExData(market='bonds', returnDfs=True)
    bondS = bondS.merge(securitieS, how='left', on='ISIN', suffixes=('_drop', '')) # дропнуть старые столбцы, оставить новые
    bondS = bondS[[column for column in bondS.columns if not column.endswith('_drop')]]
    # print('bondS.columns:', bondS.columns) # для отладки

# При отсутствии столбца Эмитент добавить его посредством функции issuerNameProcessor
    if 'Эмитент' not in bondS.columns:
        bondS = issuerProcessor.issuerNameProcessor(bondS, issuerS)
            # теперь в bondS у каждой облигации указан эмитент с названием, соотнесённым со Словарём эмитентов
                # (все эти эмитенты представлены в issuerS)

# 2.2 Импорт словарей (актуального и прошлого) эмитентов с рейтингом из Акуальные эмитенты.xlsx в issuerS_withActualRating
    # print('folder:', folder) # для отладки
    if os.path.exists(folder + 'Замеры рейтингов') != True:
        print('Найдите и запустите скрипт bondsRatingS')
        issuerS_withActualRating = pandas.DataFrame()

    fileUptodateName_0 = files2df.getFileUptodateName('_Акуальные эмитенты', None, folder + 'Замеры рейтингов')
    # print('fileUptodateName_0:', fileUptodateName_0) # для отладки

    issuerS_withActualRating = pandas.read_excel(folder + 'Замеры рейтингов' + slash + fileUptodateName_0)
    # display('issuerS_withActualRating:', issuerS_withActualRating) # для отладки

    fileUptodateName_1 = files2df.getFileUptodateName('_Акуальные эмитенты', [fileUptodateName_0], folder + 'Замеры рейтингов')
    # print('fileUptodateName_1:', fileUptodateName_1) # для отладки

    issuerS_withActualRating_previous = pandas.read_excel(folder + 'Замеры рейтингов' + slash + fileUptodateName_1)
    # display('issuerS_withActualRating_previous:', issuerS_withActualRating_previous) # для отладки

    issuerS_withActualRating = issuerS_withActualRating.merge(issuerS_withActualRating_previous[['Эмитент', 'Issuer D Rating']], how='left', on='Эмитент', suffixes=("", " Previous"))
    # display('issuerS_withActualRating:', issuerS_withActualRating) # для отладки

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

    bondS = bondS.merge(issuerS_withActualRating[['Эмитент', 'Issuer D Rating']], how="left", on='Эмитент', suffixes=("_drop", ""))
    bondS = bondS[[column for column in bondS.columns if not column.endswith("_drop")]]

    # display('bondS:', bondS) # для отладки

# При отсутствии столбца Issuer D Rating добавить его, "расшерив" рейтинг из issuerS_withActualRating # и выгрузка с сайта moex.ru
    for issuer_withActualRating in issuerS_withActualRating['Эмитент']:
        # print('issuer_withActualRating:', issuer_withActualRating) # для отладки
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

#         bondS_unsubordinated_noRating = ratingProcessor.ratingMoExForBondsWithoutRating(bondS_unsubordinated_noRating, pause)
#                 # NB: Почему-то некоторые облигации не имеют SECNAME

#         # display('bondS_unsubordinated_noRating:', bondS_unsubordinated_noRating) # для отладки

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

#     bondS_subordinated = ratingProcessor.ratingMoExForBondsWithoutRating(bondS_subordinated, pause, subordinated=True)
#     display('bondS_subordinated 2:', bondS_subordinated) # для отладки

#     bondS = pandas.concat([bondS_unsubordinated, bondS_subordinated])

# 2.2 Фильтры по датам
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

# 2.3 Расчёт эффекта валютных курсов для иновалютных облигаций

    # Предобрабока столбцов с финансовой информацией в bondS
    for column in ['ACCRUEDINT', 'COUPONPERCENT', 'FACEVALUE', 'PRICE']:

        bondS.loc[(bondS[column].notna()) & (bondS[column] != ''), column] =\
            bondS.loc[(bondS[column].notna()) & (bondS[column] != ''), column].astype(float)

    currencieS = list(bondS['FACEUNIT'].unique()) # валюта номинала
    print('currencieS:', currencieS) # для отладки

    currencieS.remove('SUR')
    if len(currencieS) > 0: bondS = currencyEffectProcessor(bondS, currencieS)

    # for column in ['ACCRUEDINT', 'COUPONPERCENT', 'FACEVALUE', 'PRICE']:
    #     bondS.loc[(bondS[column].notna()) & (bondS[column] != ''), column] = bondS.loc[(bondS[column].notna()) & (bondS[column] != ''), column].astype(float)

    # # Умножение FACEVALUE и ACCRUEDINT на цену валюты в рублях
    # boardS, columnsDescriptionS, exchangesRaw = getMoExData.getMoExData(market='forts', returnDfs=True)
    # exchangesRaw = exchangesRaw[['SHORTNAME', 'LAST', 'SETTLEPRICE']]
    # exchangesRaw.columns = ['Unnamed: 0', 'Цена послед.', 'Цена закр.']
    # # display(exchangesRaw) # для отладки

    # # Из QUIK
    # # exchangesRaw = pandas(r'C:\Users\Alexey\Dropbox\QUIK_УралСиб_Driver\Текущие_торги.xlsx', usecols='A, D, F')
    # # display(exchangesRaw) # для отладки

    # currencieS = list(bondS['FACEUNIT'].unique()) # валюта номинала
    # print('currencieS:', currencieS) # для отладки
    # currencieS.remove('SUR')
    # exchangeS = pandas.DataFrame()

    # for currency in currencieS:
    # # for currency in currencieS[0:1]: # для отладки
    #     exchangesAdditional = exchangesRaw[exchangesRaw['Unnamed: 0'].str.contains(currency, case=False)]
    #     # display('exchangesAdditional:', exchangesAdditional) # для отладки
    #     if len(exchangesAdditional) > 1: exchangesAdditional = exchangesAdditional.iloc[[0], :] # чтобы не брать пару USD|CNY
    #     # display('exchangesAdditional:', exchangesAdditional) # для отладки
    #     exchangesAdditional['Валюта'] = currency
    #     exchangeS = pandas.concat([exchangeS, exchangesAdditional])

    # for column in ['Цена послед.', 'Цена закр.']:
    #     exchangeS[column] = exchangeS[column].astype(float)

    # display('exchangeS:', exchangeS[['Цена послед.', 'Цена закр.', 'Валюта']]) # для отладки

    # exchangeS.loc[exchangeS['Цена послед.'] == 0, 'Цена послед.'] = exchangeS.loc[exchangeS['Цена послед.'] == 0, 'Цена закр.'] # на случай нулей в столбце 'Цена послед.'
    # exchangeS = exchangeS.drop(['Unnamed: 0', 'Цена закр.'], axis=1)

    # # Поскольку исходно CHF в паре с USD
    # if (exchangeS['Валюта'] == 'CHF').sum() > 0:
    #     exchangeS.loc[exchangeS['Валюта'] == 'CHF', 'Цена послед.'] =\
    #         exchangeS.loc[exchangeS['Валюта'] == 'USD', 'Цена послед.'][exchangeS[exchangeS['Валюта'] == 'USD'].index[0]]\
    #         / exchangeS.loc[exchangeS['Валюта'] == 'CHF', 'Цена послед.'][exchangeS[exchangeS['Валюта'] == 'CHF'].index[0]]

    # exchangeS = exchangeS.sort_values('Валюта').reset_index(drop=True)
    # # display('exchangeS:', exchangeS) # для отладки

    # for currency in currencieS:
    #     currencyExchangeValue = exchangeS.loc[exchangeS['Валюта'] == currency, 'Цена послед.'][exchangeS[exchangeS['Валюта'] == currency].index[0]]
    #     # print('currencyExchangeValue:', currencyExchangeValue) # для отладки
    #     # print('type(currencyExchangeValue):', type(currencyExchangeValue)) # для отладки
    #     bondS.loc[bondS['FACEUNIT'] == currency, 'FACEVALUE'] *= currencyExchangeValue
    #     bondS.loc[bondS['CURRENCYID'] == currency, 'ACCRUEDINT'] *= currencyExchangeValue # валюта расчётов

# 2.4 Расчёт годовой доходности до оферты | погашения
    for isin in tqdm(bondS['ISIN']):
    # for isin in tqdm(bondS['ISIN'][674:]): # для отладки
        print('\nisin:', isin) # для отладки
        bond_df = bondS[bondS['ISIN'] == isin]
        # display('bond_df:', bond_df) # для отладки

        bond_df_index = bond_df.index[0]
        # print('bond_df_index:', bond_df_index) # для отладки

        # print('folder:', folder) # для отладки
        path_1 = folder + 'Таблицы FinAM'
        if os.path.exists(path_1) != True:
            print(
    '''Найдите и запустите скрипт bondsRatingS, после чего снова запустите текущий скрипт
    А сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть'''
                )
            input()
            sys.exit()

        fileUptodateName = files2df.getFileUptodateName(isin, None, path_1)
        # print('fileUptodateName:', fileUptodateName) # для отладки

    # <Сравнение текущего момента и рекомендованной повторной даты выгрузки информации с FinAM; при необходимости, новая выгрузка>
        if fileUptodateName: 
            date_call = fileUptodateName.split(' ')[0]
            date_call = datetime.strptime(date_call, '%Y%m%d') if date_call != 'No' else momentCurrent
                # else momentCurrent -- заглушка; нужна для работы условия momentCurrent > date_call

            # print('date_call:', date_call) # для отладки

        else: date_call = momentCurrent - timedelta(days=2) # заглушка; нужна для работы условия momentCurrent > date_call

        if momentCurrent > date_call: # если текущий момент оставил позади рекомендованную повторную дату выгрузки информации с FinAM

            print('Требуется новая выгрузка с FinAM')

            bondsFinAM = pandas.DataFrame(columns=[
                'Эмитент',
                'ISIN', 
                'Амортизация FinAm',
                'Описание платежей',
                'Спред',
                'URL FinAM',
                'SecName FinAM',
                'REGNUMBER FinAM'
                ])

            bondsFinAM, counter, driver, driver_TB, goS = finamParser.finamParser(attemptsMax,
                                                                                  bondsFinAM,
                                                                                  bond_df,
                                                                                  'ISIN',
                                                                                  driver,
                                                                                  driver_TB,
                                                                                  momentCurrent,
                                                                                  pause,
                                                                                  bond_df, # список ISIN или эмитентов
                                                                                  # новым эмитентам и рейтингам на этом этапе появиться неоткуда

                                                                                  version_main)

            bondsFinAM.loc[:, 'Момент обращения к FinAM'] = momentCurrent.strftime('%Y%m%d_%H%M')
            display('bondsFinAM:', bondsFinAM) # для отладки

            display('bondsFinAM:', bondsFinAM) # для отладки
            bondsFinAM.to_excel(folder + 'Замеры рейтингов' + slash + momentCurrent.strftime('%Y%m%d_%H%M') + '_bondsFinAM.xlsx', index=False)
                # на случай ошибки

            # Следует мёрджить по ISIN , причём требуется не обновление данных, а дополнение, поэтому cellsLeftMerger
            bond_df = cellsLeftMerger.cellsLeftMerger(bondsFinAM[
                                                        ['Эмитент',
                                                        'ISIN',
                                                        'URL FinAM',
                                                        'SecName FinAM',
                                                        'REGNUMBER FinAM',
                                                        'Момент обращения к FinAM',
                                                        'Амортизация FinAm',
                                                        'Описание платежей',
                                                        'Спред']
                                                        ],
                                                      bond_df,
                                                      'ISIN') # следует мёрджить по ISIN

            fileUptodateName = files2df.getFileUptodateName(isin, None, path_1)
            # print('fileUptodateName:', fileUptodateName) # для отладки
    # <\Сравнение текущего момента и рекомендованной повторной даты выгрузки информации с FinAM; при необходимости, новая выгрузка>

        table_FinAM = pandas.read_excel(path_1 + slash + fileUptodateName, header=[0, 1], index_col=0)
        # display('table_FinAM:', table_FinAM) # для отладки

        for table_FinAM_column in table_FinAM.columns: # на всякий случай, воспроизведение предобработки из finamParser.getTableByURL_FinAM
            if table_FinAM[table_FinAM_column].dtype == 'object': # только текстовые столбцы
                table_FinAM[table_FinAM_column] = table_FinAM[table_FinAM_column].str.replace('RUR', '', regex=False).str.strip()
                table_FinAM[table_FinAM_column] = table_FinAM[table_FinAM_column].str.replace(',', '.')
                table_FinAM[table_FinAM_column] = pandas.to_numeric(table_FinAM[table_FinAM_column], errors='ignore')

        if bond_df.loc[bond_df_index, 'BUYBACKDATE'] != '0000-00-00': # есть оферта
            # print("bond_df.loc[bond_df_index, 'BUYBACKDATE'] != '0000-00-00'") # для отладки   
            date_final = bond_df.loc[bond_df_index, 'BUYBACKDATE']

        elif bond_df.loc[bond_df_index, 'MATDATE'] != '0000-00-00': # есть конечная дата обращения
            # print("bond_df.loc[bond_df_index, 'MATDATE'] != '0000-00-00'") # для отладки   
            date_final = bond_df.loc[bond_df_index, 'MATDATE']

        else: # нет оферты и нет конечной даты обращения
            # print('Нет оферты и нет конечной даты обращения') # для отладки   
            date_final = table_FinAM.loc[table_FinAM.index[-1], (             'Купоны',                'Дата')].strftime('%Y-%m-%d')

        date_final = datetime.strptime(date_final, '%Y-%m-%d').date()
        print('date_final:', date_final) # для отладки

        df_current = table_FinAM[table_FinAM[(             'Купоны',                'Дата')].dt.date >= momentCurrent.date()] # фильтр дата >= сегодняшней
        df_current = df_current[df_current[(             'Купоны',                'Дата')].dt.date <= date_final] # фильтр дата <= date_final
        df_current = df_current.sort_values((             'Купоны',                'Дата'))
        df_current = df_current.drop([(   'Купоны', '% от Номинала'), (   'Купоны',  'Размер (ден)')], axis=1) # столбцы лишние
        df_current.columns = df_current.columns.droplevel(0) # MultiIndex -> обычные заголовки
        df_current = df_current.reset_index(drop=True)
        # display('df_current:', df_current) # для отладки

        if len(df_current) > 0:
            bond_df, df_current = bondYieldCalculator(bond_df, bond_df_index, df_current, driver_CB, momentCurrent)
            if sum(df_current['Ставка'].notna()) > 0:
                date_call = df_current.loc[df_current[df_current['Ставка'].notna()].index[-1], 'Дата'].date().strftime('%Y%m%d')
                print('date_call:', date_call) # для отладки

            else: date_call = 'No rate'

            path_2 = folder + 'Таблицы текущего периода'
            if os.path.exists(path_2) != True: os.makedirs(path_2)
            df_current.to_excel(path_2 + slash + f'{date_call + ' ' if date_call else ''}{isin}.xlsx')

        else:
            bond_df.loc[bond_df_index, 'Не в обращении'] = 1
            path_3 = path_1 + slash + 'Таблицы FinAM Архив'
            if os.path.exists(path_3) != True: os.makedirs(path_3)
            os.rename(path_1 + slash + fileUptodateName, path_3 + slash + fileUptodateName)

        # Следует мёрджить по ISIN , причём требуется не обновление данных, а дополнение, поэтому cellsLeftMerger
        bondS = cellsLeftMerger.cellsLeftMerger(bond_df, bondS, 'ISIN') # следует мёрджить по ISIN
        bondS.to_excel(folder + 'Замеры рейтингов' + slash + momentCurrent.strftime('%Y%m%d_%H%M') + '_bondS.xlsx')

    bondS.to_excel(folder + 'Замеры рейтингов' + slash + momentCurrent.strftime('%Y%m%d_%H%M') + '_bondS.xlsx')



    # # Если купить примерно на 1000 единиц валюты, то придётся заплатить
    # bondS['Полная цена покупки'] = 1000 + 1000 / bondS['FACEVALUE'] * bondS['ACCRUEDINT']
    # bondS['Полная цена покупки'] = bondS['Полная цена покупки'].astype(float).round(2)

    # # На 1000 единиц валюты к погашению будет начислен купоный доход
    # if 'Купон RB' in bondS.columns:
    #     bondS.loc[(bondS['COUPONPERCENT'].isna()) | (bondS['COUPONPERCENT'] == ''), 'Сводный купон'] =\
    #         bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'Купон RB']

    # bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'Сводный купон'] =\
    #     bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'COUPONPERCENT']

    # bondS['Купоный доход к погашению'] = 1000 * bondS['Сводный купон'] / 36500 * bondS['До возможности погасить']
    # bondS['Купоный доход к погашению'] = bondS['Купоный доход к погашению'].astype(float).round(2)

    # # Плюс 1000 единиц валюты изменятся к погашению в связи с приведением цены к номиналу
    # bondS['Бескупонная доходность к погашению'] = 1000 * (100 - bondS['PRICE']) / 100
    # bondS['Бескупонная доходность к погашению'] = bondS['Бескупонная доходность к погашению'].astype(float).round(2)

    # # Годовая доходность к погашению = суммарный доход к полной цене покупки
    # bondS['Доходность годовых к погашению'] =\
    #     365 * (
    #     100 * (1000 + bondS['Купоный доход к погашению'] + bondS['Бескупонная доходность к погашению'])\
    #     / bondS['Полная цена покупки'] - 100\
    #     ) / bondS['До возможности погасить']
    # bondS['Доходность годовых к погашению'] = bondS['Доходность годовых к погашению'].astype(float).round(4)

    # !!! Стоимость!!!
    if 'Лотов' in bondS.columns: bondS['Стоимость'] = bondS['Лотов'] * bondS['PRICE'] * 10 * bondS['FACEVALUE'] / 1000
        # если поданы на вход облигации из портфеля (уже купленные)
    # display(bondS) # для отладки

# 1.7 Интегральная переменная Специфика
    if 'Сектор рынка' in bondS.columns:
        bondS.loc[(bondS['Сектор рынка'] == 'Гос') | (bondS['SECNAME'].str.contains('ОФЗ|Россия', case=False)), 'Сектор рынка'] = 'Гос'
        bondS.loc[(bondS['Сектор рынка'].str.contains('Гос|Корп|Мун', case=False) != True), 'Сектор рынка'] = 'Корп'

    bondS.loc[(bondS['COUPONPERCENT'].isna()) | (bondS['COUPONPERCENT'] == ''), 'Купон определён'] = 0
    bondS.loc[(bondS['COUPONPERCENT'].notna()) & (bondS['COUPONPERCENT'] != ''), 'Купон определён'] = 1
    # display("bondS['Купон определён']:", bondS['Купон определён'].value_counts()) # для отладки

    bondS['Специфика'] = bondS['FACEUNIT'].str[:2]

    for column in ['Сектор рынка', 'Амортизация FinAm' if 'Амортизация FinAm' in bondS.columns else 'Амортизация T', 'Купон определён']:
        if column in bondS.columns:
            bondS[column] = bondS[column].fillna('--')
            bondS['Специфика'] += ' ' + bondS[column].astype(str).str[:1]

    print('Компоненты специфики: валюта, сектор рынка, амортизация, определён ли купон')
    display(bondS['Специфика'].value_counts().sort_index())

    if returnDfs: return bondS, driver, driver_CB, driver_TB, issuerS_withActualRating, issuerS_withActualRating_change
