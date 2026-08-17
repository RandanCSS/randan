# Модуль для для выяснения, какие инструменты (акции, облигации и т.д.) есть в портфеле, на основе брокерских отчётов

# 0. Активировать требуемые для работы скрипта модули и пакеты 
# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from datetime import date
        from IPython.display import display
        from randan.tools import coLabAdaptor # авторский модуль для адаптации текущего скрипта к файловой системе CoLab
        from randan.trading import bondsFeaturesProcessor # авторский модуль для гармонизации и обработки характеристик облигаций
        import os, pandas, re, warnings
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

coLabFolder = coLabAdaptor.coLabAdaptor()

# 1.0 Авторские функции
# # 1.0.0 поиска в брокерском отчёте строчек, ограничивающих интересующий раздел
# def boundColibrator(assetS, bound, col, name):
#     if len(bound) == 1:
#         bound = bound[0]
#     elif len(bound) > 1:
#         # print('Найдено строк:', len(bound)) # для оталдки
#         # display(assetS.loc[bound, col]) # для оталдки
#         bound = assetS.loc[bound, col][
#             assetS.loc[bound, col].dropna().astype(str).str.contains(name)
#                 ].index[0]
#     return bound

# .. поиска в таблице строчек, ограничивающих интересующий раздел
def boundColibrator(bound, column, df, softCondition, text):
    # if type(bound) == list:
    if len(bound) == 1: bound = bound[0]
    elif len(bound) > 1:
        # print('Найдено строк:', len(bound)) # для оталдки
        # display(df.loc[bound, column]) # для оталдки
        if softCondition: bound = df.loc[bound, column][df.loc[bound, column].dropna().astype(str).str.contains(text)].index[0]
        else: bound = df.loc[bound, column][df.loc[bound, column] == text].index[0]

    return bound


# 1.0.1 организации обработки отчётов каждого брокера за интересующий период
def brokerReportsProcessor(broker, fileNameS, path, period, slash):
    assetS = pandas.DataFrame()
    # print('fileNameS:', fileNameS) # для отладки
    for fileName in fileNameS:
        # print('fileName:', fileName) # для отладки
        assetS_additional = pandas.read_excel(path + broker + slash + 'Отчёты' + slash + fileName, header=None)
        fileNameSpecification = fileName.replace('.xlsx', '').replace(str(period), '')
        # print('fileNameSpecification:', fileNameSpecification) # для отладки
        print(f"Распарсиваю отчёт брокера{'' if fileNameSpecification == '' else ' (' + fileNameSpecification + ')'}", broker, 'за', str(period)) #, end='\r'
        if broker == 'ВТБ': assetS_additional = ВТБ(assetS_additional)
        if broker == 'Тинькофф': assetS_additional = Тинькофф(assetS_additional)
        if broker == 'УралСиб': assetS_additional = УралСиб(assetS_additional)
        # display('assetS_additional:', assetS_additional) # для оталдки        

        assetS_additional.loc[:, 'Брокер'] = broker + fileNameSpecification
        assetS = pandas.concat([assetS, assetS_additional], ignore_index=True)
        # display(assetS['Брокер']) # для отладки
        # display(assetS) # для отладки
    return assetS

# 1.0.2 .. поиска в таблице столбцов, чьи ячейки содержат ключевой текст
def columnFinder(df, text):
    # Найти первый слева столбец, содержащий ключевой текст
    # print('df.columns:', df.columns) # для оталдки
    for column in df.columns:
        if sum(df.loc[:, column].dropna().astype(str).str.contains(text)) > 0:
            # print('column:', column) # для оталдки
            break
    column = column if sum(df.loc[:, column].dropna().astype(str).str.contains(text)) > 0 else None
    return column

# 1.0.3 .. поиска в таблице столбцов, содержащих в своих заголовках ключевой текст
def columnNameFinder(df, text):
    # Найти первый слева столбец, содержащий ключевой текст
    for column in df.columns:
        if text in column:
            break
    column = column if text in column else None
    return column

# 1.0.4 поиска в директориях брокеров отчётов за интересующий период
def reportSearch(broker, path, period, slash):
    fileNameS = []
    goC = True
    while goC:
        for fileName in os.listdir(path + broker + slash + 'Отчёты'):
            if str(period) in fileName:
                # print(f"Нашёл файл '{fileName}'") # для оталдки
                fileNameS.append(fileName)
        if len(fileNameS) > 0:
            goC = False
            return fileNameS, period
            break
        if goC:
            print(f"НЕ нашёл файл за период '{period}'; перехожу к предыдущему периоду")
            period = period - 1 if str(period)[-2:] != '01' else period - 89 # на случай января
            if period == 202001:
                print(f"Безуспешно добрался до периода '{period}'; завершаю поиск и исполнение текущей функции")
                print('Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть')
                input()
                sys.exit()
                break

# # 1.0.5 поиска в брокерском отчёте фрагментов, соответствующих интересующим разделам
# def sectionFinder(assetS, text, textNext):
#     column = columnFinder(assetS, name)
#     upper_bound = assetS[assetS[col].notna() & assetS[col].str.contains(name)].index
#     upper_bound = boundColibrator(upper_bound, column, assetS, True, text)

#     lower_bound = assetS[assetS[col].notna()].index[-1]
#     if textNext != '':
#         lower_bound = assetS[assetS[col].notna() & assetS[col].str.contains(textNext)].index
#         lower_bound = boundColibrator(lower_bound, column, assetS, True, text)
#     return assetS, lower_bound, upper_bound

# .. поиска в таблице фрагментов, содержащих ключевой текст
def sectionFinder(df, softCondition, text, textClosing): # textClosing -- текст, следующий за искомым подразделом; должен располагаться в том же столбце
    column = columnFinder(df, text)
    # print('column:', column) # для оталдки

    boundSmaller = df[df[column].notna() & df[column].str.contains(text)].index if softCondition else df[df[column] == text].index
    # print('boundSmaller:', boundSmaller) # для оталдки

    boundSmaller = boundColibrator(boundSmaller, column, df, softCondition, text)
    # print('boundSmaller:', boundSmaller) # для оталдки

    boundLarger = df[df[column].notna()].index[-1]
    if textClosing != '':
        boundLarger = df[df[column].notna() & df[column].str.contains(textClosing)].index if softCondition else df[df[column] == textClosing].index
        # print('boundLarger:', boundLarger) # для оталдки

        boundLarger = boundColibrator(boundLarger, column, df, softCondition, text)
    # print('boundLarger:', boundLarger) # для оталдки
    return boundLarger, boundSmaller

# 1.1 Авторские функции-адаптеры по брокерам
def ВТБ(assetS):
    # display('assetS:', assetS) # для отладки
    # Найти срез датафрейма, соответствующий искомому разделу (раздел предполагает свои наименования столбцов)
    text = 'т об остатках ценных бумаг' # вместо "Отчет", чтобы избежать чередования букв е и ё
    textNext = ''
    # print('\nИскомый раздел:', text) # для оталдки

    # if textNext != '': print('Раздел, следующий за искомым:', textNext) # для оталдки
    boundLarger, boundSmaller = sectionFinder(assetS, True, text, textNext)
    # print('boundSmaller:', boundSmaller, ', boundLarger:', boundLarger) # для оталдки
    
    colS = list(assetS.loc[boundSmaller + 1, :].astype(str)) # поскольку раздел предполагает свои наименования столбцов
    assetS = assetS.loc[boundSmaller + 2: boundLarger - 1, :]
    # display('assetS:', assetS) # для отладки
    
    # Найти срез датафрейма, соответствующий искомому подразделу (подраздел НЕ предполагает свои наименования столбцов)
    text = 'ОБЛИГАЦИЯ'
    textNext = 'ИТОГО:' # Слово(сочетание), следующие за искомым подразделом, должно располагаться в том же столбце
    # print('\nИскомый подраздел:', text) # для оталдки

    # if textNext != '': print('Слово(сочетание), следующие за искомым подразделом:', textNext) # для оталдки
    boundLarger, boundSmaller = sectionFinder(assetS, True, text, textNext)
    # print('boundSmaller:', boundSmaller, ', boundLarger:', boundLarger) # для оталдки

    assetS = assetS.loc[boundSmaller + 1: boundLarger - 1, :]
    assetS.columns = colS

    # Найти в этом срезе столбцы с ISIN и с позицией
    name = 'ISIN'
    ISIN_clmn = columnNameFinder(assetS, name)
    # print('\nИмя столбца с ISIN:', ISIN_clmn) # для оталдки

    name = 'Плановый исходящий остаток'
    pose_clmn = columnNameFinder(assetS, name)
    # print('\nИмя столбца с позицией:', pose_clmn) # для оталдки
    assetS = assetS[[ISIN_clmn, pose_clmn]]
    assetS.columns = ['ISIN', 'Лотов']

    assetS['ISIN'] = assetS['ISIN'].str.split(',').str[-1]
    assetS = assetS[assetS['Лотов'].notna()] # избавиться от заголовка подтаблицы "ПАИ"
    assetS['Лотов'] = assetS['Лотов'].astype(str).str.split('.').str[0].str.replace(',', '').astype(int)
    # display('assetS:', assetS) # для отладки
    return assetS

def Тинькофф(assetS):
    # display(assetS) # для отладки
    # Раздел "3.1 Движение по ценным бумагам инвестора"
    # Найти столбец с указанием раздела
    for i in range(len(assetS.columns)):
        if sum(assetS.iloc[:, i].dropna().astype(str).str.contains('Движение ')) > 0:
            break
    boundS = assetS[assetS[i].notna() & assetS[i].str.contains('Движение ')].index
    # print(boundS) # для отладки
    boundSmaller = boundS[0]
    boundLarger = boundS[-1]
    assetS = assetS.loc[boundSmaller + 1: boundLarger - 1, :]
    # display(assetS) # для отладки

    # Найти столбец с ISIN
    for i in range(len(assetS.columns)):
        if len(assetS.iloc[:, i].dropna()) > 0:
            if sum(assetS.iloc[:, i].dropna().str.contains('RU000')) > 0:
                break
    ISIN_clmn = i # имя столбца с ISIN
    # print(ISIN_clmn)
    bond_indeseS = assetS[assetS[ISIN_clmn].notna() & assetS[ISIN_clmn].str.contains('RU000')].index # строки с бондами
    # print(bond_indeseS) # для отладки
    assetS = assetS.loc[bond_indeseS, :]
    # display(assetS) # для отладки
    clmnS = assetS.loc[bond_indeseS[0], :] # непустые ячейки первой строки с бондами
    # display(clmnS.dropna())
    pose_clmn = clmnS.dropna().index[-1] # имя столбца с Позицией
    # print(pose_clmn) # для отладки
    assetS = assetS[[ISIN_clmn, pose_clmn]]
    assetS.columns = ['ISIN', 'Лотов']
    assetS = assetS[assetS['Лотов'].notna()] # избавиться от заголовка подтаблицы "ПАИ"
    assetS['Лотов'] = assetS['Лотов'].astype(str).str.split('.').str[0].str.replace(',', '').astype(int)
    # display(assetS) # для отладки
    return assetS

def УралСиб(assetS):
    # display(assetS) # для отладки
    assetS = assetS[[3, 10]]
    ISIN_clmn = 3
    bond_indeseS = assetS[assetS[ISIN_clmn].notna() & assetS[ISIN_clmn].str.contains('RU000')].index # строки с акциями и бондами
    # print('bond_indeseS:', bond_indeseS) # для отладки
    assetS = assetS.loc[bond_indeseS, :]
    assetS.columns = ['ISIN', 'Лотов']
    # display(assetS) # для отладки
    return assetS

# 2. Авторская функция исполнения скрипта
def getAssets(
              brokerS=['ВТБ', 'Тинькофф', 'УралСиб'],
              path=coLabFolder,
              returnDfs=False
              ):
    """
    Функция для выяснения, какие облигации есть в портфеле, на основе брокерских отчётов

    Parameters
    ----------
     brokersS : list -- список Ваших брокеров, например, такой: ['ВТБ', 'Тинькофф', 'УралСиб'] ; названия этих трёх брокеров следует писать именно так; для других брокеров функций-адаптеров пока нет.
         path : str -- путь к директории, включая её имя, в которой будут искаться файлы и куда будут сохраняться; по умолчанию, не в CoLab поиск и сохранение происходят в директории, в которой вызывается текущая функция, а в CoLab в директории Colab Notebooks

    returnDfs : bool -- в случае True функция возвращает итоговые датафрейм bondS
    """
        
# 2.0 Настройки
    print('Работаю с месячным периодом; желательно, чтобы все обрабатываемые брокерские отчёты относились к одному периоду')
    period = date.today().strftime("%Y%m%d")[:4] + date.today().strftime("%Y%m%d")[4:6]
    period = int(period) - 89 if period[-2:] == '01' else int(period) - 1 # на случай января
    # print('Целевой период:', period) # для отладки
    
    slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС
    if path == None: path = ''
    else: path += slash

    warnings.filterwarnings("ignore")

# 2.1 Выяснение, какие облигации есть в портфеле, на основе брокерских отчётов
    assetS = pandas.DataFrame()
    for broker in brokerS:
        fileNameS, period = reportSearch(broker, path, period, slash)
        # print('broker:', broker) # для отладки        
        assetS_additional = brokerReportsProcessor(broker, fileNameS, path, period, slash)
        # display(assetS_additional) # для отладки
        assetS = pandas.concat([assetS, assetS_additional], ignore_index=True)
    # display(assetS) # для отладки

    # Удалить лишние пробелы во всём накопленном датафрейме
    assetS['ISIN'] = assetS['ISIN'].apply(lambda text: re.sub(r' +', r'', text))
    # display(assetS) # для отладки

# 2.3 Облигации по ключу: ISIN
    assetS_byISIN = pandas.DataFrame(columns=brokerS)
    for isin in assetS['ISIN'].unique():
        # Для каждого брокера создать столбец  
        brokerS_byISIN = assetS[assetS['ISIN'] == isin]
        # display(brokerS_byISIN) # для отладки
        for broker in brokerS_byISIN['Брокер']: # не все brokerS , т.к. конкретный ISIN может встретиться не у всех брокеров
            # display(brokerS_byISIN[brokerS_byISIN['Брокер'] == broker]) # для отладки            
            assetS_byISIN.loc[isin, broker] = brokerS_byISIN.loc[brokerS_byISIN[brokerS_byISIN['Брокер'] == broker].index[0], 'Лотов']
    # display(assetS_byISIN) # для отладки
    assetS_byISIN['Лотов'] = assetS_byISIN.T.sum()
    assetS_byISIN['ISIN'] = assetS_byISIN.index
    assetS_byISIN = assetS_byISIN.reset_index(drop=True)
    # display('assetS_byISIN:', assetS_byISIN) # для отладки

    if returnDfs: return assetS, assetS_byISIN
