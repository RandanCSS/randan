# coding: utf-8

'''
A proprietary module to import and process bond issuer ratings from the Moscow Exchange
Авторский модуль для импорта рейтинга эмитентов торгуемых на МосБирже облигаций и его обработки
'''
# import sys
# sys.path.append(r"C:\Users\Alexey\Dropbox\Мои\RAnDan\myModules")

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
for attempt in range(1, 4):
    try:
        from IPython.display import display
        from randan.tools import coLabAdaptor # авторский модуль для адаптации текущего скрипта к файловой системе CoLab
        from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions
        from selenium.webdriver.support.ui import WebDriverWait
        from tqdm import tqdm
        import pandas, re, traceback, undetected_chromedriver
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

# Авторские функции..
# .. поиска эмитентов, у облигаций которых в столбце Bond D Rating (а) ни у одной нет рейтинга,
    # (б) у некоторых есть рейтинг и у некоторых его нет и (в) у всех есть рейтинг
def bondS_ofIssuer_ratingChecker(bondS_in):
    bondS = bondS_in.copy()

    issuerS_fromBonds = bondS[bondS['Эмитент'].notna()]['Эмитент'].drop_duplicates().tolist()
    # print('issuerS_fromBonds:', issuerS_fromBonds) # для отладки

    issuerS_fromBonds_fullRating = []
    issuerS_fromBonds_noRating = []
    issuerS_fromBonds_partialRating = []

    for issuer_fromBonds in tqdm(issuerS_fromBonds):
        bondS_ofIssuer = bondS[bondS['Эмитент'] == issuer_fromBonds]
        # display('bondS_ofIssuer:', bondS_ofIssuer) # для отладки

        if sum(bondS_ofIssuer['Bond D Rating'].notna()) == 0: issuerS_fromBonds_noRating.append(issuer_fromBonds)
        elif sum(bondS_ofIssuer['Bond D Rating'].notna()) == len(bondS_ofIssuer): issuerS_fromBonds_fullRating.append(issuer_fromBonds)
        else: issuerS_fromBonds_partialRating.append(issuer_fromBonds)
            # len(bondS_ofIssuer['Bond D Rating'].notna()) > 0 , но не == len(bondS_ofIssuer)

    print('Проверка:',
          len(issuerS_fromBonds_fullRating) + len(issuerS_fromBonds_noRating) + len(issuerS_fromBonds_partialRating) == len(issuerS_fromBonds))

    return issuerS_fromBonds_fullRating, issuerS_fromBonds_noRating, issuerS_fromBonds_partialRating

# .. импорта рейтинга с сайта moex.com
def getRatingFromMoEx(bondS_in: pandas.DataFrame,
                      columnWithRating: str, 
                      driver: undetected_chromedriver,
                      identifier: str,
                      isin: str, 
                      pause: int,
                      textTarget: str) -> pandas.DataFrame:
    bondS = bondS_in.copy()
    bondS_rowS = []

    # 'https://www.moex.com/ru/issue.aspx?board=TQOD&code=RU000A10DYP0' # для отладки
    try:
        driver.get(f'https://www.moex.com/ru/issue.aspx?code={isin}')
        print('Страница загрузилась успешно') # для отладки
    except (TimeoutError, TimeoutException, WebDriverException):
        print('Загрузка страницы длится слишком долго; перехожу к timeoutExceptionProcesser') # для отладки
        driver = timeoutExceptionProcesser(driver, isin, pause)
        if driver == None: return bondS, bondS_rowS, driver # выход из функции

    # Ожидание, чтобы страница прогрузилась
    # Архитектура
    # /html/body/div[3]/div[6]/div/div/div[1]/div/div/div/div/div[3]/div/div[3]/div[2]/div[1]/h2
    try:
        WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
            (By.XPATH, f"//div[@class='tab-content']//h2[contains(., 'Параметры инструмента')]")
            ))
        print('  ✅ Облигация найдена по ISIN') # , end='\r'
    except Exception as excptn:
        print('Exception 1:', excptn)
        print(traceback.format_exc()) # показ точной строчки кода с ошибкой

        pageSource = driver.page_source
        print(pageSource) # для отладки
        
        print('  ✅ Облигация НЕ найдена по ISIN, ищу по SECID') # , end='\r'
        secidIndex = bondS.loc[bondS['ISIN'] == isin, 'SECID'].index
        secid = bondS.loc[secidIndex[0], 'SECID']

        try: driver.get(f'https://www.moex.com/ru/issue.aspx?code={secid}')
        except Exception as excptn:
            print('Exception 2:', excptn)
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой

            bondS.loc[bondS['Эмитент'] == identifier, 'Rating Notation'] = 'Проблема загрузки страницы'

            return bondS, bondS_rowS, driver # выход из функции

        WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
            (By.XPATH, f"//div[@class='tab-content']//h2[contains(., 'Параметры инструмента')]")
            ))
        print('  ✅ Облигация найдена по SECID') # , end='\r'

    body_text = driver.find_element("tag name", "body").text
    # Не_согласен_pattern = re.compile(rf"\b{re.escape('Не согласен')}\b", re.IGNORECASE) # чтобы не спутать с похожими формулировками
    Согласен_pattern = re.compile(rf"\b{re.escape('Согласен')}\b", re.IGNORECASE) # чтобы не спутать с похожими формулировками
    # СогласенНе_согласен_pattern = re.compile(rf"\b{re.escape('СогласенНе согласен')}\b", re.IGNORECASE) # чтобы не спутать с похожими формулировками

    if Согласен_pattern.search(body_text.strip()): # наличие этго слова -- сигнал к проверке предупреждения про Cookie и дисклеймера
        try:
            cookieAnchor = WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
                (By.XPATH, "//div[@class='_usagePolicy']")
                ))
            cookieAnchor.find_element(By.XPATH, ".//p[text()='Согласен']").click()
            print('  ✅ Предупреждение про Cookie закрыто') # , end='\r'
        except Exception as excptn:
            print('Exception 3:', excptn)
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой
            print('  ✅ Предупреждение про Cookie не найдено') # , end='\r'

        try:
            disclaimerAnchor = WebDriverWait(driver, pause).until(expected_conditions.presence_of_element_located(
                (By.XPATH, "//div[@class='ui-dialog-buttonset']")
                ))        
            disclaimerAnchor.find_element(By.XPATH, ".//button[text()='Согласен']").click()
            print('  ✅ Дисклеймер закрыт') # , end='\r'
        except Exception as excptn:
            print('Exception 4:', excptn)
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой
            print('  ✅ Дисклеймер не найден') # , end='\r'

    # textTarget = 'Кредитный рейтинг эмитента' # для отладки
    # textTarget = 'Кредитный рейтинг выпуска облигаций' # для отладки

    textTarget_pattern = re.compile(rf"\b{re.escape(textTarget)}\b", re.IGNORECASE) # чтобы не спутать с похожими формулировками
    if textTarget_pattern.search(body_text.strip()):
        print('textTarget:', textTarget) # для отладки

        # Поиск таблицы с соответствующим кредитным рейтингом
        tableWithRating = driver.find_element(By.XPATH, f"//h2[contains(., '{textTarget}')]/following-sibling::div")
        # print('tableWithRating.text:', tableWithRating.text) # для отладки
        
        # Извлечь заголовки столбцов
        try:
            headerElementS = tableWithRating.find_elements(By.XPATH, ".//thead//th")
        except Exception as excptn: # если нет thead, ищем th в первой строке
            print('Exception 5:', excptn)
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой
            headerElementS = tableWithRating.find_elements(By.XPATH, ".//tr[1]/th")

        headerS = []
        headerS = [th.text.strip() for th in headerElementS if th.text.strip()]
        print('headerS:', headerS) # для отладки

        if identifier != isin:
            # print('identifier != isin') # для отладки
            bondS_rowS = list(bondS[bondS['Эмитент'] == identifier].index)
        else:
            # print('identifier == isin') # для отладки
            bondS_rowS = list(bondS[bondS['ISIN'] == identifier].index)

        if len(headerS) == 0: bondS.loc[bondS_rowS, 'Rating Notation'] = 'Рейтинг не присвоен или неизвестен, или отозван'
        else:
            dictS_withRating = []
            for dataRow in tableWithRating.find_elements(By.XPATH, ".//tbody//tr | .//tr[td]"):
                dataCellS = dataRow.find_elements(By.TAG_NAME, "td")
                dataCellTextS = [cell.text.strip() for cell in dataCellS if cell.text.strip()]

                # отбор строк с содержательными данными (обычно 3+ ячейки)
                if len(dataCellTextS) >= 3 and any(keyword in ' '.join(dataCellTextS) for keyword in ['АКРА', 'НКР', 'НРА', 'Эксперт']):
                    # воспроизвести таблицу, сопоставляя данные с заголовками
                    record = {"Тип рейтинга": textTarget.replace('Кредитный рейтинг ', '').strip()}

                    # Заполнить известные поля
                    for i, header in enumerate(headerS[:len(dataCellTextS)]):
                        record[header] = dataCellTextS[i]

                    # Заполняем оставшиеся данные, если столбцов больше, чем заголовков
                    for i in range(len(headerS), len(dataCellTextS)):
                        record[f"Доп. поле {i + 1}"] = dataCellTextS[i]

                    dictS_withRating.append(record)
            # print('dictS_withRating:', dictS_withRating) # для отладки

            oneBondRating = pandas.DataFrame(dictS_withRating)
            display('oneBondRating:', oneBondRating) # для отладки
            oneBondRating = oneBondRating[oneBondRating['Значение кредитного рейтинга'].str.contains('Отозван', case=False) != True] # не интересует, если рейтинг отозван
            oneBondRating[columnWithRating] = oneBondRating['Значение кредитного рейтинга'].apply(ratingDigitizer, args=('RB',))
            # display(oneBondRating) # для отладки

            bondS.loc[bondS_rowS, columnWithRating] = oneBondRating[columnWithRating].mean()
                # присвоить рейтинг эмитента всем его облигациям в bondS

            # if identifier != isin:
            #     # print('identifier != isin') # для отладки
            #     bondS.loc[bondS['Эмитент'] == identifier, columnWithRating] = oneBondRating[columnWithRating].mean()
            #         # присвоить рейтинг эмитента всем его облигациям в bondS
            # else:
            #     # print('idenatifier == isin') # для отладки
            #     bondS.loc[bondS['ISIN'] == identifier, columnWithRating] = oneBondRating[columnWithRating].mean()

    return bondS, bondS_rowS, driver

    # перевода в число рейтинга с эмитентов торгуемых на МосБирже облигаций
def ratingDigitizer(letters, raitingSource):
    if raitingSource == 'RB':
        letters = letters.replace(' ГМ', '').replace(' Неквалы', '').replace(' Остальные', '')
        if letters == 'ГМ':
            return 18
        letters = re.findall(r'[ABАВ\-\+]+', letters)[0]
        # print('letters :', letters) # для отладки
        subtracted = 1 if ('-' in letters) | ('+' in letters) else 0
        # print('subtracted :', subtracted) # для отладки

    if raitingSource == 'RA': # в записях не только рейтинги, но и вспомогательные символы; как образец смотреть директорию "boughtIssuerS Архив"
        subtracted = 3 if ('-' in letters) | ('+' in letters) else 2
        # print('subtracted:', subtracted) # для отладки

    x = 1 if ('A' in letters) | ('А' in letters) else 0
    # print('x :', x) # для отладки

    if '-' in letters:
        y = -2
    elif '+' in letters:
        y = 0
    else:
        y = -1
    # print('y :', y)
    # print('len(letters) :', len(letters)) # для отладки
    return (3 * (len(letters) - subtracted) + y) + 9 * x

def ratingMoExForBondsWithoutRating(bondS_in, pause, subordinated=False):
    bondS = bondS_in.copy()

    userChoice = ' '
    if subordinated != True:
        print(
'''
Если обнаружу облигации без рейтинга в столбце Bond D Rating и при этом обнаружу у этой или другой облигации того же эмитента рейтинг в столбце Issuer D Rating, то
--- хотите, чтобы я пошерил располагаемый рейтинг? Тогда нажмите Enter (так быстрее)
--- хотите, чтобы я выяснял неостающий на сайте moex.ru ? Тогда нажмите ПРОБЕЛ затем и Enter (так медленнее, но потенциально точнее, хотя обычно рейтинги эмитента и его несубординированной облигации одинаковы)
'''
              )
        userChoice = input()

    textTargetDict = {'Кредитный рейтинг эмитента': 'Issuer D Rating', 'Кредитный рейтинг выпуска облигаций': 'Bond D Rating'}
    if len(userChoice) == 0: del textTargetDict['Кредитный рейтинг выпуска облигаций'] # в этом случе следующий далее цикл будет иметь одну итерацию
    for textTarget in textTargetDict.keys():
        column_tagert = textTargetDict[textTarget]
        bondS_withoutRating = bondS[bondS[column_tagert].isna()]
        # bondS_withoutRating = bondS_withoutRating.head() # для отладки
        display(textTarget, 'отсутствует у следующих облигаций:', bondS_withoutRating) # для отладки

        if len(bondS_withoutRating) == 0: print('В bondS у всех эмитентов и их облигаций отражён их рейтинг')
        else:
            options = undetected_chromedriver.ChromeOptions()
            options.add_argument('--disable-backgrounding-occluded-windows') # запрет браузеру засыпать в фоне
            options.add_argument('--disable-background-timer-throttling') # отключить троттлинг таймеров
            # options.headless = True # невидимый режим
            driver = undetected_chromedriver.Chrome(options=options, use_subprocess=True, version_main=150) # 
            driver.set_page_load_timeout(100 * pause)

            if textTarget == 'Кредитный рейтинг эмитента': identifierS = bondS_withoutRating.drop_duplicates('Эмитент')['Эмитент'].tolist()
            else: identifierS = bondS_withoutRating['ISIN'].tolist() # т.е. textTarget == 'Кредитный рейтинг выпуска облигаций'

            identifierS.sort()
            print('identifierS:', identifierS) # для отладки

            # Импорт рейтинга с сайта moex.com    
            bondS_rowS_processed = [] # список обработанных строк датафрейма bondS ;
                # подаётся при повторных запусках функции ratingMoExForBondsWithoutRating и её оболочек, чтобы избежать повторных в рамках сессии скрипта обращений к этим строкам

            counter = 0
            for identifier in identifierS:
            # for identifier in identifierS[0:5]: # для отладки
                if textTarget == 'Кредитный рейтинг эмитента':
                    isin = bondS_withoutRating[bondS_withoutRating['Эмитент'] == identifier]['ISIN'].tolist()[-1] # последний попавшийся ISIN итерируемого эмитента
                    print('issuer', identifier, '; ISIN', isin)  
                else:
                    isin = identifier
                    print('ISIN', isin)    

                # На всякий случай, например, обрыва связи
                try: bondS, bondS_rowS, driver = getRatingFromMoEx(bondS, textTargetDict[textTarget], driver, identifier, isin, pause, textTarget)
                except Exception as excptn:
                    print('Exception 1:', excptn)
                    print(traceback.format_exc()) # показ точной строчки кода с ошибкой
                    return bondS, bondS_rowS_processed

                bondS_rowS_processed.extend(bondS_rowS)
                counter += 1
                print('Элементов множества обработано:', counter, 'из', len(identifierS))
                print("="*60 + "\n")

            print('На сайте moex.com могут оказаться рейтинги не для всех облигаций, поэтому следует проверить визуально:')
            display(bondS[bondS[column_tagert].isna()])
            driver.quit()

    if len(userChoice) == 0:
        print('\nПриступаю к функции ratingThroughIssuer') # для отладки
        bondS = ratingThroughIssuer(bondS, 'Issuer D Rating', 'Bond D Rating')
            # заполнить стобец Bond D Rating, если у этой же или другой облигации того же эмитента отражён рейтинг в столбце Issuer D Rating

    return bondS, bondS_rowS_processed

def timeoutExceptionProcesser(driver, isin, pause):
    for attempt in range(3):
        print('attempt:', attempt) # для отладки    

        if driver is not None:
            try:
                driver.quit()
            except Exception as excptn:
                print('Exception 1:', excptn)
                print(traceback.format_exc()) # показ точной строчки кода с ошибкой

                print(f"Не удалось закрыть драйвер")
            finally:
                driver = None  # Обнуляем ссылку

        try:
            options = undetected_chromedriver.ChromeOptions()
            options.add_argument("--pageLoadStrategy=none") # стратегия загрузки: 'none' -- не ждать загрузки вообще;
                # это позволяет начать парсить сразу после получения HTML

            driver = undetected_chromedriver.Chrome(options=options, use_subprocess=True, version_main=150) # 
            driver.set_page_load_timeout((1 + attempt) * 100 * pause)

            try: driver.get(f'https://www.moex.com/ru/issue.aspx?code={isin}')
            except TimeoutException:
                print('Загрука страницы прервана для проверки наличия на ней искомого текста') # для отладки

            pageSource = driver.page_source
            # print(pageSource) # для отладки

            if 'инструмент' in pageSource or 'согласен' in pageSource:
                print('Условие наличия на странице искомого текста выполнено') # для отладки
                return driver

        except Exception as excptn:
            print('Exception 1:', excptn)
            print(traceback.format_exc()) # показ точной строчки кода с ошибкой

            if driver:
                try: driver.quit()
                except: pass
                driver = None
            continue
    
    return None

def ratingThroughIssuer(bondS_in, columnSource, columnTarget):
# Обработка облигаций без рейтинга в столбце Issuer D Rating или Bond D Rating
# Функция заполняет эти стобцы, если у другой облигации того же эмитента отражён рейтинг в столбце Issuer D Rating или Bond D Rating
    bondS = bondS_in.copy()
    issuerS_withRating = bondS[bondS[['Эмитент', columnSource]].notna()].drop_duplicates('Эмитент', ignore_index=True)
    # display('issuerS_withRating:', issuerS_withRating) # для отладки, это датафрейм

    issuerS_withoutRating = bondS[(bondS['Эмитент'].notna()) & (bondS[columnTarget].isna())]['Эмитент'].drop_duplicates().tolist()
    issuerS_withoutRating.sort()
    # print('issuerS_withoutRating:', issuerS_withoutRating) # для отладки, это список

    # Сравнить issuerS_withoutRating с issuerS_withRating['Эмитент'].tolist() и в случае совпадения заполнить bondS[columnTarget]
    for issuer_withoutRating in issuerS_withoutRating:
        if issuer_withoutRating in issuerS_withRating['Эмитент'].tolist():
            # если у эмитента облигации без рейтинга в columnTarget есть облигация с рейтингом в columnSource, расшерить этот рейтинг
            # (допустимо только для несубординированных облигаций)

            # print('issuer_withoutRating:', issuer_withoutRating) # для отладки

            issuerS_withRating_index = issuerS_withRating[issuerS_withRating['Эмитент'] == issuer_withoutRating].index
            # print('issuerS_withRating_index:', issuerS_withRating_index) # для отладки

            bondS.loc[bondS['Эмитент'] == issuer_withoutRating, columnTarget] =\
                issuerS_withRating.loc[issuerS_withRating_index, columnSource][issuerS_withRating_index[0]]
    return bondS
