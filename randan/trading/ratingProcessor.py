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
while True:
    try:
        from randan.tools import coLabAdaptor, forSelenium # авторские модули для
            # (а) адаптации текущего скрипта к файловой системе CoLab
            # (б) упрощения некоторых оперций в selenium

        from selenium.common.exceptions import NoSuchElementException, TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions
        from selenium.webdriver.support.ui import WebDriverWait
        import pandas, re, undetected_chromedriver
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

options = undetected_chromedriver.ChromeOptions()
options.add_argument('--disable-backgrounding-occluded-windows') # запрет браузеру засыпать в фоне
options.add_argument('--disable-background-timer-throttling') # отключить троттлинг таймеров
# options.headless = True # невидимый режим

# Авторские функции..
    # .. импорта рейтинга с сайта moex.com
def getRatingFromMoEx(bondS_in, columnWithRating, driver, identifier, isin, textTarget):
    bondS = bondS_in.copy()

    driver.get(f'https://www.moex.com/ru/issue.aspx?code={isin}')

    tryer = 0
    while tryer < 2: # на случай появления Cookie и дисклеймера
        # print('tryer:', tryer) # для отладки
        try: # на случай появления Cookie и дисклеймера
            textOfTheBody = driver.find_element("tag name", "body").text
            # print(f'\n{textTarget}\n' in textOfTheBody) # для отладки
            if f'\n{textTarget}\n' in textOfTheBody:
                if f'\nРейтинг не присвоен\n' not in textOfTheBody:

                    # Определяем тип рейтинга
                    rating_header = WebDriverWait(driver, 5).until(expected_conditions.presence_of_element_located((By.XPATH, f"//h2[contains(., '{textTarget}')]")))
                    header_text = rating_header.text
                    print('  ✅ Найден заголовок:', header_text) # , end='\r'
                    
# НАЙТИ ТАБЛИЦУ по структуре из HTML
                    # Ищем ближайшую таблицу после заголовка, используя структуру страницы
                    # Основной контейнер: div с классом 'widget desc-left' и id='creditRating'
                    try:
                        # Стратегия 1.1: Найти таблицу по её характерному классу
                        data_table = WebDriverWait(driver, 5).until(
                            expected_conditions.presence_of_element_located(
                                (By.XPATH, f"//h2[contains(., '{header_text[:15]}')]/following::table[contains(@class, 'emitent-credit-rating-table')]")
                                                                            )
                                                                    )

                        print("  Стратегия 1.1: Таблица найдена по классу 'emitent-credit-rating-table'")
                    except TimeoutException:
                        try:
                            # Стратегия 1.2: Найти ближайший контейнер-виджет с таблицей внутри
                            print("  Стратегия 1.1 не сработала, пробую Стратегию 1.2...")
                            widget_container = rating_header.find_element(By.XPATH, "./following::div[@id='creditRating' or contains(@class, 'widget')][1]")
                            data_table = widget_container.find_element(By.TAG_NAME, "table")
                            print("  Стратегия 1.2: Таблица найдена через контейнер-виджет")

                        except NoSuchElementException:
                            print('  ❌ Ошибка поиска таблицы по структуре из HTML:', sys.exc_info()) # для отладки

# АЛЬТЕРНАТИВНЫЙ ПОИСК контейнера с данными
                            # Поднимаемся от заголовка на несколько уровней вверх, чтобы найти контейнер с данными
                            print('  Стратегия 2: Поиск контейнера с данными:', sys.exc_info()) # для отладки
                            data_table = None
                            max_levels = 5  # Максимальная глубина поиска родительских контейнеров

                            for level in range(1, max_levels + 1):
                                try:
                                    # Создаем XPath для поднятия на level уровней вверх
                                    xpath_parent = "./" + "/".join(["parent::*"] * level)
                                    potential_container = rating_header.find_element(By.XPATH, xpath_parent)

                                    # Проверяем, есть ли в контейнере строки с данными
                                    rows_inside = potential_container.find_elements(By.XPATH,
                                                                                    ".//div[contains(@class, 'row') or contains(@class, 'tr')] | .//tr")
                            
                                    if len(rows_inside) >= 2:  # Если есть хотя бы 2 строки (включая возможные заголовки)
                                        data_table = potential_container
                                        print(f"  Найден контейнер уровня {level}: тег <{data_table.tag_name}>, "
                                              f"класс '{data_table.get_attribute('class')}'")
                                        break

                                except Exception: continue

                            if not data_table:
                                # Если не нашли по уровням, попробуем альтернативные подходы
                                print("  ⚠️ Контейнер не найден по иерархии, использую альтернативные методы...")

                                # Способ 1: Ищем ближайший div с типичными классами
                                try:
                                    data_table = driver.find_element(By.XPATH,
                                        f"//div[contains(@class, 'col-md-') or contains(@class, 'rating')][.//h2[contains(., '{header_text[:20]}')]]")
                                    print(f"   Найден контейнер по классу: {data_table.get_attribute('class')}")
                                except:
                                    # Способ 2: Просто используем родителя h2
                                    data_table = rating_header.find_element(By.XPATH, "./parent::*")
                                    print(f"   Использую непосредственного родителя: тег <{data_table.tag_name}>")

# УНИВЕРСАЛЬНЫЙ ПАРСИНГ ТАБЛИЦЫ (работает с любым количеством столбцов)
                    print("  📊 Извлекаю данные из таблицы...")

                    # Получаем заголовки столбцов
                    headers = []
                    try:
                        header_rows = data_table.find_elements(By.XPATH, ".//thead//th")
                        headers = [th.text.strip() for th in header_rows if th.text.strip()]
                    except:
                        # Если нет thead, ищем th в первой строке
                        header_rows = data_table.find_elements(By.XPATH, ".//tr[1]/th")
                        headers = [th.text.strip() for th in header_rows if th.text.strip()]

                    if not headers:
                        # Если заголовки не найдены, создаем стандартные
                        headers = ['Наименование КРА', 'Значение кредитного рейтинга', 'Дата рейтинга', 'Прогноз']
                        # print("    ⚠️ Заголовки таблицы не найдены, использую стандартные") # для отладки
                    # else:
                        # print(f"  Обнаружены столбцы: {headers}") # для отладки

                    # Получаем все строки данных (исключая возможные пустые строки)
                    data_rows = data_table.find_elements(By.XPATH, ".//tbody//tr | .//tr[td]")
                    rating_data = []

                    for row in data_rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        cell_texts = [cell.text.strip() for cell in cells if cell.text.strip()]

                        # Отбираем строки с содержательными данными (обычно 3+ ячейки)
                        if len(cell_texts) >= 3 and any(keyword in ' '.join(cell_texts) for keyword in ['АКРА', 'НКР', 'НРА', 'Эксперт']):
                            # Создаем запись, сопоставляя данные с заголовками
                            record = {"Тип рейтинга": header_text.replace('Кредитный рейтинг ', '').strip()}

                            # Заполняем известные поля
                            for i, header in enumerate(headers[:len(cell_texts)]):
                                record[header] = cell_texts[i]

                            # Заполняем оставшиеся данные, если столбцов больше, чем заголовков
                            for i in range(len(headers), len(cell_texts)):
                                record[f"Доп. поле {i+1}"] = cell_texts[i]

                            rating_data.append(record)

                    oneBondRating = pandas.DataFrame(rating_data)
                    display('oneBondRating:', oneBondRating) # для отладки
                    oneBondRating = oneBondRating[oneBondRating['Значение кредитного рейтинга'].str.contains('Отозван', case=False) != True] # не интересует, если рейтинг отозван
                    oneBondRating[columnWithRating] = oneBondRating['Значение кредитного рейтинга'].apply(ratingDigitizer, args=('RB',))
                    # display(oneBondRating) # для отладки
                    if identifier != isin:
                        # print('identifier != isin') # для отладки
                        bondS.loc[bondS['Эмитент'] == identifier, columnWithRating] = oneBondRating[columnWithRating].mean()
                    else:
                        # print('identifier == isin') # для отладки
                        bondS.loc[bondS['ISIN'] == identifier, columnWithRating] = oneBondRating[columnWithRating].mean()

                    break # на случай появления Cookie и дисклеймера

                else:
                    print(f'  ❌ {textTarget} не присвоен') # для отладки
                    break # на случай появления Cookie и дисклеймера

            else:
                print(f"  ❌ Упоминание о '{textTarget}' отсутствует на странице") # для отладки
                break # на случай появления Cookie и дисклеймера

        except (KeyError, TimeoutException): # на случай появления Cookie и дисклеймера
            # Предупреждение про Cookie закрыть
            # /html/body/div[9]/div/div/div/div/div/div/button[2]/div/p
            if '\nСогласен\n' in driver.find_element("tag name", "body").text:
                driver.find_element(By.XPATH, "//p[text()='Согласен']").click()
                print('  ✅ Предупреждение про Cookie закрыто') # , end='\r'
            
            # Дисклеймер закрыть
            # /html/body/div[14]/div[3]/div/button[1]
            if '\nСогласенНе согласен' in driver.find_element("tag name", "body").text:
                driver.find_element(By.XPATH, "//button[text()='Согласен']").click()
                print('  ✅ Дисклеймер закрыт') # , end='\r'

            tryer += 1

    return bondS

    # перевода в число рейтинга с эмитентов торгуемых на МосБирже облигаций
def ratingMoExForBondsWithoutRating(bondS_in, byIssuer=True):
    bondS = bondS_in.copy()
    bondS_withoutRating = bondS[bondS['Bond D Rating'].isna()]
    # display('bondS_withoutRating:', bondS_withoutRating) # для отладки

    if len(bondS_withoutRating) == 0: print('Отсутствуют эмитенты и их облигации оставшиеся без рейтинга')
    else:
        driver = undetected_chromedriver.Chrome()

        if byIssuer: identifierS = bondS_withoutRating.drop_duplicates('Эмитент')['Эмитент'].tolist()
        else: identifierS = bondS_withoutRating['ISIN'].tolist()

        # print('identifierS:', identifierS) # для отладки
        identifierS.sort()
    
# Импорт рейтинга с сайта moex.com    
        counter = 0
        for identifier in identifierS:
        # for identifier in identifierS[0:10]: # для отладки
            if byIssuer:
                isin = bondS_withoutRating[bondS_withoutRating['Эмитент'] == identifier]['ISIN'].tolist()[-1] # последний попавшийся ISIN итерируемого эмитента
                print('issuer', identifier, '; ISIN', isin)  
            else:
                isin = identifier
                print('ISIN', isin)    

            textTargetDict = {'Кредитный рейтинг эмитента': 'Bond D Rating', 'Кредитный рейтинг выпуска облигаций': 'Bond Rating D'}
            for textTarget in textTargetDict.keys():
                try: # на случай обрыва связи
                    bondS = getRatingFromMoEx(bondS, textTargetDict[textTarget], driver, identifier, isin, textTarget)
                except: # на случай обрыва связи
                    print(sys.exc_info())
                    return bondS
                    print('--- Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть. Автоматическое исполнение скрипта приостанавливается. Далее вручную перезапустите текущий чанк и последующие')
                    input()
                    driver.quit()
                    sys.exit()

            counter += 1
            print('Элементов множества обработано:', counter, 'из', len(identifierS))
            print("="*60 + "\n")
    
        print('На сайте moex.com могут оказаться рейтинги не для всех облигаций, поэтому следует проверить визуально:')
        display(bondS[bondS['Bond D Rating'].isna()])
        driver.quit()

    return bondS
