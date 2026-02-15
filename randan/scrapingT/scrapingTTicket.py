# country = 'Благовещенск'
# country = 'Египет'
# country = 'Индия'
# country = 'Иордания'
# country = 'Кипр'
country = 'Мальдивы'
# country = 'ОАЭ'
# country = 'Саудовская Аравия'
# country = 'Тайланд'
# country = 'Шри-Ланка'
priceTarget = 30000
transferscount = 1
withbaggage = True

from selenium import webdriver
from selenium.webdriver.common.by import By # для поиска элементов HTML-кода
import datetime, pandas, os, sys, time, warnings
import selenium.common.exceptions
pause = 1

def билеты_есть(city, date, datePriceS, driver, pause, priceTarget, url):
    goF = True # взаимодействует с билетов_нет()
    price = 0 # заранее -- на случай отсутствия билетов       
    print('    Предварительно билеты есть                   ') #, end='\r'

    # Отсортировать по увеличению цены

    firstMoment = datetime.datetime.now()
    while goF:
        try:
            driver.find_element(By.XPATH, "//button[@data-qa-file='SortDropdown']")
            sortDropdown = driver.find_element(By.XPATH, "//button[@data-qa-file='SortDropdown']")
            print('    Кнопка сортировки появилась               ') #, end='\r'
            break
        except:
            # print('\n    ', sys.exc_info()[1]) # для отладки
            print('    Жду кнопку сортировки                   ', end='\r') #, end='\r'
            time.sleep(pause)
            lastMoment = datetime.datetime.now()
            diffetenceSeconds = lastMoment - firstMoment
            print('    Ожидание составляет:', diffetenceSeconds.seconds, 'сек.', end='\r') #, end='\r'
            if 'Билетов нет' in driver.find_element("tag name", "body").text:
                driver, goF = билетов_нет(driver, goF, pause, url)
                # print('goF:', goF) # для отладки

    if goF:
        if 'Дешевле' not in sortDropdown.text:
            print('    Меняю сортировку                   ') #, end='\r'
            sortDropdown.click()
            driver.find_element(By.XPATH, "//div[@data-qa-file='item' and contains(., 'Дешевле')]").click()
            time.sleep(pause)
    
        time.sleep(pause * 25) # чтобы успеть посмотреть на цены на странице
    
        # Выгрузить верхнюю цену
        try:
            flightRoutesListPanel = driver.find_element(By.XPATH, "//div[@data-qa-file='FlightRoutesListPanel']")
            # print('flightRoutesListPanel.text:', flightRoutesListPanel.text) # для отладки
            price = flightRoutesListPanel.find_element(By.XPATH, ".//div[@data-qa-file='PricesOffers']").text
            # print('price:', price) # для отладки
            price = price.split('\n')[0].replace(' ', '')
            price = int(price)
        except:
            print('\n    ', sys.exc_info()[1]) # для отладки          
            print('    Билетов нет                   ', end='\r')
        print('    Цена с багажом от', price, '          ')
    
        datePriceS.loc[city + ' ' + date, 'date'] = date
        datePriceS.loc[city + ' ' + date, 'city'] = city
        datePriceS.loc[city + ' ' + date, 'price'] = price
        datePriceS.loc[city + ' ' + date, 'URL'] = url
        if price < priceTarget:
            print(f'    Найдена цена ниже целевой ({priceTarget})', end='\r')
            time.sleep(pause)
    
    return driver, price

def билетов_нет(driver, goF, pause, url):
    print('    Предварительно билетов нет          ')
    driver.close()
    driver = webdriver.Chrome()
    driver.get(url)

# Поиск лучших предложений
# <div class="DynamicLoaderComponent__label_zQnoE" data-qa-file="DynamicLoaderComponent">Ищем лучшие предложения</div>driver = webdriver.Chrome()
    firstMoment = datetime.datetime.now()
    while True:
        try:
            driver.find_element(By.XPATH, "//div[@data-qa-file='DynamicLoaderComponent' and contains(., 'Ищем лучшие предложения')]")
            print('    Ищем лучшие предложения               ', end='\r') #
            time.sleep(pause)
            lastMoment = datetime.datetime.now()
            diffetenceSeconds = lastMoment - firstMoment
            print('    Ожидание составляет:', diffetenceSeconds.seconds, 'сек.') #, end='\r'
        except:
            # print('\n    ', sys.exc_info()[1]) # для отладки
            print('    Поиск лучших предложений завершён               ')
            break   
    driver.find_element(By.XPATH, "//span[@data-qa-file='ErrorBannerContent' and contains(., 'Билетов нет')]")

    if 'Билетов нет' in driver.find_element("tag name", "body").text:
        print('        Билетов нет          ')
        goF = False

    return driver, goF

def выгрузка_цен(airportS, datePriceS, direction, driver, row):
    goS = True # для взаимодействия с циклом while вне выгрузка_цен()
    try: # обработать ошибку или сигнал сигнал прерывания, поданный на любом этапе сбора данных
        while row <= airportS.index[-1]:
            city = airportS['Город'][row]
            print('\n', city)
            code = airportS['Код'][row]
            
            dateFrom = airportS['Ближайшая дата начала поездки'][row]
            dateFrom = datetime.date(int(str(datetime.date.today()).split('-')[0]), int(dateFrom.split('-')[1]), int(dateFrom.split('-')[2]))
            delta = int(airportS['Максимальный отcтуп'][row])
            
            for dayShift in range(0, delta + 1):
                url = 'https://www.tbank.ru/travel/flights/one-way'
                url += f'/MOW-{code}' if direction == 0 else f'/{code}-MOW'
                date = '-'.join(str(dateFrom + datetime.timedelta(days=dayShift)).split('-')[1:])
                # print('date:', date) # для отладки          
                url += f'/{date}'
                # print("f'/{date}'", f'/{date}') # для отладки          
                url += '/?adults=1&children=0&infants=0&cabin=Y&composite=0'
                if withbaggage: url += '&baggagetype=withbaggage'
                url += f'&transferscount={transferscount}'
                print('                                                  ')
                print('  Захожу по URL-адресу:', url)
                driver.get(url)
                
                firstMoment = datetime.datetime.now()
                while True:
                    try:
                        driver.find_element(By.XPATH, "//span[@data-qa-file='LoadingButton' and contains(., 'Найти')]")
                        print('    Кнопка "Найти" появилась               ') #, end='\r'
                        break   
                    except:
                        print('\n    ', sys.exc_info()[1]) # для отладки
                        print('    Жду кнопку "Найти"')
                        time.sleep(pause)
                        lastMoment = datetime.datetime.now()
                        diffetenceSeconds = lastMoment - firstMoment
                        print('    Ожидание составляет:', diffetenceSeconds.seconds, 'сек.') #, end='\r'
    
                if 'Билетов нет' in driver.find_element("tag name", "body").text:
                    driver, goF = билетов_нет(driver, True, pause, url) # выдача функции здеь не нужна, но пусть будет
                else:
                    driver, price = билеты_есть(city, date, datePriceS, driver, pause, priceTarget, url)
    
            # Конец цика for
    
            row += 1
            if len(datePriceS) > 0:
                print('\n    Промежуточный итог:                          ')
                display(datePriceS[datePriceS['price'] > 0].sort_values('price').head(10))
        # Конец тела внешнего цикла
    
        if len(datePriceS) == 0:
            print('    В интересующий период билетов не найдено')
        return datePriceS, goS, row

    except: # обработать ошибку или сигнал прерывания, поданный на любом этапе сбора данных
        print('\n    ', sys.exc_info()[1]) # для отладки
        goS = False
        return datePriceS, goS, row

driver = webdriver.Chrome()

print('Начинаю поиск билетов туда (если конец сезона, то их можно покупать БЛИЖЕ к дате начала поездки)')
airportS = pandas.read_excel(f'Аэропорты {country}.xlsx', sheet_name='Аэропорты Вылет')
display(airportS)
airportS['Ближайшая дата начала поездки'] = airportS['Ближайшая дата начала поездки'].astype(str)
direction = 0

datePriceS_туда = pandas.DataFrame(columns=['price']) # не внёс внутрь функции выгрузка_цен,
    # чтобы при прерывании исполнения функции можно было продолжить с места прерывания
row = airportS.index[0]

%%time
goS = True
while (row < len(airportS.index)) & goS:
    datePriceS_туда, goS, row = выгрузка_цен(airportS, datePriceS_туда, direction, driver, row)
    datePriceS_туда['direction'] = 'Туда'
print('Завершён поиск билетов туда')

print('Начинаю поиск билетов оттуда (если конец сезона, то их можно покупать ЗАРАНЕЕ)')
airportS = pandas.read_excel(f'Аэропорты {country}.xlsx', sheet_name='Аэропорты Прилёт')
display(airportS)
airportS['Ближайшая дата начала поездки'] = airportS['Ближайшая дата начала поездки'].astype(str)
direction = 1

datePriceS_оттуда = pandas.DataFrame(columns=['price']) # не внёс внутрь функции выгрузка_цен,
    # чтобы при прерывании исполнения функции можно было продолжить с места прерывания
row = airportS.index[0]

%%time
goS = True
while (row < len(airportS.index)) & goS:
    datePriceS_оттуда, goS, row = выгрузка_цен(airportS, datePriceS_оттуда, direction, driver, row)
    datePriceS_оттуда['direction'] = 'Оттуда'
print('Завершён поиск билетов оттуда')

datePriceS = pandas.concat([datePriceS_туда[datePriceS_туда['price'] > 0].sort_values('price'),
                            datePriceS_оттуда[datePriceS_оттуда['price'] > 0].sort_values('price', ascending=False)])
display(datePriceS)

datePriceS['Дата парсинга'] = datetime.date.today().strftime('%Y%m%d')
datePriceS.to_excel(f'parsingTTicket {datetime.date.today().strftime('%Y%m%d')} {country}.xlsx')
try: driver.close()
except: print('driver уже закрыт')

input('Скрипт исполнен')
warnings.filterwarnings("ignore")
sys.exit()



# Тестер
for i in driver.find_elements(By.XPATH, "//div[@data-qa-file='FlightRoutesListPanel']"): # and contains(., 'Дешевле')
    print(i.text)

city = airportS['Город'][row]
date = '04-21'
datePriceS = datePriceS_туда.copy()
url = 'https://www.tbank.ru/travel/flights/one-way/MOW-FJR/04-21/?adults=1&children=0&infants=0&cabin=Y&composite=0&baggagetype=withbaggage&transferscount=1'
driver.get(url)
if 'Билетов нет' in driver.find_element("tag name", "body").text: driver, goF = билетов_нет(driver, True, pause, url)
else:
    driver, price = билеты_есть(city, date, datePriceS, driver, pause, priceTarget, url)
