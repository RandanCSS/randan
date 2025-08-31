def blockSearch(attemptsMax, text, xPathS):
    block = None 
    trCounter = 1
    while True:
        # print('xPath:', xPathS[0] + str(trCounter) + xPathS[1]) # для отладки
        try:
            block = driver.find_element(By.XPATH, xPathS[0] + str(trCounter) + xPathS[1]).text
            if text in block: break
            trCounter += 1
        except:
            # print('trCounter:', trCounter) # для отладки
            trCounter += 1
            if trCounter > attemptsMax: break # против бесконечного цикла при пустом блоке страницы
    return block if text in block else None
