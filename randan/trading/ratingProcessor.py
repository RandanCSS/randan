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
