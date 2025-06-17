#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for topic modelling
(RU) Авторский модуль для тематического моделирования
'''

# 0 Активировать требуемые для работы скрипта модули и пакеты + пререквизиты

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from io import BytesIO
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image
        from openpyxl.utils.dataframe import dataframe_to_rows
        from randan import descriptive_statistics, dimension_reduction
        from tqdm import tqdm
        import matplotlib.pyplot as plt, os, pandas, time, warnings
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[1]
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
tqdm.pandas() # для визуализации прогресса функций, применяемых к датафреймам

# 1 Авторские функции для
# 1.0 оформления токенов на полюсах топиков
def summaryPole(loadingsThreshold, minusPlus, tokensLimit, topicDocS, topicLoadingS, topicName):
    summaryPole = pandas.DataFrame()
    if len(topicLoadingS[topicLoadingS[topicName] * minusPlus > loadingsThreshold].sort_values(topicName, ascending=False).head(tokensLimit)) > 0:
        summaryPole = topicDocS[topicDocS[topicName] * minusPlus > 0].round(3)
        summaryPole.loc[:, 'Топик'] = topicName
        summaryPole.loc[:, 'Токены'] = ', '.join(list(topicLoadingS[topicLoadingS[topicName] * minusPlus > loadingsThreshold].sort_values(topicName, ascending=False).head(tokensLimit).index))
        summaryPole.loc[:, 'Усреднённая связь токена с топиком'] = round(topicLoadingS[topicLoadingS[topicName] * minusPlus > loadingsThreshold].sort_values(topicName, ascending=False).head(tokensLimit).mean()[topicName], 3)
        summaryPole.loc[:, 'Релевантность теме исследования'] = ''
        summaryPole.loc[:, 'Интерпретация топика'] = ''
    return summaryPole

# 1.1 описания каждого топика через его полюса и формирующие их токены и релевантные фрагменты располагаемых на них документов
def snippetByDoc(df, loadingsThreshold, pole, poleDocsIndeceS, poleTokenS, supplementarieS):
    print(f'{pole.capitalize()}ый полюс топика')
    if poleTokenS == []:
        print(f'Величина loadings токенов {pole}ого полюса не достигает заданного порога |{round(loadingsThreshold, 2)}|.'
              , f'Поэтому {pole}ый полюс НЕ выражен и НЕ требует интерпретации')
        docs_snippetS = pandas.DataFrame()
    else:
        print(f'{pole.capitalize()}ый полюс сформирован токен[ом, ами]'
              , str(poleTokenS).replace('[', '').replace(']', ''), '-- по величине вклада')
        print(f'На {pole}ом полюсе расположен[ы] документ[ы]', str(poleDocsIndeceS).replace('[', '').replace(']', ''), '-- по близости к полюсу')
        docs_snippetS = pandas.DataFrame()
        for docIndex in poleDocsIndeceS:
            print(f'Посмотрите на фрагменты документа {docIndex}, содержащие указанные выше токены и их окружение.'
                  , 'Документ:', docIndex)
            row = 0
            docSnippetS = pandas.DataFrame(columns=['min', 'max', 'token'])
            for token in poleTokenS:
                goC = True
                # print('Токен:', token)
                textFull_lemmatized_list = df['textFull_lemmatized'][docIndex].split()
                textFull_list = df['textFull_simbolsCleaned'][docIndex].split()
                tokenProximity = 7  # лимит на длину окружения интересующего токена влево и вправо в нелемматизиованном документе
                tokenPosition = -1
                tokenPositionIntervalInitial = [0, 0]
                while goC:
                    try:
                        # Выяснить номер интересующего токена в списке всех токенов лемматизиованного документа
                        tokenPosition = textFull_lemmatized_list.index(token, tokenPosition + 1)

                        tokenPositionInterval = [tokenPosition - tokenProximity, tokenPosition + tokenProximity]
                        # print('tokenPositionIntervalInitial', tokenPositionIntervalInitial)
                        # print('tokenPositionInterval', tokenPositionInterval)

                        if tokenPositionIntervalInitial[-1] > tokenPositionInterval[0]:
                            # print('tokenPositionIntervalInitial и tokenPositionInterval пересеклись\n'
                            #       , 'Поэтому объединяю их\n'
                            #       , 'И не вывожу промежуточный фрагмент на экран')
                            tokenPositionInterval[0] = tokenPositionIntervalInitial[0]
                            # print('Объединённый tokenPositionInterval', tokenPositionInterval)
                        docSnippetS.loc[row, 'min'] = tokenPositionInterval[0]
                        docSnippetS.loc[row, 'max'] = tokenPositionInterval[-1]
                        docSnippetS.loc[row, 'token'] = token
                        tokenPositionIntervalInitial = tokenPositionInterval.copy()
                        row += 1

                    except ValueError:
                        # print('Поиск в документе', docIndex, 'по токену "', token, '"завершён')
                        goC = False
                # display(docSnippetS)
                docSnippetS = docSnippetS.drop_duplicates('min', keep='last')
                # display(docSnippetS)

            rowInvaderToDropS = []
            tokenReceiverS = poleTokenS.copy()
            for tokenInvader in poleTokenS[:-1]:
                tokenInvaderIndex = docSnippetS[docSnippetS['token'] == tokenInvader].index
                tokenReceiverS.remove(tokenInvader)
                for tokenReceiver in tokenReceiverS:
                    # print('tokenInvader:', tokenInvader, '-> tokenReceiver:', tokenReceiver)
                    tokenReceiverIndex = docSnippetS[docSnippetS['token'] == tokenReceiver].index
                    for rowInvader in tokenInvaderIndex:
                        for rowReceiver in tokenReceiverIndex:
                            # print('Иду по:', rowInvader, rowReceiver)
                            if (docSnippetS['min'][rowInvader] >= docSnippetS['min'][rowReceiver])\
                                    & (docSnippetS['min'][rowInvader] <= docSnippetS['max'][rowReceiver])\
                                | (docSnippetS['max'][rowInvader] >= docSnippetS['min'][rowReceiver])\
                                    & (docSnippetS['max'][rowInvader] <= docSnippetS['max'][rowReceiver])\
                                | (docSnippetS['min'][rowReceiver] >= docSnippetS['min'][rowInvader])\
                                    & (docSnippetS['min'][rowReceiver] <= docSnippetS['max'][rowInvader])\
                                | (docSnippetS['max'][rowReceiver] >= docSnippetS['min'][rowInvader])\
                                    & (docSnippetS['max'][rowReceiver] <= docSnippetS['max'][rowInvader]):
                                # print(rowInvader, rowReceiver)
                                docSnippetS.loc[rowReceiver, 'min'] = min(docSnippetS['min'][rowInvader], docSnippetS['min'][rowReceiver])
                                docSnippetS.loc[rowReceiver, 'max'] = max(docSnippetS['max'][rowInvader], docSnippetS['max'][rowReceiver])
                                docSnippetS.loc[rowReceiver, 'token'] += ' ' + tokenInvader
                                rowInvaderToDropS.append(rowInvader)
            docSnippetS = docSnippetS.drop(rowInvaderToDropS)

            for row in docSnippetS.index:
                # По границам интервала вывести окружение интересующего токена в нелемматизиованном документе
                docSnippetS.loc[row, 'pole'] = pole.capitalize() + 'ый'
                docSnippetS.loc[row, 'textSnippet'] = '..' + ' '.join(textFull_list[docSnippetS['min'][row]: docSnippetS['max'][row]]) + '..'
                for supplementary in supplementarieS:
                    docSnippetS.loc[row, supplementary] = df[supplementary][docIndex]

                print(docSnippetS['textSnippet'][row])
            docs_snippetS = pandas.concat([docs_snippetS, docSnippetS])
        print('\n')
    return docs_snippetS

def randanTopic(dfIn, matrix_df, docsLimit=5, loadingsThreshold=0.5, returnDfs=False, supplementarieS=None, textFull_lemmatized='textFull_lemmatized', textFull_simbolsCleaned='textFull_simbolsCleaned', tokensLimit=10, topicsCount=None):
    '''    Метод тематического моделирования randanTopic основан на методе главных компонент (по-английски: principal components analisys, PCA). То есть он НЕ использует нейросети и embeddings, а работает с "мешком слов" (по-английски: bag of words, BoW). Поэтому важно качественно подготовить "мешок слов" через очистку исходного текста от лишних символов, лемматизацию и удаление стоп-слов. Эти три этапа предобработки исходного текста, а также (при необходимости) автокоррекция грамматических ошибок могут быть выполнены функциями из скрипта textPreprocessor пакета randan https://github.com/RandanCSS/randan/blob/master/randan/tools/textPreprocessor.py . Причём для удобства последующей интерпретации рекомендую результаты очистки от лишних символов и результаты лемматизации сохранить в отдельные столбцы: textFull_simbolsCleaned и textFull_lemmatized соответственно.
        Если для последующей интерпретации Вам пригодятся дополнительные столбцы (скажем, URL, ID, заголовок, дата и т.п.), впишите их в формате списка текстовых объектов в аргумент supplementaries= текущей функции. Причём первый из этих столбцов будет принят алгоритмом в качестве идентификатора анализируемых текстов. Если аргумент supplementaries= останется незаполненным, то тексты будут идентифицироваться так же, как в исходном датафрейме.
    
    Parameters
    --------------------
                   dfIn : DataFrame
      loadingsThreshold : float -- порог (по модулю) усреднённой связи токена с топиком; нужен для отбора токенов, косвенно лимитирует число токенов при интерпретации         
              docsLimit : int -- лимит на число документов на полюсе топика; нужен для отбора документов            
              returnDfs : bool -- в случае True функция возвращает датафреймы с фрагментами документов docs_snippetS и реквизитами документов и вспомогательными переменными scoreS
        supplementarieS : list -- дополнительные столбцы для последующей интерпретации
    textFull_lemmatized : str -- столбец с текстом, прошедшем лемматизацию, но из которого НЕ удалены стоп-слова
    textFull_lemmatized : str -- столбец с текстом, прошедшем удаление лишних символов, но НЕ прошедший лемматизацию и из которого НЕ удалены стоп-слова
            tokensLimit : int -- лимит на число токенов на полюсе топика; нужен для отбора токенов           
            topicsCount : int -- частота самого высокочастотного слова из тех, которые предложены автокорректором в качестве правильного варианта; этот аргумент необходим для корректной работы аргумента userWordS'''
    if (returnDfs == False) & (supplementarieS == None) & (topicsCount == None):
        # print('Пользователь не подал аргументы') # для отладки
        expiriencedMode = False
    else:
        expiriencedMode = True

    if expiriencedMode == False:
        print(
'''    Метод тематического моделирования randanTopic основан на методе главных компонент (по-английски: principal components analisys, PCA). То есть он НЕ использует нейросети и embeddings, а работает с "мешком слов" (по-английски: bag of words, BoW). Поэтому важно качественно подготовить "мешок слов" через очистку исходного текста от лишних символов, лемматизацию и удаление стоп-слов. Эти три этапа предобработки исходного текста, а также (при необходимости) автокоррекция грамматических ошибок могут быть выполнены функциями из скрипта textPreprocessor пакета randan https://github.com/RandanCSS/randan/blob/master/randan/tools/textPreprocessor.py . Причём для удобства последующей интерпретации рекомендую результаты очистки от лишних символов и результаты лемматизации сохранить в отдельные столбцы: textFull_simbolsCleaned и textFull_lemmatized соответственно.
--- После прочтения этой инструкции нажмите Enter'''
              )
        input()

    if 'textFull_lemmatized' not in dfIn.columns:
        print(
f'''В поданном Вами датафрейме отсутствует столбец textFull_lemmatized . Какой столбец в поданном Вами датафрейме содержит лемматизированные тексты (из которого НЕ удалены стоп-слова)?'''
              )
        while True:
            print('--- Впишите, пожалуйста, его название')
            textFull_lemmatized = input()
            if textFull_lemmatized in dfIn.columns: break
            else:
                print('--- Вы вписали что-то не то')

    if 'textFull_simbolsCleaned' not in dfIn.columns:
        print(
f'''В поданном Вами датафрейме отсутствует столбец textFull_simbolsCleaned . Какой столбец в поданном Вами датафрейме содержит тексты, из которых удалены лишние символы, но НЕ лемматизированные?'''
              )
        while True:
            print('--- Впишите, пожалуйста, его название')
            textFull_lemmatized = input()
            if textFull_lemmatized in dfIn.columns: break
            else:
                print('--- Вы вписали что-то не то')  
                    
# 0. Определение желаемого числа топиков
    if topicsCount == None:
        print(
'''Вы уже знаете, сколько топиков хотите, или предпочитаете посмотреть на возможные ориентиры числа топиков, чтобы затем выбрать их число?
--- Если знаете, то введите желаемое число топиков и нажмите Enter
--- Если предпочитаете посмотреть, то просто нажмите Enter -- по умолчанию будут выведены ориентиры для числа топиков не более 30,
поскольку большее число топиков (а) вряд ли захотите проинтерпретировать и (б) расчёт ориентиров для них может занять слишком продолжительное время'''
              )
        while True:
            topicsCount = input()
            try:
                topicsCount = None if topicsCount == '' else int(topicsCount)
                break
            except ValueError:
                print('--- Вы вписали что-то не то'
                      , '\n--- Впишите, пожалуйста, целое число и нажмите Enter: ')

        if topicsCount == None:
            print(
'''Ориентиры для выбора числа топиков: (а) описательная статистика частотности токенов во всём корпусе и в отдельных документах, (б) приемлемый уровнь объяснительной способности модели, критерий Кайзера и скачки объяснительной способности модели (диаграмма "каменистой осыпи"), (в) Ваша субъективная готовность проинтерпретировать не более стольки-то топиков.'''
                  )
# 1.0. Первичный анализ частот токенов в корпусе документов в целом и по каждому документу
# Посчитать сумму по ТОКЕНАМ
            встречаемостьТокенов = pandas.DataFrame(matrix_df.sum(), columns=['Встречаемость токенов'])
            # display(встречаемостьТокенов)
# Посчитать сумму по ДОКУМЕНТАМ
            токеныДокумент = pandas.DataFrame(matrix_df.T.sum(), columns=['Наполненность токенами документов'])
            # display(токеныДокумент)
# 1.1. Описательная статистика для интервальной переменной встречаемостьТокенов
            descriptive_statistics.ScaleStatistics(встречаемостьТокенов[['Встречаемость токенов']])
    # .. и для интервальной переменной токеныДокумент
            descriptive_statistics.ScaleStatistics(токеныДокумент[['Наполненность токенами документов']])

# 2. Пробный запуск (без вращения, но с выводом результатов), чтобы посмотреть `Explained variance`
            dimension_reduction.PCA(n_components='Kaiser' if len(matrix_df.columns) < 30 else 30).fit(matrix_df)
            print('\n--- Введите желаемое число топиков и нажмите Enter')
            while True:
                topicsCount = input()
                try:
                    topicsCount = int(topicsCount)
                    break
                except ValueError:
                    print('--- Вы вписали что-то не то'
                          , '\n--- Впишите, пожалуйста, целое число и нажмите Enter: ')

# 3. Итоговый запуск (с вращением, но с без вывода части результатов)
# Настроить класс PCA: предупредить, сколько будет топиков и какое будет вращение
    if topicsCount > len(matrix_df.columns):
        print('Число топиков принудительно снижено до', len(matrix_df.columns), ', поскольку значение topicsCount, равное', topicsCount, ', слишком велико для располагаемых данных.')
        topicsCount = len(matrix_df.columns)
    pca = dimension_reduction.PCA(n_components=topicsCount, rotation='varimax')
# Подать токены в настроенный класс PCA
    pca = pca.fit(matrix_df, show_results=False)

# 4. Четыре датафрейма, ключевых для оформления и интерпретации результатов тематического моделирования 
    component_loadings_rotated = pca.component_loadings_rotated
    display(component_loadings_rotated) # для отладки
    topicNameS = component_loadings_rotated.columns
    minLoading_among_maxLoadings_list = []
    for topicName in topicNameS:
        minLoading_among_maxLoadings_list.append(component_loadings_rotated[topicName].abs().max())
    minLoading_among_maxLoadings_list.sort()
    minLoading_among_maxLoadings = minLoading_among_maxLoadings_list[0]
    if loadingsThreshold > minLoading_among_maxLoadings:
        print('Порог (по модулю) усреднённой связи токена с топиком снижен до', round(minLoading_among_maxLoadings, 2), ', поскольку значение loadingsThreshold, равное', loadingsThreshold, ', слишком велико для располагаемых данных.')
        loadingsThreshold = minLoading_among_maxLoadings * 0.99 # не выше минимума из максимальных по модулю loadings среди компонент
# Матрица "документы-топики" (и величины в ячейках матрицы названы так же)
    df = dfIn.copy()
    scoreS = pca.transform(matrix_df)
    scoreTextS = scoreS.copy()

    rowsNumerator = df.index
    if supplementarieS != None:
        rowsNumerator = df[supplementarieS[0]]
        df.index = rowsNumerator
        # df = df.drop(supplementarieS[0], axis=1)
        scoreS.index = rowsNumerator
        scoreTextS.index = rowsNumerator
        # print('df.columns :', df.columns) # для отладки
        scoreTextS = pandas.concat([scoreTextS, df[supplementarieS[1:]]], axis=1)
    
    display(scoreTextS.head())
    print('Число столбцов:', scoreTextS.shape[1], ', число строк', scoreTextS.shape[0])

# 5. Цикл для прохода по всем топикам
    poleS = ['лев', 'прав']
    
    summary = pandas.DataFrame()
    docs_snippetS = pandas.DataFrame()
    errorS = []

    wb = Workbook()
    summary_ws = wb.active
    summary_ws.title = "Сводка"

    for topicName in topicNameS:
    # for topicName in topicNameS[:1]: # для отладки
        print('\n\n\nTopic', topicName)

    # Документы-топики
        topicScoreS = scoreS[topicName]
    # Токены-топики
        topicLoadingS = component_loadings_rotated[[topicName]]

        plt.figure(figsize=(9, 9))
# Распределение документов по осям
    # Гистограмма
        plt.subplot(3, 2, 1) # три строки, три столбца, первое место
        plt.hist(topicScoreS.dropna(), color='grey')
        plt.title(f"Docs Distribution by Scores\nin {topicName}")
        plt.xlabel('Scores')
        plt.ylabel('Frequency');
    # Боксплот
        plt.subplot(3, 2, 2) # три строки, три столбца, третье место
        plt.boxplot(topicScoreS.dropna())
        plt.title(f"Docs Distribution by Scores\nin {topicName}")
        plt.xticks([])
        plt.ylabel('Scores')
        # plt.show()
# Распределение токенов по осям
    # Гистограмма
        plt.subplot(3, 2, 3) # три строки, три столбца, седьмое место
        plt.hist(topicLoadingS.dropna(), color='grey')
        plt.title(f"Tokens Distribution by Loadings\nin {topicName}")
        plt.xlabel('Loadings')
        plt.ylabel('Frequency');
    # Боксплот
        plt.subplot(3, 2, 4) # три строки, три столбца, девятое место
        plt.boxplot(topicLoadingS.dropna())
        plt.title(f"Tokens Distribution by Loadings\nin {topicName}")
        plt.xticks([])
        plt.ylabel('Loadings')
        
        plt.tight_layout()
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        plt.show() # обязательно после savefig

    # Полярные документы
        topicDocS = pandas.concat([scoreS.sort_values(topicName)[[topicName]].head(docsLimit), scoreS.sort_values(topicName)[[topicName]].tail(docsLimit)])
    # Полярные токены
        topicTokenS = pandas.concat([topicLoadingS[topicLoadingS[topicName] < -loadingsThreshold].sort_values(topicName).head(tokensLimit)
                                     , topicLoadingS[topicLoadingS[topicName] > loadingsThreshold].sort_values(topicName).tail(tokensLimit)])
        print('Токены на полюсах топика', topicName)
        topicTokenS = topicTokenS.dropna()
        display(topicTokenS)
        
    # Обработка полярных документов и токенов; запись в датафрейм
        summaryMinus = summaryPole(loadingsThreshold, -1, tokensLimit, topicDocS, topicLoadingS, topicName)
        summaryPlus = summaryPole(loadingsThreshold, 1, tokensLimit, topicDocS, topicLoadingS, topicName)
        summary_additional = pandas.concat([summaryMinus, summaryPlus])
        summary = pandas.concat([summary, summary_additional])
        
    # Описание каждого топика через его полюса и формирующие их токены и релевантные фрагменты располагаемых на них документов
        poleMinusTokenS = list(topicTokenS[topicTokenS[topicName] < 0].sort_values(topicName).index)
        poleMinusDocsIndeceS = list(topicDocS[topicDocS[topicName] < 0].sort_values(topicName).index)
        minusDocs_snippetS = snippetByDoc(df, loadingsThreshold, poleS[0], poleMinusDocsIndeceS, poleMinusTokenS, supplementarieS)
        polePlusTokenS = list(topicTokenS[topicTokenS[topicName] > 0].index)
        polePlusDocsIndeceS = list(topicDocS[topicDocS[topicName] > 0].index)
        plusDocs_snippetS = snippetByDoc(df, loadingsThreshold, poleS[-1], polePlusDocsIndeceS, polePlusTokenS, supplementarieS)
        
        docs_snippetS_additional = pandas.concat([minusDocs_snippetS, plusDocs_snippetS])
        docs_snippetS_additional = docs_snippetS_additional.drop(['min', 'max'], axis=1)
        docs_snippetS_additional = docs_snippetS_additional.reset_index(drop=True)
        docs_snippetS_additional.loc[:, 'Интерпретация топика'] = ''
        docs_snippetS = pandas.concat([docs_snippetS, docs_snippetS_additional])
        # print('\n\n\n')

        ws = wb.create_sheet(title=topicName)
        for r in dataframe_to_rows(docs_snippetS_additional, index=False, header=True):
            ws.append(r)
        img_data.seek(0)
        img = Image(img_data)
        ws.add_image(img, "K1")

    docs_snippetS = docs_snippetS.reset_index(drop=True)

# 6. Если норм, отправить эти документы в Excel
    # Желательно после начала интерпретации назвать Excel по-новому, чтобы он не перезаписался в дальнейшем.
    summaryColumns = ['Топик', 'Токены', 'Усреднённая связь токена с топиком']
    summaryColumns.extend(scoreS.columns)
    summaryColumns.extend(['Релевантность теме исследования', 'Интерпретация топика'])

    for r in dataframe_to_rows(summary[summaryColumns], index=False, header=True):
        summary_ws.append(r)

    fileName = 'Топики'
    attempt = 0
# Чтобы не перезаписать ранее созднный файл, в котором может быть интерпретация
    while os.path.exists(fileName + ' ' + str(attempt) + ".xlsx"):
        attempt += 1

    print(f'Cоздаю файл "{fileName + ' ' + str(attempt)}.xlsx"')
    wb.save(fileName + ' ' + str(attempt) + ".xlsx")

    # # Причём разместить релевантные фрагменты документов на отдельныъх страницах
    # with pandas.ExcelWriter("Топики.xlsx") as writer:
    #     summary[summaryColumns].to_excel(writer, sheet_name="Сводка")
    #     for topicName in scoreS.columns:
    #         docs_snippetS[docs_snippetS['topicName'] == topicName].to_excel(writer, sheet_name=topicName)

    print(
'''Рекомендации по дальнейшей интерпретации топиков:
1.1. Интерпретация топиков по ключевым токенам: на листе Сводка посмотрите перечень ключевых токенов каждого топика. Если токенов слишком мало или много, перезапустите скрипт, понизив или повысив порог loadingsThreshold соответственно, либо повысив или понизив порог tokensLimit соответственно.
1.2. Попробуйте в одной или нескольких фразах сормулировать смысл каждого топика по его ключевым токенам и внесите сформулированный смысл в столбец Интерпретация топика. Если у топика два полюса, лучше для каждого из них формулировать смысл отдельно.
2.1. Интерпретация топиков по выдержкам из ключевых документов и ключевым токенам: на листе Сводка посмотрите перечень ключевых документов каждого топика. Если документов слишком мало или много, перезапустите скрипт, повысив или понизив порог docsLimit соответственно.
2.2. На листе PC1_vrmx читайте выдержки (с акцентом на соответствующие ключевые токены) и при обнаружении принципиально нового смысла, вносите его в свободный столбец. Нюансы и оттенки уже выписанных смыслов лучше игнорировать, чтобы не "закопаться".
2.3. Повторите процедуру для остальных листов PC..._vrmx. По завершении постарайтесь в одной или нескольких фразах обобщить смыслы для каждого листа (топика) и внесите сформулированные обобщения в столбец Интерпретация топика листа Сводка. Если у топика два полюса, лучше для каждого из них формулировать смысл отдельно.
3.1. Интерпретация топиков по ключевым документам и ключевым токенам с помощью ИИ: на листе Сводка посмотрите перечень ключевых токенов и документов каждого топика. Если их слишком мало или много, выполните пункты 2.1 или 3.1.
3.2. На листе Сводка выберите несколько (можете ориентироваться на диаграмму Docs distribution by scores...) ключевых документов с максимальным значением в столбце PC1_vrmx , подайте их в ИИ с просьбой вычленить основные мысли с привязкой к соответствующим ключевым токенам.
3.3. Ответ ИИ внесите в столбец Интерпретация топика напротив первого топика. Повторите процедуру для остальных топиков.'''
          )
    if returnDfs: return docs_snippetS, scoreS
