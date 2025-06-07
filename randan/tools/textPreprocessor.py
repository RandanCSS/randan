#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for an unstandardized text preprocessing
(RU) Авторский модуль для предобработки нестандартизированнрого текста
'''

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from autocorrect import Speller
        import itertools, numpy, os, pandas, pymystem3, re, stop_words, time, warnings
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

def autoCorrectorText(fast, language, text, tokensCorrectedQuantityMax, userWordS):
    """
    Функция для автокоррекции грамматики текста
    Parameters
    --------------------
                      fast : bool -- режим упрощённой автокоррекции и, как следствие, ускоренной
                  language : str
                      text : str
tokensCorrectedQuantityMax : int -- частота самого высокочастотного слова из тех, которые предложены автокорректором в качестве правильного варианта; этот аргумент необходим для корректной работы аргумента userWordS
                 userWordS : list -- слова, добавляемые пользователем в словарь грамматически корректных слов (не будут исправляться автокорректором)
    """
    spellerInstance = Speller(fast=fast, lang=language) # настройки: выбор языка
    if userWordS != None:
        for word in userWordS:
            spellerInstance.nlp_data[word] = tokensCorrectedQuantityMax
            # print('word:', word) # для отладки
            # print('word in spellerInstance.nlp_data:', word in spellerInstance.nlp_data) # для отладки

    # # Оставить только уникальные слова (для ускорения работы)
    # tokenS = text.split(' ')
    # tokenS = list(dict.fromkeys(tokenS))
    # textCorrected = spellerInstance(' '.join(tokenS))

    correctionResults = {}
    corrections = {}
    textCorrected = spellerInstance(text) # применение к тексту
    tokenS = text.split(' ')
    tokensCorrectedActual = textCorrected.split(' ')
    tokensCorrectedPotential = tokensCorrectedActual.copy()
    tokensCorrectedQuantityMax = 0
    for tokenPosition in range(len(tokenS)):
        # print('tokenPosition:', tokenPosition)
        token = tokenS[tokenPosition]
        tokenCorrected = tokensCorrectedPotential[tokenPosition]
        # print(f'Проверяю токен {token}, его порядковый номер: {tokenPosition} -- из {len(tokenS)}', '                    ') #, end='\r'
        # print('                                                                                ', end='\r')
        if token == tokenCorrected: tokensCorrectedActual.remove(tokenCorrected)
        else:
            # print(f'Обнаружил несоответствие: token: "{token}", tokenCorrected: "{tokenCorrected}"') # для отладки
            corrections[token] = tokenCorrected
            # print('spellerInstance.nlp_data[tokenCorrected]:', spellerInstance.nlp_data[tokenCorrected]) # для отладки
            if tokenCorrected in spellerInstance.nlp_data.keys(): # как ни странно, автокоррект иногда берёт слова для исправления не из словаря своего (например, "Щёлкнул")
                if spellerInstance.nlp_data[tokenCorrected] > tokensCorrectedQuantityMax: tokensCorrectedQuantityMax = spellerInstance.nlp_data[tokenCorrected]
    correctionCoefficience = 100 * round(len(corrections.keys()) / len(numpy.unique(tokenS)), 4)
    correctionResults['correctionCoefficience'] = correctionCoefficience
    corrections = dict(sorted(corrections.items()))
    correctionResults['corrections'] = corrections
    correctionResults['tokensCorrectedQuantityMax'] = tokensCorrectedQuantityMax
    correctionResults['textCorrected'] = textCorrected
    return correctionResults

def autoCorrectorTextS(columnWithText, dfIn, fast=False, language='ru', probeMode=True, tokensCorrectedQuantityMax=0, userWordS=None):
    """
    Функция для автокоррекции грамматики текстов, организованных в формате столбца датафрейма

    Parameters
    --------------------
            columnWithText : str
                      dfIn : pandas DataFrame
                      fast : bool -- режим упрощённой автокоррекции и, как следствие, ускоренной
                  language : str
                 probeMode : bool -- режим пробной автокоррекции, выполняемой на 10% вероятностно отобранных текстов из всего корпуса текстов (но не более 100 текстов)
tokensCorrectedQuantityMax : int -- частота самого высокочастотного слова из тех, которые предложены автокорректором в качестве правильного варианта; этот аргумент необходим для корректной работы аргумента userWordS
                 userWordS : list -- слова, добавляемые пользователем в словарь грамматически корректных слов (не будут исправляться автокорректором)
    """
    df = dfIn.copy()
    if probeMode:
        dfProbe = df.sample(min(100, int(round(len(df) / 10, 0))))
        # display(dfProbe) # для отладки
        print('\nПробный запуск автокорректора')
        dfProbe['correctionResults'] = dfProbe[columnWithText].progress_apply(lambda text: autoCorrectorText(fast=False, language=language, text=text, tokensCorrectedQuantityMax=tokensCorrectedQuantityMax, userWordS=userWordS))
        correctionCoefficienceMean = dfProbe['correctionResults'].apply(lambda dictionary: dictionary['correctionCoefficience']).mean()

        dfIncorrected = df.drop(dfProbe.index)
        df = pandas.concat([dfProbe, dfIncorrected])

        print(f'Расчитанный на выборке текстов коэффициент исправлений равен {round(correctionCoefficienceMean, 2)}%')
        if correctionCoefficienceMean < 5: print('Поскольку он меньше 5% , применение автокорректора не рекомендуется или требует предварительного рассмотрения предлагаемых автокорректором исправлений')
        else: print('Поскольку он НЕ меньше 5% , применение автокорректора рекомендуется')
        print(
'''--- Для полноценного (на всех текстах) запуска функции autoCorrectorTextS , укажите в её круглых скобках: probeMode=False
--- Перед запуском желательно предварительно рассмотреть предлагаемые автокорректором исправления (код: df['correctionResults'].apply(lambda dictionary: dictionary['corrections']) )
и, возможно, добавить список дополнений в словарь правильных слов, указав в круглых скобках userWordS=[список закавыченных слов через запятую с пробелом],
а также внести в tokensCorrectedQuantityMax= то число tokensCorrectedQuantityMax , которые Вы получите чуть ниже'''
              )
    else:
        print(
'''
Полноценный (на всех текстах) запуск автокорректора. Приготовьтесь, что если тексты большие, то на каждые 3 текста уйдёт минута, а то и больше.
Для ускорения можно в круглых скобках функции autoCorrectorTextS указать: fast=True'''
              )
        if ('correctionResults' in df.columns) & (userWordS == None): # автокоррекция уже выполнялась пробно и пользователь НЕ запросил подстройку словаря, поэтому она будет применена только к остальным текстам
            dfCorrected = df[df['correctionResults'].notna()]
            dfIncorrected = df[df['correctionResults'].isna()]
        else: # автокоррекция НЕ выполнялась пробно или пользователь запросил подстройку словаря, поэтому она будет применена ко ВСЕМ текстам
            dfCorrected = pandas.DataFrame()
            dfIncorrected = df.copy()
        dfIncorrected['correctionResults'] = dfIncorrected[columnWithText].progress_apply(lambda text: autoCorrectorText(fast=False, language=language, text=text, tokensCorrectedQuantityMax=tokensCorrectedQuantityMax, userWordS=userWordS))
        df = pandas.concat([dfCorrected, dfIncorrected]).sort_index()
# print('--- Автокоррекция всех текстов уже выполнялась, если хотите повторить, то удалите из Вашего датафрейма столбец correctionResults и перезапустите функцию autoCorrectorTextS')
# warnings.filterwarnings("ignore")
# sys.exit()
# else: df['correctionResults'] = df[columnWithText].progress_apply(lambda text: autoCorrectorText(fast=False, language=language, text=text, tokensCorrectedQuantityMax=tokensCorrectedQuantityMax, userWordS=userWordS))
        correctionCoefficienceMean = df['correctionResults'].apply(lambda dictionary: dictionary['correctionCoefficience']).mean()
        print(f'Расчитанный на всех текстах коэффициент исправлений равен {round(correctionCoefficienceMean, 4)}%')

    corrections = {}
    for row in df['correctionResults'].dropna().apply(lambda dictionary: dictionary['corrections']):
        for key in row.keys():
            corrections[key] = row[key]
    corrections = dict(sorted(corrections.items()))

    tokensCorrectedQuantityMax = df['correctionResults'].dropna().apply(lambda dictionary: dictionary['tokensCorrectedQuantityMax']).max()

    print('\nСловарь предлагаемых автокорректором исправлений (первые 100):', dict(itertools.islice(corrections.items(), 100)), '\ntokensCorrectedQuantityMax:', tokensCorrectedQuantityMax)
    return corrections, df, tokensCorrectedQuantityMax

def multispaceCleaner(text):
    textCleaned = text
    if len(text) > 0: # т.к., например, после чистки текста от эмодзи в нём может остаться пустота
        while '  ' in textCleaned: textCleaned = textCleaned.replace('  ', ' ')
        while textCleaned[0] == ' ': textCleaned = textCleaned[1:] # избавиться от пробелов в начале текста
        while textCleaned[-1] == ' ': textCleaned = textCleaned[:-1] # избавиться от пробелов в конце текста
    return textCleaned

def pymystemLemmatizer(dfIn, columnWithText):
    """
    Функция для лемматизации текстов пакетом pymystem3

    Parameters
    ----------
              dfIn : pandas DataFrame
    columnWithText : str
    """
    df = dfIn.copy()
    mstem = pymystem3.Mystem()
    separator = r'|||'
    while len(df[df[columnWithText].str.contains(separator)]) == 0:
        print('--- Не получается использовать разделитель', separator, 'по которому тексты из всех ячеек сначала объединятся в один, а потом снова сепарируются'
              , '\n--- Придумайте другой разделитель, не забывая об особенностях спецсимволов, впишите его и нажмите Enter')
        separator = input()
    textS = df[columnWithText].tolist()
    textS = mstem.lemmatize(separator.join(textS))
    textS = ' '.join(textS)
    df[columnWithText] = textS.split(separator)
    df[columnWithText] = df[columnWithText].str.strip() # убрать появившиеся после лемматизации \n на концах лемматизированных текстов
    df[columnWithText] = df[columnWithText].apply(lambda text: re.sub(r'  +', ' ', text))
    return df[columnWithText]

def simbolsCleaner(text):
    """
    Функция для чистки текстов от лишних символов (не являющихся буквами или цифрами)

    Parameters
    ----------
    text : str
    """
    textCleaned = ''
    for a in text:
        if (a.isalnum()) | (a == ' '): textCleaned += a
        else: textCleaned += ' ' # чтобы при удалении лишних символов, за которым не следует пробел, оставшиеся символы не сливались
    textCleaned = multispaceCleaner(textCleaned)
    return textCleaned

def stopwordsDropper(text, userStopWordsToAdd=None, userStopWordsToRemove=None, language='russian'):
    """
    Функция для чистки текстов от стоп-слов пакетом stop_words

    Parameters
    --------------------
                 text : str
   userStopWordsToAdd : list -- слова, добавляемые пользователем в список стоп-слов (будут удалены из обрабатываемого текста)
userStopWordsToRemove : list -- слова, исключаемые пользователем из списка стоп-слов (будут сохранены в обрабатываемом тексте)
             language : str
    """
    stopWordS = stop_words.get_stop_words(language)
    if userStopWordsToAdd != None: stopWordS.extend(userStopWordsToAdd)
    if userStopWordsToRemove != None:
        for word in userStopWordsToRemove: stopWordS.remove(word)
    textCleaned = ''
    for word in text.split(' '):
        if word not in stopWordS:
            textCleaned += ' ' + word
    textCleaned = textCleaned.strip()
    # textCleaned = textCleaned.split(' ')
    return textCleaned
