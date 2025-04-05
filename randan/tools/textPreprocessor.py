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

def autoCorrectorText(fast, language, text, userWordS):
    spellerInstance = Speller(fast=fast, lang=language) # настройки: выбор языка
    spellerInstance(text) # применение к тексту
    if userWordS != None:
        for word in userWordS: spellerInstance.nlp_data[word] = 0

    # # Оставить только уникальные слова (для ускорения работы)
    # tokenS = text.split(' ')
    # tokenS = list(dict.fromkeys(tokenS))
    # textCorrected = spellerInstance(' '.join(tokenS))

    correctionResults = {}
    corrections = {}
    textCorrected = spellerInstance(text)
    tokenS = text.split(' ')
    tokensCorrectedActual = textCorrected.split(' ')
    tokensCorrectedPotential = tokensCorrectedActual.copy()
    for tokenPosition in range(len(tokenS)):
        # print('tokenPosition:', tokenPosition)
        token = tokenS[tokenPosition]
        tokenCorrected = tokensCorrectedPotential[tokenPosition]
        # print(f'Проверяю токен {token}, его порядковый номер: {tokenPosition} -- из {len(tokenS)}', '                    ') #, end='\r'
        # print('                                                                                ', end='\r')
        if token == tokenCorrected: tokensCorrectedActual.remove(tokenCorrected)
        else:
            # print(f'Обнаружил несоответствие: token: "{token}", tokenCorrected: "{tokenCorrected}"')
            corrections[token] = tokenCorrected
    correctionCoefficience = round(len(corrections.keys()) / len(numpy.unique(tokenS)), 2)
    correctionResults['correctionCoefficience'] = correctionCoefficience
    correctionResults['textCorrected'] = textCorrected
    corrections = dict(sorted(corrections.items()))
    correctionResults['corrections'] = corrections
    return correctionResults

def autoCorrectorTextS(columnWithText, dfIn, fast=False, language='ru', probeMode=True, userWordS=None):
    df = dfIn.copy()
    if probeMode:
        dfProbe = df.sample(min(100, int(round(len(df) / 10, 0))))
        # display(dfProbe) # для отладки
        print('\nПробный запуск автокорректора')
        dfProbe['correctionResults'] = dfProbe[columnWithText].progress_apply(lambda text: autoCorrectorText(fast=False, language=language, text=text, userWordS=userWordS))
        correctionCoefficienceMean = dfProbe['correctionResults'].apply(lambda dictionary: dictionary['correctionCoefficience']).mean()

        dfIncorrected = df.drop(dfProbe.index)
        df = pandas.concat([dfProbe, dfIncorrected])

        print(f'Расчитанный на выборке текстов коэффициент исправлений равен {round(correctionCoefficienceMean, 2)}%')
        if correctionCoefficienceMean < 5: print('Поскольку он меньше 5% , применение автокорректора не рекомендуется или требует предварительного рассмотрения предлагаемых автокорректором исправлений')
        else: print('Поскольку он НЕ меньше 5% , применение автокорректора рекомендуется')
        print(
'''--- Для полноценного (на всех текстах) запуска функции autoCorrectorTextS , укажите в её круглых скобках: probeMode=False
--- Перед запуском желательно предварительно рассмотреть предлагаемые автокорректором исправления (код: df['correctionResults'].apply(lambda dictionary: dictionary['corrections']) )
и, возможно, добавить список дополнений в словарь правильных слов, указав в круглых скобках userWordS=[список закавыченных слов через запятую с пробелом]'''
              )
    else:
        print(
'''
Полноценный (на всех текстах) запуск автокорректора. Приготовьтесь, что если тексты большие, то на каждые 3 текста уйдёт минута, а то и больше.
Для ускорения можно в круглых скобках функции autoCorrectorTextS указать: fast=True'''
              )
        if 'correctionResults' in df.columns:
            if len(df) > len(df[df['correctionResults'].notna()]):
                dfCorrected = df[df['correctionResults'].notna()]
                dfIncorrected = df[df['correctionResults'].isna()]
                dfIncorrected['correctionResults'] = dfIncorrected[columnWithText].progress_apply(lambda text: autoCorrectorText(fast=False, language=language, text=text, userWordS=userWordS))
                df = pandas.concat([dfCorrected, dfIncorrected]).sort_index()
            else:
                print('--- Автокоррекция всех текстов уже выполнялась, если хотите повторить, то удалите из Вашего датафрейма столбец correctionResults и перезапустите функцию autoCorrectorTextS')
                warnings.filterwarnings("ignore")
                sys.exit()
        else: df['correctionResults'] = df[columnWithText].progress_apply(lambda text: autoCorrectorText(fast=False, language=language, text=text, userWordS=userWordS))
        correctionCoefficienceMean = df['correctionResults'].apply(lambda dictionary: dictionary['correctionCoefficience']).mean()
        print(f'Расчитанный на всех текстах коэффициент исправлений равен {round(correctionCoefficienceMean, 2)}%')

    corrections = {}
    for row in df['correctionResults'].dropna().apply(lambda dictionary: dictionary['corrections']):
        for key in row.keys():
            corrections[key] = row[key]
    print('\nСловарь предлагаемых автокорректором исправлений (первые 100):', dict(itertools.islice(corrections.items(), 100)))
    return corrections, df

def multispaceCleaner(text):
    textCleaned = text
    while '  ' in textCleaned: textCleaned = textCleaned.replace('  ', ' ')
    while textCleaned[0] == ' ': textCleaned = textCleaned[1:] # избавиться от пробелов в начале текста
    while textCleaned[-1] == ' ': textCleaned = textCleaned[:-1] # избавиться от пробелов в конце текста
    return textCleaned

def pymystemLemmatizer(dfIn, columnWithText):
    """
    Функция для лемматизации текстов пакетом pymystem3

    Parameters
    ----------
              dfIn : DataFrame
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
    Функция для чистки текстов от невербального мусора (ненужных символов)

    Parameters
    ----------
    text : str
    """
    textCleaned = ''
    for a in text:
        if (a.isalnum()) | (a == ' '): textCleaned += a
        else: textCleaned += ' ' # чтобы при удалении невербального мусора, за которым не следует пробел, оставшиеся символы не сливались
    textCleaned = multispaceCleaner(textCleaned)
    return textCleaned

def stopwordsDropper(text, userStopWordsToAdd=None, userStopWordsToRemove=None):
    """
    Функция для чистки текстов от стоп-слов пакетом stop_words

    Parameters
    --------------------
                 text : str
   userStopWordsToAdd : list
userStopWordsToRemove : list
    """
    stopWordS = stop_words.get_stop_words('russian')
    if userStopWordsToAdd != None: stopWordS.extend(userStopWords)
    if userStopWordsToRemove != None:
        for word in userStopWordsToRemove: stopWordS.remove(word)
    textCleaned = ''
    for word in text.split(' '):
        if word not in stopWordS:
            textCleaned += ' ' + word
    textCleaned = textCleaned.strip()
    # textCleaned = textCleaned.split(' ')
    return textCleaned
