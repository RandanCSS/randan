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
        import os, pandas, pymystem3, re, stop_words, time
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

def stopwordsDropper(text, userStopWords=None):
    """
    Функция для чистки текстов от стоп-слов пакетом stop_words

    Parameters
    ----------
         text : str
userStopWords : list
    """
    stopWordS = stop_words.get_stop_words('russian')
    if userStopWords != None: stopWordS.extend(userStopWords)
    textCleaned = ''
    for word in text.split(' '):
        if word not in stopWordS:
            textCleaned += ' ' + word
    textCleaned = textCleaned.strip()
    # textCleaned = textCleaned.split(' ')
    return textCleaned
