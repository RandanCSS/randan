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
    cleaned_text = text
    while '  ' in text: cleaned_text = text.replace('  ', ' ')
    while cleaned_text[0] == ' ': cleaned_text = cleaned_text[1:] # избавиться от пробелов в начале текста
    while cleaned_text[-1] == ' ': cleaned_text = cleaned_text[:-1] # избавиться от пробелов в конце текста
    return cleaned_text

def pymystemLemmatizer(dfIn, columnWithText):
    """
    Функция для лемматизации текстов пакетом pymystem3

    Parameters
    ----------
              dfIn : DataFrame
    columnWithText : str
    """
    df = dfIn.copy()
    time.sleep(0.01)
    # print('                                                                                          ', end='\r')
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
    df[columnWithText] = df[columnWithText].apply(lambda text: re.sub(r'  +', ' ', text))
    return df[columnWithText]

def simbolsCleaner(text):
    """
    Функция для чистки текстов от невербального мусора (ненужных символов)

    Parameters
    ----------
    text : str
    """
    time.sleep(0.01)
    # print('                                                                                          ', end='\r')
    cleaned_text = ''
    for a in text:
        if (a.isalnum()) | (a == ' '): cleaned_text += a

    while '  ' in cleaned_text: cleaned_text = cleaned_text.replace('  ', ' ')
    while cleaned_text[0] == ' ': cleaned_text = cleaned_text[1:] # избавиться от пробелов в начале текста
    while cleaned_text[-1] == ' ': cleaned_text = cleaned_text[:-1] # избавиться от пробелов в конце текста
    return cleaned_text

def stopwordsDropper(text):
    """
    Функция для чистки текстов от стоп-слов пакетом stop_words

    Parameters
    ----------
    text : str
    """
    time.sleep(0.01)
    # print('                                                                                          ', end='\r')
    stopwords_list = stop_words.get_stop_words('russian')
    text_cleaned = ''
    for word in text.split(' '):
        if word not in stopwords_list:
            text_cleaned += ' ' + word
    text_cleaned = text_cleaned.strip()
    # text_cleaned = text_cleaned.split(' ')
    return text_cleaned
