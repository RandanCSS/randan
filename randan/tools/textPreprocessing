#!/usr/bin/env python
# coding: utf-8

'''
A module for an unstandardized text preprocessing
'''

# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        import os, pandas, pymystem3, re, stop_words
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[1] 
        print('Пакет', module,
              'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

def pymystemLemmatizer(dfIn, columnWithText):
    df = dfIn.copy()
    print('Эта функция предназначена для лмматизации текстов пакетом pymystem3', end='\r')
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
    print('Эта функция предназначена для чистки текстов от невербального мусора (ненужных символов)', end='\r')
    cleaned_text = ''
    for a in text:
        if (a.isalnum()) or (a == ' '):
            cleaned_text = cleaned_text + a
    while '  ' in cleaned_text:
        cleaned_text = cleaned_text.replace('  ', ' ')
    return cleaned_text

def stopwordsDropper(text):
    print('Эта функция предназначена для чистки текстов от стоп-слов пакетом stop_words', end='\r')
    stopwords_list = stop_words.get_stop_words('russian')
    text_cleaned = ''
    for word in text.split(' '):
        if word not in stopwords_list:
            text_cleaned += ' ' + word
    text_cleaned = text_cleaned.strip()
    # text_cleaned = text_cleaned.split(' ')
    return text_cleaned
