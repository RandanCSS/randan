#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for processing two (semi)standardized text variables: brandingSettings.channel.keywords or snippet.tags and topic Details.topic Categories. They are variables from the output of the YouTube API methods `channels` or `videos`. As well the module creates a column with a channels's or video's absolute link
(RU) Авторский модуль для обработки двух (полу)стандартизированных текстовых переменных выгрузки методов channels или videos API YouTube: brandingSettings.channel.keywords или snippet.tags и topicDetails.topicCategories, а также создания столбца с абсолютной ссылкой на канала или видео
'''

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys, warnings
from randan.tools import textPreprocessor
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        import pandas, warnings
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '').replace('_', '')
        if '.' in module: module = module.split('.')[1]
        print('Пакет', module,
              'НЕ прединсталлируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталлирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])
        attempt += 1
        if  attempt == 10:
            print('Пакет', module
                  , 'НЕ прединсталлируется с установкой Анаконды, для работы скрипта требуется этот пакет,'
                  , 'но инсталлировать его не удаётся, попробуйте инсталлировать его вручную, после чего снова запустите требуемый скрипт пакета\n')
            break

def keywordTagTopicURL(dfIn):
    """
    Функция для обработки двух (полу)стандартизированных текстовых переменных выгрузки метода videos: snippet.tags, topicDetails.topicCategories, а также создания столбца с абсолютной ссылкой

    Parameters
    ----------
    dfIn : DataFrame
    """
    df = dfIn.copy()
    contentType = 'channel' if 'brandingSettings.channel.keywords' in df.columns else 'video'
    
    warnings.filterwarnings("ignore")
    varS = ['brandingSettings.channel.keywords' if contentType == 'channel' else 'snippet.tags', 'topicDetails.topicCategories']
    for var in varS:
        nanDf = df[df[var].isna()]
        notNanDf = df[df[var].notna()]
        if var == 'brandingSettings.channel.keywords': notNanDf[var] = notNanDf[var].str.split(' ')
        notNanDf[var] = notNanDf[var].apply(lambda cellContent: ' '.join(cellContent))
        if var == 'topicDetails.topicCategories': notNanDf['https://en.wikipedia.org/wiki/'] = notNanDf[var].str.replace('https://en.wikipedia.org/wiki/', '')
        else: # расщепить на отдельные слова и удалить среди них дубликаты
            notNanDf[var] = notNanDf[var].apply(lambda cellContent: textPreprocessor.simbolsCleaner(cellContent))
            notNanDf[var] = textPreprocessor.pymystemLemmatizer(notNanDf, var)
            notNanDf[var] = notNanDf[var].apply(lambda cellContent: textPreprocessor.stopwordsDropper(cellContent))
        df = pandas.concat([nanDf, notNanDf])
    
    df['URL'] = f'https://www.youtube.com/{contentType}/' + df['id'] # столбец с абсолютной ссылкой
    
    display(df[[varS[0], 'https://en.wikipedia.org/wiki/', 'URL']].tail())
    print('Число строк', df.shape[0])
    return df
