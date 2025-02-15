#!/usr/bin/env python
# coding: utf-8

'''
A module for processing two (semi)standardized text variables: snippet.tags, topic Details.topic Categories. They are variables from the output of the YouTube API method `videos`. As well the module creates a column with a video's absolute link
'''

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
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

def tag_topic_URL(videoS):
    """
    Функция для обработки двух (полу)стандартизированных текстовых переменных выгрузки метода videos: snippet.tags, topicDetails.topicCategories, а также создания столбца с абсолютной ссылкой

    Parameters
    ----------
    videoS : DataFrame
    """
    warnings.filterwarnings("ignore")
    varS = ['snippet.tags', 'topicDetails.topicCategories']
    for var in varS:
        nanVideoS = videoS[videoS[var].isna()]
        notNanVideoS = videoS[videoS[var].notna()]
        notNanVideoS[var] = notNanVideoS[var].apply(lambda content: ' '.join(content))
        if var == 'snippet.tags': # расщепить на отдельные слова и удалить средли них дубликаты
             notNanVideoS[var] =  notNanVideoS[var].str.split().apply(lambda content: list(dict.fromkeys(content))).apply(lambda content: ' '.join(content))
        if var == 'topicDetails.topicCategories':
            notNanVideoS['https://en.wikipedia.org/wiki/'] = notNanVideoS[var].str.replace('https://en.wikipedia.org/wiki/', '')
        videoS = pandas.concat([nanVideoS, notNanVideoS])

    videoS['URL'] = 'https://www.youtube.com/video/' + videoS['id'] # столбец с абсолютной ссылкой

    display(videoS[['snippet.tags', 'https://en.wikipedia.org/wiki/', 'URL']].tail())
    print('Число строк', videoS.shape[0])
    return videoS
