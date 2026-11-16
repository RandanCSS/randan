#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module for processing two (semi)standardized text variables: brandingSettings.channel.keywords or snippet.tags and topic Details.topic Categories. They are variables from the output of the YouTube API methods `channels` or `videos`. As well the module creates a column with a channels's or video's absolute link
(RU) Модуль для обработки двух (полу)стандартизированных текстовых переменных выгрузки методов channels или videos API YouTube: brandingSettings.channel.keywords или snippet.tags и topicDetails.topicCategories, а также создания столбца с абсолютной ссылкой на канала или видео
'''

# Активировать требуемые для работы скрипта модули и пакеты + пререквизиты
# В общем случае требуются следующие модули и пакеты (запасной код, т.к. они прописаны в setup)
# subprocess & sys -- эти пакеты обычно предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
from subprocess import check_call, CalledProcessError
import sys

# --- остальные модули и пакеты
MAX_ATTEMPTS = 3
attempt = 1

while True:
    try:
        import pandas, warnings
        break # выход из цикла while True

    except (ImportError, ModuleNotFoundError) as excptn_1:
        module = (excptn_1.name or '').split('.')[0]
        if attempt > MAX_ATTEMPTS:
            print(
f'Пакет {module} НЕ удалось импортировать за {MAX_ATTEMPTS} попытки; он требуется для работы скрипта, поэтому попробуйте инсталлировать его вручную, после чего снова запустите скрипт')

            raise

        print(
f'Пакет {module} НЕ прединсталлирован, но он требуется для работы скрипта, поэтому будет инсталлирован сейчас. Попытка № {attempt} из {MAX_ATTEMPTS}')

        try: check_call([sys.executable, '-m', 'pip', 'install', module, '--quiet', '--disable-pip-version-check'])
        except CalledProcessError as excptn_2:
            print(f"Не удалось установить {module}. {type(excptn_2).__name__}: {str(excptn_2).split('Stacktrace:')[0].strip()}")
            raise

        attempt += 1

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
            # print('var:', var) # для отладки
            notNanDf[var] = notNanDf[var].apply(lambda cellContent: textPreprocessor.simbolsCleaner(cellContent))
            notNanDf[var] = textPreprocessor.pymystemLemmatizer(notNanDf, var)
            notNanDf[var] = notNanDf[var].apply(lambda cellContent: textPreprocessor.stopwordsDropper(cellContent))
        df = pandas.concat([nanDf, notNanDf])
    
    df['URL'] = f'https://www.youtube.com/{contentType}/' + df['id'] # столбец с абсолютной ссылкой
    
    display(df[[varS[0], 'https://en.wikipedia.org/wiki/', 'URL']].tail())
    print('Число строк', df.shape[0])
    return df
