#!/usr/bin/env python
# coding: utf-8

'''
A module for creating a matrix `documents-tokens` from a corpus of documents using the methods of the CountVectorizer and TfidfVectorizer classes
Модуль для создания матрицы "документы-токенц" из корпуса документов посредством методов классов CountVectorizer и TfidfVectorizer
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
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        import pandas
        break # выход из цикла while True

    except (ImportError, ModuleNotFoundError) as excptn_1:
        module = excptn_1.name.split('.')[0]
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

# Процесс векторизации
def vectProcess(frequencyMetric, column, df, min_df, ngram=1, returnVect=False):
    df = df[df[column].notna()]
    vect = CountVectorizer(min_df=min_df, ngram_range=(ngram, ngram)).fit(df[column]) if frequencyMetric == 'c' else TfidfVectorizer(min_df=min_df, ngram_range=(ngram, ngram)).fit(df[column])
    matrix = vect.transform(df[column]) # оформить новый датафрейм..
    matrix_df = pandas.DataFrame(matrix.toarray(), columns=vect.get_feature_names_out(), index=df.index) # .. у которого
        # по столбцам слова из столбца column, а по строкам -- индексы строк исходного датафрейма df
    # # Показалось, что требуются строки с ненулевой суммой частот
    # matrix_df_T_Sum = matrix_df.T.sum()
    # df = df[matrix_df_T_Sum == 0]
    matrix_df = pandas.DataFrame(matrix.toarray(), columns=vect.get_feature_names_out(), index=df.index)
    print('\nАбсолютные частоты' if frequencyMetric == 'c' else '\nОтносительные частоты TF-IDF')
    display(matrix_df.head())
    print('Число столбцов:', matrix_df.shape[1], ', число строк', matrix_df.shape[0])
    print('\nЧастотность токенов:')
    display(matrix_df.sum().sort_values(ascending=False))
    print(
'''--- Если хотите посмотреть топ мешка токенов глубже, чем сейчас он выведен, можете использовать в следующем чанке такой код:
matrix_df.sum().sort_values(ascending=False).head(25) , где 25 -- требуемая глубина (максимально можно увеличить до 50)'''
          )
    if returnVect:
        print('matrix_df, vect') # для отладки   
        return matrix_df, vect
    else:
        print('matrix_df') # для отладки
        return matrix_df

# Выбор настроек векторизации
def vectSettings(column, df, ngram=1):
    vectProcess('c', column, df, 1, ngram=ngram)
    vectProcess('t', column, df, 1, ngram=ngram)

    print('--- Выше представлены наиболее и наименее характерные токены с т.з. абсолютных частот и TF-IDF')
    goC_0 = True
    while goC_0:
        print('--- Если хотите отсеять менее частотные токены, впишите предпочитаемую границу как дробное число -- минимальная доля встречаемости'
            , '\n--- Или как целое число -- минимальная частота (абсолютная или относительная) встречаемости; после этого нажмите Enter'
            , '\n--- Если НЕ хотите отсеять, просто нажмите Enter')

        goC_1 = True
        while goC_1:
            try:
                min_df = input()
                min_df = '1' if len(min_df) == 0 else min_df
                min_df = float(min_df) if '.' in min_df else int(min_df)
                goC_1 = False
            except ValueError:
                print('--- Вы вписали что-то не то'
                      , '\n--- Впишите, пожалуйста, дробное или целое число и нажмите Enter: ')

        print('--- В результате векторизации можно получить абсолютные или относительные частоты токенов в документах (TF-IDF)'
            , '\n--- Выше представлены наиболее и наименее характерные токены с т.з. абсолютных частот и TF-IDF'
            , '\n--- Опираясь на них, впишите предпочитаемый вариант векторизаци: "c" -- абсолютные частоты, "t" -- TF-IDF'
            , '\n--- После этого нажмите Enter')
        frequencyMetric = input()

        goC_2 = True
        while goC_2:
            if frequencyMetric == 'c':
                # print('Абсолютные частоты')
                goC_2 = False
            elif frequencyMetric == 't':
                # print('Относительные частоты TF-IDF')
                goC_2 = False
            else:
                print('--- Вы вписали что-то не то'
                      , '\n--- Впишите, пожалуйста, сторого следуя инструкции: "c" -- абсолютные частоты, "t" -- TF-IDF'
                      , '\n--- После этого нажмите Enter: ')
                frequencyMetric = input()

        matrix_df = vectProcess(frequencyMetric=frequencyMetric, column=column, df=df, min_df=min_df, ngram=ngram)
        print('--- Если результат Вас устраивает, нажмите Enter'
              , '\n--- Если НЕ устраивает и хотите заново вписать предпочитаемые границу и вариант векторизаци, нажмите пробел и затем Enter')
        if len(input()) == 0:
            goC_0 = False
    return frequencyMetric, column, df, min_df
