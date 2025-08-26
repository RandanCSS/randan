#!/usr/bin/env python
# coding: utf-8

'''
A module for creating a matrix `documents-tokens` from a corpus of documents using the methods of the CountVectorizer and TfidfVectorizer classes
'''

# sys & subprocess -- эти пакеты должны быть предустановлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
attempt = 0
while True:
    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        import pandas
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

# Процесс векторизации
def vectProcess(frequencyMetric, column, df, min_df, ngram=1):
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
              , '\n--- Если НЕ устраивает и хотите заново вписать предпочитаемые границу и вариант векторизаци, введите любой символ и нажмите Enter')
        if len(input()) == 0:
            goC_0 = False
    return frequencyMetric, column, df, min_df
