#!/usr/bin/env python
# coding: utf-8

'''
A module for preprocessing variables of nominal, ordinal, interval, and higher-level measurement
'''

# sys & subprocess -- эти пакеты должны быть предустанавлены. Если с ними какая-то проблема, то из этого скрипта решить их сложно
import sys
from subprocess import check_call

# --- остальные модули и пакеты
while True:
    try:
        import matplotlib.pyplot as plt, pandas
        break
    except ModuleNotFoundError:
        errorDescription = sys.exc_info()
        module = str(errorDescription[1]).replace("No module named '", '').replace("'", '') #.replace('_', '')
        if '.' in module: module = module.split('.')[1] 
        print('Пакет', module,
              'НЕ прединсталируется с установкой Анаконды, но для работы скрипта требуется этот пакет, поэтому он будет инсталирован сейчас\n')
        check_call([sys.executable, "-m", "pip", "install", module])

def varHist(dfIn, var):
    plt.figure()
    plt.hist(dfIn[var])
    plt.title(f'Values distribution of {var}')
    plt.xlabel(f'{var}')
    plt.xticks(dfIn[var].unique(), minor=True)
    plt.ylabel('Frequency')
    plt.show();    

def valuesDropper(dfIn, var):
    df = dfIn.copy()
    print('В этой функции переменная рассматривается как порядковая или более высокого типа шкалы')
    varHist(df, var)
    print('\n--- Если требуется исключить из дальнейшего рассмотрения какие-то значения переменной, введите в окно МИНИМАЛЬНОЕ из этих значений и нажмите Enter'
        , '\n--- В противном случае, просто нажмите Enter')
    valueForRemoving = input()
    if len(valueForRemoving) != 0:
        valueS_ForRemoving = [valueForRemoving]
        print('\n--- Теперь введите МАКСИМАЛЬНОЕ из этих значений и нажмите Enter'
            , '\n--- Если максимум равен минимуму, просто нажмите Enter')
        valueForRemoving = input()
        if len(valueForRemoving) != 0:
            valueS_ForRemoving.append(valueForRemoving)
            print('\n--- Если ВСЕ числа в обзначенном Вами диапазоне требуется исключить из дальнейшего рассмотрения, просто нажмите Enter'
                , '\n--- В противном случае введите конкретное число из этого диапазона и нажмите Enter')
            valueForRemoving = input()
            if len(valueForRemoving) != 0:
                while len(valueForRemoving) != 0:
                    valueS_ForRemoving.append(valueForRemoving)
                    print('\n--- Если требуется исключить из дальнейшего рассмотрения ЕЩЁ одно конкретное число из этого диапазона, введите его и нажмите Enter'
                        , '\n--- В противном случае просто нажмите Enter')
                    valueForRemoving = input()
                    valueS_ForRemoving.append(valueForRemoving)
                # Исключить из рассмотрения диапазон
                valueS_ForRemoving.remove('')
                for valueForRemoving in valueS_ForRemoving:
                    df = df[dataForML[var] != float(valueForRemoving)]
            else:
                # Исключить из рассмотрения диапазон
                valueS_ForRemoving.sort()
                df = df[(df[var] < float(valueS_ForRemoving[0])) | (df[var] > float(valueS_ForRemoving[-1]))]
    
        print('\nДля дальнейшего рассмотрения остались следующие значения игрека и их частоты:')
        varHist(df, var)
    return df

# # 1.1.3 Если требуется выбрать только интересующие значения игрека
# print('В этом чанке игрек рассматривается как номинальная переменная'
#     , '\n--- Если требуется объединить какие-то значения игрека в одно, впишите в окно эти значения без кавычек через запятую с пробелом и нажмите Enter'
#     , '\n--- В противном случае, просто нажмите Enter\n')
# dataForML[varY] = dataForML[varY].astype(str)
# valueS_ForJoining = input()
# if len(valueS_ForJoining) != 0:
#     while len(valueS_ForJoining) != 0:
#         valueS_ForJoining = valueS_ForJoining.split(', ')
#         labelForJoined = input('--- Введите в окно метку для обозначения этих значений и нажмите Enter')
#         for valueForJoining in valueS_ForJoining:
#             print('Поглащается значение', valueForJoining)
#             dataForML.loc[dataForML[varY] == valueForJoining, varY] = labelForJoined
#         print('--- Если требуется объединить ещё какие-то значения игрека в одно, введите в окно эти значения через запятую и пробел, после чего нажмите Enter'
#             , '\n--- В противном случае, просто нажмите Enter\n')
#         valueS_ForJoining = input()

#     print('\nДля дальнейшего рассмотрения остались следующие значения игрека и их частоты:')
#     display(dataForML[varY].value_counts())
#     display(dataForML[varY].value_counts(normalize=True))
#     dataForML[varY].value_counts().plot.bar()
