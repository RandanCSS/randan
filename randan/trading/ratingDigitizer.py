import re
def ratingDigitizer(letters, raitingSource):
    if raitingSource == 'RB':
        letters = letters.replace(' ГМ', '').replace(' Неквалы', '').replace(' Остальные', '')
        if letters == 'ГМ':
            return 18
        letters = re.findall(r'[ABАВ]+.?', letters)[0]
        # print('letters :', letters) # для отладки
        subtracted = 1 if ('-' in letters) | ('+' in letters) else 0
        # print('subtracted :', subtracted) # для отладки

    if raitingSource == 'RA': # в записях не только рейтинги, но и вспомогательные символы; как образец смотреть директорию "boughtIssuerS Архив"
        subtracted = 3 if ('-' in letters) | ('+' in letters) else 2
        # print('subtracted:', subtracted) # для отладки

    x = 1 if ('A' in letters) | ('А' in letters) else 0
    # print('x :', x) # для отладки

    if '-' in letters:
        y = -2
    elif '+' in letters:
        y = 0
    else:
        y = -1
    # print('y :', y)
    # print('len(letters) :', len(letters)) # для отладки
    return (3 * (len(letters) - subtracted) + y) + 9 * x
