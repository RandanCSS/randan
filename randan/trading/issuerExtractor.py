def issuerExtractor(dfIn):
    df = dfIn.copy()
    df['Эмитент'] = df['SECNAME'].str.replace('_', ' ').str.replace('-', ' ')
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r' [БЗ][OОPР]П?.+', '', text))
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r'Б\d+', '', text))
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r' 0\d+.*', '', text))
    df['Эмитент'] = df['Эмитент'].apply(lambda text: re.sub(r' \d+ обл\.?', '', text))
    df['Эмитент'] = df['Эмитент'].apply(textPreprocessor.simbolsCleaner)
    df['Эмитент'] = df['Эмитент'].str.replace('ПАО ', '').str.replace(' ПАО', '')\
        .str.replace('АО ', '').str.replace(' АО', '')\
        .str.replace('ООО ', '').str.replace(' ООО', '')\
        .str.replace('"', '')
    df['Эмитент'] = df['Эмитент'].apply(lambda text: text if text[0] != ' ' else text[1:])
    df['Эмитент'] = df['Эмитент'].apply(lambda text: text if text[-1] != ' ' else text[:-1])
    df['Эмитент'] = df['Эмитент'].str.split(' Б ').str[0]
    df['Эмитент'] = df['Эмитент'].str.lower()
    df.loc[df['Эмитент'].str.contains('офз'), 'Эмитент'] = 'минфин рф'
    df.loc[(df['Эмитент'].str.contains('воз')) & (df['Эмитент'].str.contains('рф')), 'Эмитент'] = 'минфин рф'
    return df
