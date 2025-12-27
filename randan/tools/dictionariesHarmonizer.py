def dictionariesHarmonizer(df_Large, dfIn_Small, columnName):
    df_Small = dfIn_Small.copy() # df_Small -- датафрейм, редактируемый в столбце columnName на основе того же столбца датафрейма df_Large

    # Шаг № 1. Грубая сверка
    df_Small_Matching = df_Small[df_Small[columnName].isin(df_Large[columnName])]
    df_Small_New_1 = df_Small[df_Small[columnName].isin(df_Large[columnName]) != True]

    # Шаг № 2. Тонкая сверка
    rowS_toDrop = []
    df_Small_New_2 = df_Small_New_1.copy()
    elementS_Small = df_Small_New_1[columnName]
    for element_Small in elementS_Small:
        for element_Large in df_Large[columnName]:
            if element_Large in element_Small:
                df_Small_New_1.loc[df_Small_New_1[columnName] == element_Small, columnName] = element_Large # заменить element_Small на element_Large ,
                    # что обеспечивает совместимость обрабатываемых тут строчек df_Small_New_1 b df_Large
                df_Small_New_2 = df_Small_New_2[df_Small_New_2[columnName] != element_Small] 
    return df_Small_Matching, df_Small_New_1, df_Small_New_2
    # df_Small_New_1 -- часть редактируемого датафрейма (df_Small), которая не прошла грубую сверку, но прошла тонкую сверку
    # df_Small_New_2 -- часть редактируемого датафрейма (df_Small), которая не прошла ни грубую, ни тонкую сверку
