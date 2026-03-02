import re

txtFile = open("mezencev_da_svyaz-poziciy-francuzskih-grajdan-po-diskusionnym-voprosam-v-politicheskom-pole-strany-s-patteranmi-e_321619.txt", 'r')
txtContent = txtFile.read()
txtFile.close()
txtContent # содержит в т.ч. подстрочные ссылки (как отдельный блок в конце документа)

'Bankert ' in txtContent

# # !pip install Spire.Doc

# from spire.doc import *
# from spire.doc.common import *

# # Создайте объект документа
# doc = Document()

# # Загрузите файл Word
# doc.LoadFromFile("mezencev_da_svyaz-poziciy-francuzskih-grajdan-po-diskusionnym-voprosam-v-politicheskom-pole-strany-s-patteranmi-e_321619.docx")

# # Получите текст из всего документа
# docContent = doc.GetText()
# docContent # не теряется ли часть текста? В блокноте поиск по такому большому текту работает некорректно
# # НЕ содержит подстрочные ссылки

# 'Bankert ' in docContent

def extractReference(text):
    pattern = r'''(
        ((,\s*)?[A-ZА-ЯЁ][a-zó\?а-яё]+\s+[A-ZА-ЯЁ]\.(\s*+[A-ZА-ЯЁ]\.)?)+(\s+et\s+al\.)?.+
        # Фамилии [со]авторов и их инициалы, в т.ч. двойные, в т.ч. через пробел[ы] и "et al.", при необходимости
                   )'''
    referenceS = re.findall(pattern, text, re.VERBOSE | re.MULTILINE)
    referenceS_refined = []
    for listElement in referenceS:
        referenceS_refined.append(listElement[0].strip())
    return referenceS_refined

txtReferenceS = extractReference(txtContent)
print(txtReferenceS)
len(txtReferenceS)

txtReferenceS.sort()
references = '\n'.join(txtReferenceS).strip()
print(references)

docReferenceS = extractReference(docContent)
print(docReferenceS)
len(docReferenceS)

txtReferenceS_copy = txtReferenceS.copy()
txtReferenceS_copy

docReferenceS_copy = docReferenceS.copy()
docReferenceS_copy

for txtReference in txtReferenceS_copy:
    if txtReference in docReferenceS:
        txtReferenceS.remove(txtReference)
        docReferenceS.remove(txtReference)

# txtReferenceS

# docReferenceS

print('Скрипт исполнен. Сейчас появится надпись: "An exception has occurred, use %tb to see the full traceback.\nSystemExit" -- так и должно быть')
input()
sys.exit()

# Тестовый текст
text = """
Пример текста с библиографическими ссылками:
9.	Anderson M. Conservative politics in France. – Taylor & Francis, 2023.
10.	Ayll?n S., Valbuena J., Plum A. Youth unemployment and stigmatization over the business cycle in Europe //Oxford Bulletin of Economics and Statistics. – 2022. – Т. 84. – №. 1. – С. 103-129.
11.	Bankert A., Huddy L., Rosema M. Measuring partisanship as a social identity in multi-party systems //Political behavior. – 2017. – Т. 39. – С. 103-132.
"""

text

pattern = r'''(
    (?:((,\s*)?[A-ZА-ЯЁ][a-zа-яё]+\s+[A-ZА-ЯЁ]\.(\s*+[A-ZА-ЯЁ]\.)?)+(\s+et\s+al\.)?)* # Фамилии [со]авторов и их инициалы, в т.ч. двойные, в т.ч. через пробел[ы] и "et al.", при необходимости
    (?:\s*(\(\d{4}\))|(\d{4}\.))* # Год в скобках или с точкой
               )'''

pattern = r'''(
    ((,\s*)?[A-ZА-ЯЁ][a-zó\?а-яё]+\s+[A-ZА-ЯЁ]\.(\s*+[A-ZА-ЯЁ]\.)?)+(\s+et\s+al\.)?.+
    # Фамилии [со]авторов и их инициалы, в т.ч. двойные, в т.ч. через пробел[ы] и "et al.", при необходимости
               )'''
# Надо бы поработать с фамилиями, в которых возникают "?" вместо неизвестных символов 

re.findall(pattern, text, re.VERBOSE | re.MULTILINE)

# Кажется, что подстрочные ссылки трудны для многих бесплатных ИИ-помощников. + по таким ссылкам проще автоматизированная проверка неиспользованных в тексте пунктов библиографии.
# Также при прямом цитировании и при воспроизведении отдельных идей источника, не сводящегося к этой идее, следует требовать указывать страницы источника.
# Результаты корпоративной проверки на плагиат и ИИ следует сделать доступными не только руководителю, но и рецензенту.
