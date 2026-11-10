#!/usr/bin/env python
# coding: utf-8

'''
(EN) A module that simplifies and maximizes VK content extraction using the platform's official API method wall.get
(RU) Модуль для упрощения выгрузки контента ВК методом его API wall.get и максимизации размера этой выгрузки
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
      from typing import Callable, Optional                     # только для аннотаций типов (можно убрать)
      import os
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

# Функции..
def list_by_freshness(
    folder,
    *,                                                          # дальше — только keyword-only аргументы
    filter_dirEntry: Optional[Callable[[os.DirEntry], bool]] = None,  # предикат по DirEntry (до обхода поддерева)
    follow_symlinks: bool = False,                              # прокидываем в newest_mtime_ns
):
    """Содержимое folder, отсортированное по «свежести»:
       max(st_mtime_ns) по каждым поддиректории и файлу и всем его потомкам, свежие сверху.

       filter_dirEntry применяется ДО обхода поддерева, так что отфильтрованные кандидаты не обходятся вовсе.

       Недоступные|ошибочные поддиректории и файлы не выкидываются, а получают ключ -1 (окажутся в конце).
       Если их нужно скрывать — уберите их в filter_dirEntry.
    """

    try:                                                         # scandir(folder) может упасть (нет folder / нет прав)
        with os.scandir(folder) as nameS_in_folder:              # итератор по непосредственному содержимому folder
            nameS_in_folder = list(nameS_in_folder)              # итерации происходят внутри list(...)

    except OSError:                                              # путь недоступен — содержимого нет
        return []                                                # пустой список вместо исключения: удобно для вызывающего

    if filter_dirEntry is not None: nameS_in_folder_filtered = [name_in_folder for name_in_folder in nameS_in_folder if filter_dirEntry(name_in_folder)]
    else: nameS_in_folder_filtered = nameS_in_folder

    keyed = []                                                   # будем хранить (ключ, имя, путь) для сортировки
    for name_in_folder_filtered in nameS_in_folder_filtered:     # по каждому оставшемуся ребёнку
        freshest = newest_mtime_ns(name_in_folder_filtered.path, follow_symlinks=follow_symlinks)  # «свежесть» поддиректории
        keyed.append((freshest if freshest is not None else -1, name_in_folder_filtered.name, name_in_folder_filtered.path))
            # ^ None (недоступно) → -1, чтобы такие поддиректории ушли в конец;
            #   name_in_folder_filtered.name кладём как вторичный ключ для детерминизма при равных mtime

    keyed.sort(key=lambda triple: (-triple[0], triple[1]))       # сортировка: по mtime по убыванию, по имени по возрастанию
        # ^ -triple[0] превращает «по убыванию» в «по возрастанию» для sort без reverse=True,
            # чтобы вторичный ключ triple[1] остался по возрастанию (reverse=True отсортировал бы оба по убыванию).

    return [triple[2] for triple in keyed]                            # разворачиваем тройки обратно в список путей

def newest_mtime_ns(folder, *, follow_symlinks=False):
    # ^ follow_symlinks=False — не разворачивать симлинки, заходить только в настоящие директории
    #  это ключевой аргумент; он намеренно keyword-only (после `*`), чтобы нельзя было передать его позиционно по ошибке — это влияет на безопасность обхода

    '''Максимальный st_mtime_ns в поддереве folder (включая сам folder). None — если folder недоступен'''

    try:                                                        # обёртка: stat может упасть
        freshest = os.stat(folder, follow_symlinks=follow_symlinks).st_mtime_ns
            # ^ «собственный» mtime стартового folder как начальный максимум; одновременно проверка доступности
            #   st_mtime_ns — int в наносекундах, точнее, чем float из getmtime()

    except OSError: return None                                 # недоступно из-за FileNotFoundError, PermissionError,..

    stack = [folder]                                            # итеративный DFS: список путей-директорий к обходу
        # ^ именно итеративный, а не рекурсивный, чтобы на глубоких деревьях не упереться в лимит рекурсии

    while stack:
        # ^ на первой итерации в списке только folder , которую .pop() сразу перекладывает в path_current ,
            # но если в folder есть поддиректории, .append() добавляет в stack пути к ним на этой и следующих итерациях

        path_current = stack.pop()                              # LIFO: берём последнюю добавленный директорию
        try:                                                    # scandir тоже может упасть (гонка, права)
            with os.scandir(path_current) as nameS_in_path_current:
                # ^ scandir даёт DirEntry с кешем stat — экономя сисколов

                for name_in_path_current in nameS_in_path_current:
                    # ^ по каждому имени текущей директории

                    try:                                        # отдельный try на каждое имя
                        name_in_path_current_status = name_in_path_current.stat(follow_symlinks=follow_symlinks) 
                            # кеш о поддиректории или файле из DirEntry , которые скрываются под итерируемым именем

                    except OSError: continue                    # битый inode, гонка «файл удалён», нет прав

                    if name_in_path_current_status.st_mtime_ns > freshest: freshest = name_in_path_current_status.st_mtime_ns
                        # ^ найден более свежий элемент

                    if name_in_path_current.is_dir(follow_symlinks=follow_symlinks): stack.append(name_in_path_current.path)
                        # ^ добавление директории в очередь на обход

        except OSError: continue                                # если scandir(path_current) упал целиком
    return freshest                                             # максимум по folder , включая всё её содержимое;
                                                                    # симлинки не разворачиваются, но их собственный mtime учитывается
