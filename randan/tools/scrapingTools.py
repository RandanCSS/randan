#!/usr/bin/env python
# coding: utf-8

'''
(EN) A proprietary module that simplifies and manages the web scraping workflow
(RU) Авторский модуль для упрощения скрапинга
'''

def argument_key_comparison(argument, key, params):
    if (key in params.keys()) & (argument != None):
        if params[key] != argument:
            print(f'!!   Вы подали {key} и как аргумент, и через словарь params , причём Вы подали разные значения туда и туда; будет использовано значение, поданное в params !!\n')
            argument = params[key]
    elif (key in params.keys()) & (argument == None): argument = params[key]
    elif (key not in params.keys()) & (argument != None): pass # отдельный аргумент определён, поэтому запрос к пользователю не поступит
    else: pass # НИ ключ params , НИ отдельный аргумент НЕ определены, поэтому запрос к пользователю поступит
    return argument
