#! /usr/bin/env python
# coding: utf-8
from __future__ import print_function

import pickle

fr = open('wxpy.pkl', 'rb')
inf = pickle.load(fr)
print(inf.values())
# print(inf.items())
# print(inf.has_key(u'肖长省'))
# print(inf.has_key(u'trade-test'))
fr.close()