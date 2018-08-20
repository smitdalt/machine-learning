# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:10:37 2016

@author: jayurbain
"""

from pandas.io.data import DataReader
from datetime import datetime

goog = DataReader("GOOG",  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
goog["Adj Close"]
Date
2004-08-19    100.34
2004-08-20    108.31
2004-08-23    109.40
2004-08-24    104.87
2004-08-25    106.00