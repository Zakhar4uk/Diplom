#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys

value = sys.argv[1]

file_name = 'my_file.txt'
f = open(file_name, 'a+')  # open file in append mode
f.write(value)
f.close()