#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jlul 30, 2020 2:46:25 PM$"

import sys

import numpy as np

class CSV:
  def __init__(self, fileName, format):
    self.file = open(fileName, 'r')
    nbrOfFields = len( format)
    # self.fields = [0]*nbrOfFields
    self.fields = [[] for i in range(nbrOfFields)]
    # print(self.fields)
    for line in self.file:
      # print("line", line, "line[0]-", line[0], "-")
      if line[0] != '#':
        fields = line.split()
        for i, str in enumerate(fields):
          # print( i, ' ',  str)
          if format[i] == 'i':
            self.fields[i].append( int(str))
          elif format[i] == 'f':
            self.fields[i].append( float(str))
          else:
            self.fields[i].append( str )
    
    for i in range(nbrOfFields):
      self.fields[i] = np.asarray(self.fields[i])   
      # print(self.fields[i].shape)
      
    
        
          
