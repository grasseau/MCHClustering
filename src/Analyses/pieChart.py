#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

# Cluster Processing
# import C.PyCWrapper as PCWrap
import O2_Clustering.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.geometry as geom
import Util.dataTools as dUtil
# Reading MC, Reco, ... Data
import Util.IOv5 as IO
# Analyses Tool Kit
import analyseToolKit as aTK



if __name__ == "__main__":





  reco = np.array([ [0, 989, 969, 1117, 1008, 565, 557, 604, 613, 577, 595],
                   [0, 79, 50, 52, 61, 33, 12, 5, 10, 1, 3],
                   [0, 165, 158, 63, 41, 9, 3, 9, 10, 7, 8]
                  ])
  # Version 0
  # EM TP/FP/FN
  em = np.array([ [0, 1192, 1211, 1133, 1017, 565, 557, 605, 611, 576, 593],
                  [0, 91, 104, 48, 58, 29, 13, 6, 7, 3, 3],
                  [0, 191, 116, 44, 31, 11, 3, 9, 12, 8, 10]
               ])


  # Test to improve
  # noise = 0
  em = np.array([
     [0, 1220, 1235, 1141, 1014, 562, 557, 604, 609, 576, 596],
     [0, 193, 195, 142, 137, 45, 29, 25, 16, 14, 20],
     [0, 160, 85, 37, 23, 10, 3, 9, 12, 7, 7]
               ])
  # nLesss / (count) -> nLesss / (count-1)
  em = np.array([
    [0, 1180, 1199, 1128, 1015, 564, 557, 605, 609, 576, 593],
    [0, 76, 87, 53, 59, 29, 13, 7, 8, 2, 3],
    [0, 194, 118, 45, 31, 11, 3, 9, 13, 8, 10]
               ])
  em = np.array([
    [0, 1180, 1197, 1126, 1015, 564, 557, 605, 609, 576, 593],
    [0, 77, 87, 50, 58, 29, 13, 6, 7, 2, 3],
    [0, 194, 120, 45, 31, 11, 3, 9, 13, 8, 10]
               ])

  """ 
  noise = 0.01
  [0, 1160, 1175, 1077, 957, 557, 552, 596, 603, 572, 590]
  [0, 76, 87, 53, 58, 28, 13, 7, 9, 2, 3]
  [0, 197, 123, 46, 33, 11, 3, 9, 13, 8, 10]
  noise = - 0.01
  [0, 1220, 1236, 1140, 1014, 562, 555, 604, 609, 576, 596]
  [0, 196, 198, 143, 134, 45, 25, 29, 17, 15, 24]
  [0, 160, 84, 38, 23, 10, 3, 9, 12, 7, 7]
  chi-ratio = 0.9
  [0, 1157, 1184, 1126, 1013, 561, 557, 605, 608, 576, 593]
  [0, 65, 78, 48, 48, 24, 14, 6, 7, 3, 4]
  [0, 217, 132, 51, 34, 10, 3, 9, 13, 8, 10]
[0, 1129, 1170, 1122, 1008, 560, 557, 605, 608, 576, 593]
[0, 60, 72, 44, 48, 23, 15, 7, 6, 2, 6]
[0, 245, 146, 55, 38, 10, 3, 9, 13, 8, 10]
  """
  allEM = np.zeros(3)
  allReco = np.zeros(3)
  for k in range(3):
   allEM[k] += np.sum( em[k, :] )
   allReco[k] += np.sum( reco[k, :] )
  
  # Per station
  stEM = np.zeros( (3, 6) ) 
  stReco = np.zeros( (3, 6) )
  for k in range(3):
   for st in range(5):
     stEM[k, st+1] += em[k, 2*st +1] 
     stEM[k, st+1] += em[k, 2*st +2] 
     stReco[k, st+1] += reco[k, 2*st +1] 
     stReco[k, st+1] += reco[k, 2*st +2]      
   
  labels = 'TP', 'FP', 'FN'
  colors = ['palegreen', 'magenta', 'tomato']
  colors = ['palegreen', 'violet', 'coral']   
  fig, ax = plt.subplots(1,2)

  explode = (0, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
  ax[0].pie(allReco, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45, colors=colors)
  ax[1].pie(allEM, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45, colors=colors)
  # ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

  plt.show()
  
  fig, ax = plt.subplots(2,5)
  for st in range(5):  
    ax[0, st].pie(stReco[:,st+1], explode=explode, labels=labels, 
        # autopct='%1.1f%%',
        shadow=True, startangle=45, colors=colors)
    ax[1, st].pie(stEM[:,st+1], explode=explode, labels=labels, 
        # autopct='%1.1f%%',
        shadow=True, startangle=45, colors=colors)
  plt.show()