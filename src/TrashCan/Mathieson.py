#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jun 26, 2020 4:13:36 PM$"

import numpy as np


def mathieson1D( x ):
    n = x.shape[0]
    k3 = 0.71
    k2 = np.pi * 0.5 * ( 1.0 - np.sqrt( k3) *0.5 )
    sqrtk3 =  np.sqrt( k3 )
    k1 = k2 * sqrtk3 * 0.25 / (np.arctan( sqrtk3 ) )

    tanhk2sqr = np.tanh( k2 * x )
    tanhk2sqr = tanhk2sqr * tanhk2sqr

    y = k1 * (1.0  - tanhk2sqr ) / (1.0 + k3*tanhk2sqr)
    return y

class Mathieson:
  # Chamber 1, 2
  sqrtK3x1_2 = 0.7000 # Pitch= 0.21 cm
  sqrtK3y1_2 = 0.7550 # Pitch= 0.21 cm
  pitch1_2 = 0.21
  # Chamber 3, 10
  sqrtK3x3_10 = 0.7131 # Pitch= 0.25 cm
  sqrtK3y3_10= 0.7642 # Pitch= 0.25 cm
  pitch3_10 = 0.25
  
  def __init__(self, mType, chWeight):
    """
    mType [0 or 1] : Mathieson with 2 sets of cofficients 
    """
    # K3
    """
    self.K3x = np.array([ Mathieson.K3x1, Mathieson.K3x2 ])
    self.K3y = np.array([ Mathieson.K3y1, Mathieson.K3y2 ])
    """
    self.sqrtK3x = np.array([ Mathieson.sqrtK3x1_2, Mathieson.sqrtK3x3_10 ])
    self.sqrtK3y = np.array([ Mathieson.sqrtK3y1_2, Mathieson.sqrtK3y3_10 ])
    # self.sqrtK3x = np.sqrt( self.sqrtK3x )
    # self.sqrtK3y = np.sqrt( self.sqrtK3y )
    # K2
    self.K2x = np.pi * 0.5 * ( 1.0 - self.sqrtK3x * 0.5 )
    self.K2y = np.pi * 0.5 * ( 1.0 - self.sqrtK3y * 0.5 )
    # K1
    self.K1x = self.K2x * self.sqrtK3x * 0.25 / ( np.arctan( self.sqrtK3x ) )      
    self.K1y = self.K2y * self.sqrtK3y * 0.25 / ( np.arctan( self.sqrtK3y ) )
    # K4
    self.K4x = self.K1x / self.K2x / self.sqrtK3x
    self.K4y = self.K1y / self.K2y / self.sqrtK3y
    #
    self.curK1x = self.K1x[mType]
    self.curK1y = self.K1y[mType]
    
    self.curK4x = self.K4x[mType]
    self.curK4y = self.K4y[mType]
    self.curK2x = self.K2x[mType]
    self.curK2y = self.K2y[mType]

    self.curSqrtK3x = self.sqrtK3x[mType]
    self.curSqrtK3y = self.sqrtK3y[mType]
    self.curK3x = self.curSqrtK3x * self.curSqrtK3x
    self.curK3y = self.curSqrtK3y * self.curSqrtK3y
    self.Pitch = np.array([ Mathieson.pitch1_2, Mathieson.pitch3_10 ])
    self.curPitch = self.Pitch[mType]
    self.curInvPitch = 1.0 / self.curPitch
    self.integralWeight = chWeight
    return

  def computeMathieson1DIntegral( self, xInf, xSup):
    # Coef type defined with mType in construxtor
    # x/u
    uInf = self.curSqrtK3x * np.tanh( self.curK2x * self.curInvPitch * xInf )
    uSup = self.curSqrtK3x * np.tanh( self.curK2x * self.curInvPitch * xSup )
    
    I = 2.0 * self.curK4x * ( np.arctan( uSup ) - np.arctan( uInf ) ) 
    return I

  def mathieson( self, x, axe=0):
    # Coef type defined with mType in construxtor
    # x/u
    if axe == 0:
      # X axe
      u = np.tanh( self.curK2x * self.curInvPitch * x )
      num = 1.0 - u*u
      den = 1. + self.curK3x * u*u
      mathieson = self.curK1x * num / den
 
    else :
      # Y axe
      u = np.tanh( self.curK2y * self.curInvPitch * x )
      num = 1.0 - u*u
      den = 1. + self.curK3y * u*u
      mathieson = self.curK1y * num / den 
    return mathieson

  def computeAtanTanh( self, x, axe=0):
    # Coef type defined with mType in construxtor
    # x/u
    if axe == 0:
      # X axe
      # uSup = self.curSqrtK3x * np.tanh( self.curK2x * self.curInvPitch * xSup )
      u = self.curSqrtK3x * np.tanh( self.curK2x * self.curInvPitch * x )
      prim = 2.0 * self.curK4x * (np.arctan( u ))
    else :
      # Y axe
      u = self.curSqrtK3y * np.tanh( self.curK2y * self.curInvPitch * x )
      prim = 2.0 * self.curK4y * (np.arctan( u ))
    # I = 2.0 * self.curK4x * ( np.arctan( uSup ) - np.arctan( uInf ) ) 
    return prim

  def primitive( self, x, axe=0):
    return self.computeAtanTanh(x, axe)

  def computeMathieson2DIntegral( self, xInf, xSup, yInf, ySup ):
    # Coef type defined with mType in construxtor
    # x/u
    uInf = self.curSqrtK3x * np.tanh( self.curK2x * self.curInvPitch * xInf )
    uSup = self.curSqrtK3x * np.tanh( self.curK2x * self.curInvPitch * xSup )
    # y/v
    vInf = self.curSqrtK3y * np.tanh( self.curK2y * self.curInvPitch * yInf )
    vSup = self.curSqrtK3y * np.tanh( self.curK2y * self.curInvPitch * ySup )
    
    I = 4.0 * self.curK4x * ( np.arctan( uSup ) - np.arctan( uInf ) ) * self.curK4y * ( np.arctan( vSup) - np.arctan( vInf ) )
    return I
# end of Class


if __name__ == "__main__":
    print("Hello World");
