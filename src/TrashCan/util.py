# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import pickle 

def writePickle( fileName, obj ):
   file = open( fileName, "wb" )
   pickle.dump( obj, file )
   file.close()
   
def readPickle( fileName ):
   file = open( fileName, "rb" )
   obj = pickle.load( file )
   file.close()
   return obj

if __name__ == "__main__":
    print("Hello World")
