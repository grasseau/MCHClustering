
# ifndef  _CLUSTERPROCESSING_H
# define  _CLUSTERPROCESSING_H

typedef short Saturated_t;
typedef std::pair< int, const double*> DataBlock_t;
typedef short Group_t;

extern "C" {
  void setMathiesonVarianceApprox( int chId, double *theta, int K );
  
  int clusterProcess( const double *xyDxyi, const Mask_t *cathi, const Saturated_t *saturated, const double *zi, 
                       int chId, int nPads);
  
  void collectTheta( double *theta, Group_t *thetaToGroup, int N);
  
  int getNbrOfPadsInGroups();
  
  void collectPadsAndCharges( double *xyDxy, double *z, Group_t *padToGroup, int nTot);
  
  void cleanClusterProcessVariables();
}
# endif
