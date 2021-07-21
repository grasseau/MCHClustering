
# ifndef  _CLUSTERPROCESSING_H
# define  _CLUSTERPROCESSING_H

typedef std::pair< int, const double*> DataBlock_t;

extern "C" {
  void setMathiesonVarianceApprox( int chId, double *theta, int K );
  
  int clusterProcess( const double *xyDxyi, const Mask_t *cathi, const Mask_t *saturated, const double *zi, 
                       int chId, int nPads);
  
  void collectTheta( double *theta, Group_t *thetaToGroup, int N);
  
  int getNbrOfPadsInGroups();

  int getNbrOfProjPads();
  
  void collectPadsAndCharges( double *xyDxy, double *z, Group_t *padToGroup, int nTot);
  
  void collectPadToCathGroup( Mask_t *padToMGrp, int nPads );
  
  void collectLaplacian( double *laplacian, int N);
  
  void computeResidual( const double *xyDxy, const double *zObs, const double *theta, int K, int N, double *residual);
  
  void computeMathiesonResidual( const double *xyDxy, const Mask_t *cath, const double *zObs, const double *theta, int chId, int K, int N, double *residual);

  void compute2DMathiesonMixturePadIntegrals( const double *xyInfSup0, const double *theta,
                                              int N, int K, int chamberId, double Integrals[] );
  void collectResidual( double *residual, int N);
  
  int getKThetaInit();
  
  void collectThetaInit( double *thetai, int N);
  
  int getNbrOfThetaEMFinal();

  void collectThetaEMFinal( double *thetaEM, int K);
  
  void cleanClusterProcessVariables( int uniqueCath);
}
# endif
