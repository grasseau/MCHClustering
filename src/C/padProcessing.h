# ifndef  _PADPROCESSING_H
# define  _PADPROCESSING_H

typedef struct {
    PadIdx_t i;
    PadIdx_t j;
} MapKToIJ_t;

 // 8 neigbours + the center pad itself + separator (-1)
static const int MaxNeighbors = 12;

inline static PadIdx_t *getNeighborsOf( PadIdx_t *neigh,  PadIdx_t i) { return &neigh[MaxNeighbors*i]; };
inline static PadIdx_t getTheFirstNeighborOf( PadIdx_t *neigh,  PadIdx_t i) { return neigh[MaxNeighbors*i]; };

extern "C" {
    
  // Helper to get results from Python
  int getNbrProjectedPads();
  //
  void copyProjectedPads(double *xyDxy, double *chA, double *chB);
  
  PadIdx_t *getFirstNeighbors( const double *xyDxy, int N, int allocatedN);
  
  int projectChargeOnOnePlane( 
          const double *xy0InfSup, const double *ch0, 
          const double *xy1InfSup, const double *ch1, 
          PadIdx_t N0, PadIdx_t N1, int includeAlonePads);
  
  int getConnectedComponentsOfProjPads( short *padGrp );
  
  int findLocalMaxWithLaplacian( const double *xyDxy, const double *z, int N, int xyDxyAllocated, double *laplacian,  double *theta);
  
  void assignCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup);
  //
  void copyCathToGrp( short *cath0Grp, short *cath1Grp, int nCath0, int nCath1);
  
  void getMaskCathToGrp( short g, short* mask0, short *mask1, int nCath0, int nCath1);
  
  void freeMemoryPadProcessing();
}
#endif // _PADPROCESSING_H

