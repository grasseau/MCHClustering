// TODO:
// - throw exception on error see other mechanisms
#include <algorithm>
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <stdexcept>
#
# include "dataStructure.h"
# include "mathUtil.h"
# include "padProcessing.h"

# define CHECK 1
# define VERBOSE 0

// Intersection matrix
static PadIdx_t* IInterJ=0;
static PadIdx_t* JInterI=0;
static PadIdx_t* intersectionMatrix=0;

// Pad with no other cathode
static PadIdx_t*aloneIPads=0;
static PadIdx_t*aloneJPads=0;
static PadIdx_t*aloneKPads=0;

// Maps
static MapKToIJ_t *mapKToIJ=0;
static PadIdx_t   *mapIJToK=0;


// Neighbors
static PadIdx_t *neighbors=0;
static PadIdx_t *neighborsCath0=0;
static PadIdx_t *neighborsCath1=0;
static PadIdx_t *grpNeighborsCath0=0;
static PadIdx_t *grpNeighborsCath1=0;

// Projected Pads
static int maxNbrOfProjPads=0;
static int nbrOfProjPads=0;
static double* projected_xyDxy=0;
static double* projX;
static double* projDX;
static double* projY;
static double* projDY;
// Charge on the projected pads
static double* projCh0=0;
static double* projCh1=0;
static double* minProj=0;
static double* maxProj=0;

// cathodes group
static short* cath0ToGrpFromProj=0;
static short* cath1ToGrpFromProj=0;
//
static short* cath0ToTGrp=0;
static short* cath1ToTGrp=0;
//
int getNbrProjectedPads() { return nbrOfProjPads; };

void setNbrProjectedPads( int n) { nbrOfProjPads = n; maxNbrOfProjPads= n; };

void storeProjectedPads ( const double *xyDxyProj, const double *z, int nPads) {
  projected_xyDxy = new double[nPads*4];
  projCh0 = new double[nPads];
  projCh1 = new double[nPads];

  vectorCopy( xyDxyProj, 4*nPads, projected_xyDxy);
  vectorCopy( z, nPads, projCh0 );
  vectorCopy( z, nPads, projCh1 );
}

void copyProjectedPads(double *xyDxy, double *chA, double *chB) {

  for ( int i=0; i < 4; i++) {  
    for( int k=0; k < nbrOfProjPads; k++) xyDxy[i*nbrOfProjPads + k] = projected_xyDxy[i*maxNbrOfProjPads + k];
  }
  for( int k=0; k < nbrOfProjPads; k++) chA[k] = projCh0[k];
  for( int k=0; k < nbrOfProjPads; k++) chB[k] = projCh1[k];
  
}

void collectProjectedMinMax(double *chMin, double *chMax) {

  for( int k=0; k < nbrOfProjPads; k++) chMin[k] = minProj[k];
  for( int k=0; k < nbrOfProjPads; k++) chMax[k] = maxProj[k];
}

void printMatrixInt( const char *str, const int *matrix, int N, int M) {
  printf("%s\n", str);
  for ( int i=0; i < N; i++) {
    for ( int j=0; j < M; j++) {
      printf(" %2d", matrix[i*M+j]);
    }
    printf("\n");
  }
  printf("\n");
}    

void printMatrixShort( const char *str, const short *matrix, int N, int M) {
  printf("%s\n", str);
  for ( int i=0; i < N; i++) {
    for ( int j=0; j < M; j++) {
      printf(" %2d", matrix[i*M+j]);
    }
    printf("\n");
  }
  printf("\n");
}  

void printInterMap( const char *str, const PadIdx_t *inter, int N) {
  const PadIdx_t *ij_ptr = inter;
  printf("%s\n", str);
  for( PadIdx_t i=0; i < N; i++) {
    printf("row/col %d:", i);  
    for( int k = 0; *ij_ptr != -1; k++, ij_ptr++) {
      printf(" %2d", *ij_ptr);
    }
    // skip -1, row/col change
    ij_ptr++;
    printf("\n");
  }
  printf("\n");
}

void printNeighbors( PadIdx_t *neigh, int N ) {
  printf("Neighbors %d\n", N);
  for ( int i=0; i < N; i++) {
    printf("  neigh of i=%2d: ", i);
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      printf("%d, ", j);
    }
    printf("\n");
  }
}

int checkConsistencyMapKToIJ( const PadIdx_t *aloneIPads, const PadIdx_t *aloneJPads, int N0, int N1) {
  MapKToIJ_t ij;
  int n=0;
  int rc=0;
  // Consistency with intersectionMatrix
  // and aloneI/JPads
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];    
    if ((ij.i >= 0) && (ij.j >= 0)) { 
      if ( intersectionMatrix[ij.i*N1 + ij.j] != 1) {
        printf("ERROR: no intersection %d %d %d\n", ij.i , ij.j, intersectionMatrix[ij.i*N1 + ij.j]);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      } else {
          n++;
      }
    } else if (ij.i < 0) {
      if ( aloneJPads[ij.j] != k) { 
        printf("ERROR: j-pad should be alone %d %d %d %d\n", ij.i , ij.j, aloneIPads[ij.j], k);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      }
    } else if (ij.j < 0) {
        if ( aloneIPads[ij.i] != k) {
          printf("ERROR: i-pad should be alone %d %d %d %d\n", ij.i , ij.j, aloneJPads[ij.j], k);      
          throw std::overflow_error("Divide by zero exception");
          rc = -1;
        }
    }
  }
  // TODO : Make a test with alone pads ??? 
  int sum = vectorSumInt( intersectionMatrix, N0*N1 );
  if( sum != n ) {
    printf("ERROR: nbr of intersection differs %d %d \n", n, sum);      
    throw std::overflow_error("Divide by zero exception");
    rc = -1;
  }
  
  for ( int i=0; i<N0; i++ ) {
    for ( int j=0; j<N1; j++ ) {
      int k = mapIJToK[i*N1+ j];
      if (k >= 0) {
        ij = mapKToIJ[k];
        if ( ( ij.i != i) || ( ij.j != j) ) throw std::overflow_error("checkConsistencyMapKToIJ: MapIJToK/MapKToIJ");
      } 
    }      
  }
  // Check mapKToIJ / mapIJToK 
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];
    if ( ij.i < 0) {
      if (aloneJPads[ij.j] != k ) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneJPads");
      }
    } else if( ij.j < 0 ) {
      if (aloneIPads[ij.i] != k ) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneIPads");
      }
    } else if( mapIJToK[ij.i*N1+ ij.j] != k ) {
      printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
      throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK");
    }
  }
  
  return rc;
}

int getIndexByRow( PadIdx_t *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *IIdx) {
  int k = 0;
  // printf("N=%d, M=%d\n", N, M);
  for( PadIdx_t i=0; i < N; i++) {
    for( PadIdx_t j=0; j < M; j++) {    
      if (matrix[i*M+j] == 1) {
        IIdx[k]= j; k++;
        // printf("k=%d,", k);
      }
    }
    // end of row/columns
    IIdx[k]= -1; k++;
  }
  // printf("\n final k=%d \n", k);
  return k;
}

int getIndexByColumns( PadIdx_t *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *JIdx) {
  int k = 0;
  for( PadIdx_t j=0; j < M; j++) {    
    for( PadIdx_t i=0; i < N; i++) {
      if (matrix[i*M+j] == 1) {
        JIdx[k]= i; k++;
      }
    }
    // end of row/columns
    JIdx[k]= -1; k++;
  }
  return k;
}

// Build the neighbor list
PadIdx_t *getFirstNeighbors( const double *xyDxy, int N, int allocatedN ) {
  const double eps = 1.0e-5;
  const double relEps = (1.0 + 1.0e-7);
  const double *X = getConstX( xyDxy, allocatedN);
  const double *Y = getConstY( xyDxy, allocatedN);
  const double *DX = getConstDX( xyDxy, allocatedN);
  const double *DY = getConstDY( xyDxy, allocatedN); 
  // vectorPrint("X ??? ", xyDxy, allocatedN);
  // 8 neigbours + the center pad itself + separator (-1)
  PadIdx_t *neighbors_ = new PadIdx_t[MaxNeighbors*N];
  for( PadIdx_t i=0; i<N; i++) {
    PadIdx_t *i_neigh = getNeighborsOf(neighbors_, i);
    // Search neighbors of i
    for( PadIdx_t j=0; j<N; j++) {
      /*
      int xMask0 = ( fabs( X[i] - X[j]) < relEps * (DX[i] + DX[j]) );
      int yMask0 = ( fabs( Y[i] - Y[j]) < relEps * (DY[i]) );
      int xMask1 = ( fabs( X[i] - X[j]) < relEps * (DX[i]) );
      int yMask1 = ( fabs( Y[i] - Y[j]) < relEps * (DY[i] + DY[j]) );
      */
      int xMask0 = ( fabs( X[i] - X[j]) < (DX[i] + DX[j]) + eps);
      int yMask0 = ( fabs( Y[i] - Y[j]) < (DY[i] + eps) );
      int xMask1 = ( fabs( X[i] - X[j]) < (DX[i] + eps) );
      int yMask1 = ( fabs( Y[i] - Y[j]) < (DY[i] + DY[j] + eps) );
      if ( (xMask0 && yMask0) || (xMask1 && yMask1) ) {
        *i_neigh = j;
        i_neigh++;
        // Check
        // printf( "pad %d neighbor %d xMask=%d yMask=%d\n", i, j, (xMask0 && yMask0), (xMask1 && yMask1));
      }
    }
    *i_neigh = -1;
    if (  CHECK && (fabs( i_neigh - getNeighborsOf(neighbors_, i) ) > MaxNeighbors) ) {
      printf("Pad %d : nbr of neighbours %ld greater than the limit %d \n", 
              i, i_neigh - getNeighborsOf(neighbors_, i), MaxNeighbors );
      throw std::overflow_error("Not enough allocation");
    }
  }
  if (VERBOSE) printNeighbors( neighbors_, N);
  return neighbors_;
}

/// Used ???
void computeAndStoreFirstNeighbors( const double *xyDxy, int N, int allocatedN) {  
  printf("projectedcath ?????????????????? neighborsCath1 %p\n", neighborsCath1);

  neighbors = getFirstNeighbors( xyDxy, N, allocatedN);
}

int buildProjectedPads( 
            const double *xy0InfSup, const double *xy1InfSup, 
            PadIdx_t N0, PadIdx_t N1, 
            PadIdx_t *aloneIPads, PadIdx_t *aloneJPads, PadIdx_t *aloneKPads, int includeAlonePads) {
  // Use positive values of the intersectionMatrix
  // negative ones are isolated pads
  // Compute the new location of the projected pads (projected_xyDxy)
  // and the mapping mapKToIJ which maps k (projected pads) 
  // to i, j (cathode pads)
  const double *x0Inf = getConstXInf( xy0InfSup, N0);
  const double *y0Inf = getConstYInf( xy0InfSup, N0);
  const double *x0Sup = getConstXSup( xy0InfSup, N0);
  const double *y0Sup = getConstYSup( xy0InfSup, N0);
  const double *x1Inf = getConstXInf( xy1InfSup, N1);
  const double *y1Inf = getConstYInf( xy1InfSup, N1);
  const double *x1Sup = getConstXSup( xy1InfSup, N1);
  const double *y1Sup = getConstYSup( xy1InfSup, N1);
  
  double l, r, b, t;
  int k=0; PadIdx_t *ij_ptr=IInterJ;
  double countIInterJ, countJInterI;
  PadIdx_t i, j;
  for( i=0; i < N0; i++) {
    // Nbr of j-pads intersepting  i-pad
    for( countIInterJ = 0; *ij_ptr != -1; countIInterJ++, ij_ptr++) {
      j = *ij_ptr;
      // Debug
      // printf("X[0/1]inf/sup %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j,  x0Inf[i], x0Sup[i], x1Inf[j], x1Sup[j]);
      l = fmax( x0Inf[i], x1Inf[j]);
      r = fmin( x0Sup[i], x1Sup[j]);
      b = fmax( y0Inf[i], y1Inf[j]);     
      t = fmin( y0Sup[i], y1Sup[j]);
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      mapKToIJ[k].i = i;
      mapKToIJ[k].j = j;          
      mapIJToK[i*N1+ j] = k;
      // Debug
      // printf("newpad %d %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j, k, projX[k], projY[k], projDX[k], projDY[k]);
      k++;
    }
    // Test if there is no intercepting pads with i-pad
    if( (countIInterJ == 0) && includeAlonePads ) {
      l = x0Inf[i];
      r = x0Sup[i];
      b = y0Inf[i];      
      t = y0Sup[i];
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      // printf("newpad alone cath0 %d %d %9.3g %9.3g %9.3g %9.3g\n", i, k, projX[k], projY[k], projDX[k], projDY[k]);
      // Not used ??? 
      mapKToIJ[k].i =i ; mapKToIJ[k].j=-1;
      aloneIPads[i] = k;
      aloneKPads[k] = i;
      k++;
    }
    // Row change
    ij_ptr++;
  }
  // Just add alone j-pads of cathode 1
  if ( includeAlonePads ) {
    ij_ptr = JInterI;
    for( PadIdx_t j=0; j < N1; j++) {
      for( countJInterI = 0; *ij_ptr != -1; countJInterI++, ij_ptr++);
      if( countJInterI == 0) {
        l = x1Inf[j];
        r = x1Sup[j];
        b = y1Inf[j];      
        t = y1Sup[j];
        projX[k] = (l+r)*0.5;
        projY[k] = (b+t)*0.5;
        projDX[k] = (r-l)*0.5;
        projDY[k] = (t-b)*0.5;
        // Debug
        // printf("newpad alone cath1 %d %d %9.3g %9.3g %9.3g %9.3g\n", j, k, projX[k], projY[k], projDX[k], projDY[k]);
        // newCh0[k] = ch0[i];
        // Not used ??? 
        mapKToIJ[k].i=-1; mapKToIJ[k].j=j;
        aloneJPads[j] = k;
        aloneKPads[k] = j;
        k++;
      }
      // Skip -1, row/ col change
      ij_ptr++;
    }
  }
  if (VERBOSE) {
    printf("builProjectPads mapIJToK=%p, N0=%d N1=%d\\n", mapIJToK, N0, N1);
    for (int i=0; i <N0; i++) {
      for (int j=0; j <N1; j++) {
        if ( (mapIJToK[i*N1+j] != -1))
          printf(" %d inter %d\n", i, j);
      }
    }  
  }
  vectorPrintInt("builProjectPads", aloneKPads, k);
  return k;
}

void buildProjectedSaturatedPads( const Mask_t *saturated0, const Mask_t *saturated1, Mask_t *saturatedProj) {
  for( int k=0; k < nbrOfProjPads; k++) {
    MapKToIJ_t ij = mapKToIJ[k];
    saturatedProj[k] = saturated0[ij.i] || saturated1[ij.j];
  }
}

// ??? must be split: projected grid AND charge projection
int projectChargeOnOnePlane( 
        const double *xy0InfSup, const double *ch0, 
        const double *xy1InfSup, const double *ch1, 
        PadIdx_t N0, PadIdx_t N1, int includeAlonePads) {
  // xy0InfSup, ch0, N0 : cathode 0
  // xy1InfSup, ch1, N1 : cathode 1
  // i: describes the cathode-0 pads
  // j: describes the cathode-1 pads
  // double maxFloat = DBL_MAX;
  double epsilon = 1.0e-4;

  const double *x0Inf = getConstXInf( xy0InfSup, N0);
  const double *y0Inf = getConstYInf( xy0InfSup, N0);
  const double *x0Sup = getConstXSup( xy0InfSup, N0);
  const double *y0Sup = getConstYSup( xy0InfSup, N0);
  const double *x1Inf = getConstXInf( xy1InfSup, N1);
  const double *y1Inf = getConstYInf( xy1InfSup, N1);
  const double *x1Sup = getConstXSup( xy1InfSup, N1);
  const double *y1Sup = getConstYSup( xy1InfSup, N1);
  // alocate in heap    
  intersectionMatrix = new PadIdx_t[N0*N1];
  // mapIJToK = (PadIdx_t *) malloc( N0*N1 * sizeof(PadIdx_t))
  mapIJToK = new PadIdx_t[N0*N1];
  aloneIPads = new PadIdx_t[N0];
  aloneJPads = new PadIdx_t[N1];
  aloneKPads = new PadIdx_t[N0*N1];
  
  vectorSetZeroInt( intersectionMatrix, N0*N1);
  vectorSetInt( mapIJToK, -1, N0*N1);
  vectorSetInt( aloneIPads, -1, N0);
  vectorSetInt( aloneJPads, -1, N1);
  vectorSetInt( aloneKPads, -1, N0*N1);
  //
  // Looking for j pads, intercepting pad i
  // Build the intersection matrix
  //
  double xmin, xmax, ymin, ymax;
  PadIdx_t xInter, yInter;
  for( PadIdx_t i=0; i < N0; i++) {
    for( PadIdx_t j=0; j < N1; j++) {
      xmin = fmax( x0Inf[i], x1Inf[j] );
      xmax = fmin( x0Sup[i], x1Sup[j] );
      xInter = ( xmin <= (xmax - epsilon) );
      ymin = fmax( y0Inf[i], y1Inf[j] );
      ymax = fmin( y0Sup[i], y1Sup[j] );
      yInter = ( ymin <= (ymax - epsilon));
      intersectionMatrix[i*N1+j] =  (xInter & yInter);
    }
  }
  //  
  if (VERBOSE) printMatrixInt( "  Intersection Matrix", intersectionMatrix, N0, N1);
  //
  // Compute the max number of projected pads to make 
  // memory allocations 
  //
  maxNbrOfProjPads = vectorSumInt( intersectionMatrix, N0*N1 );
  int nbrOfAlonePads = 0;
  if (includeAlonePads) {
    // Add alone cath0-pads 
    for( PadIdx_t i=0; i < N0; i++) {
      if ( vectorSumRowInt( &intersectionMatrix[i*N1], N0, N1 ) == 0) nbrOfAlonePads++; 
    }
    // Add alone cath1-pads 
    for( PadIdx_t j=0; j < N1; j++) {
      if ( vectorSumColumnInt( &intersectionMatrix[j], N0, N1) == 0) nbrOfAlonePads++; 
    }  
  }
  // Add alone pas and row/column separators
  maxNbrOfProjPads += nbrOfAlonePads + fmax( N0, N1);
  if (VERBOSE) printf("  maxNbrOfProjPads %d\n", maxNbrOfProjPads);
  //
  //
  // Projected pad allocation
  // The limit maxNbrOfProjPads is alocated
  //
  projected_xyDxy = new double [4*maxNbrOfProjPads];
  projX  = getX ( projected_xyDxy, maxNbrOfProjPads);
  projY  = getY ( projected_xyDxy, maxNbrOfProjPads);
  projDX = getDX( projected_xyDxy, maxNbrOfProjPads);
  projDY = getDY( projected_xyDxy, maxNbrOfProjPads);
  //
  // Intersection Matrix Sparse representation
  //
  IInterJ = new PadIdx_t[maxNbrOfProjPads]; 
  JInterI = new PadIdx_t[maxNbrOfProjPads]; 
  int checkr = getIndexByRow( intersectionMatrix, N0, N1, IInterJ);

  int checkc = getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
  if (CHECK) { 
    if (( checkr > maxNbrOfProjPads) || (checkc > maxNbrOfProjPads)) {
      printf("Allocation pb for  IInterJ or JInterI: allocated=%d, needed for row=%d, for col=%d \n", 
          maxNbrOfProjPads, checkr, checkc);
      throw std::overflow_error(
         "Allocation pb for  IInterJ or JInterI" );
    }
  }
  if (VERBOSE) {
    printInterMap("  IInterJ", IInterJ, N0 );
    printInterMap("  JInterI", JInterI, N1 );
  }
  mapKToIJ = new MapKToIJ_t[maxNbrOfProjPads];

  //
  // Build the new pads
  //
  nbrOfProjPads = buildProjectedPads( xy0InfSup, xy1InfSup, N0, N1, 
                                      aloneIPads, aloneJPads, aloneKPads, includeAlonePads );
  
  if (CHECK == 1) checkConsistencyMapKToIJ( aloneIPads, aloneJPads, N0, N1);
  
  //
  // Get the isolated new pads
  // (they have no neighborhing)
  //
  int thereAreIsolatedPads = 0;
  neighbors = getFirstNeighbors( projected_xyDxy, nbrOfProjPads, maxNbrOfProjPads ); 
  // printNeighbors();
  MapKToIJ_t ij;
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    if (getTheFirstNeighborOf(neighbors, k) == -1) {
      // pad k is isolated     
      thereAreIsolatedPads = 1;
      ij = mapKToIJ[k];
      if( (ij.i >= 0) && (ij.j >= 0)) { 
        if (VERBOSE) printf(" Isolated pad: nul intersection i,j = %d %d\n", ij.i, ij.j);
        intersectionMatrix[ij.i*N1 + ij.j] = 0; 
      } else {
        throw std::overflow_error("I/j negative (alone pad)");
      }
    }
  }
  if (VERBOSE && thereAreIsolatedPads) printf("There are isolated pads %d\n", thereAreIsolatedPads);
  //
  if( thereAreIsolatedPads == 1) {
    // Recompute all
    getIndexByRow( intersectionMatrix, N0, N1, IInterJ);
    getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
    //
    // Build the new pads
    //
    nbrOfProjPads = buildProjectedPads( xy0InfSup, xy1InfSup, N0, N1, 
                                      aloneIPads, aloneJPads, aloneKPads, includeAlonePads );
    getFirstNeighbors( projected_xyDxy, nbrOfProjPads, maxNbrOfProjPads );
  }

  //
  // Computing charges of the projected pads
  // Ch0 part
  //
  minProj = new double[nbrOfProjPads];
  maxProj = new double[nbrOfProjPads];
  projCh0 = new double[nbrOfProjPads];
  projCh1 = new double[nbrOfProjPads];
  PadIdx_t k=0;
  double sumCh1ByRow;
  PadIdx_t *ij_ptr = IInterJ;
  PadIdx_t *rowStart;
  for( PadIdx_t i=0; i < N0; i++) {
    // Save the starting index of the begining of the row
    rowStart = ij_ptr;
    // sum of charge with intercepting j-pad  
    for( sumCh1ByRow = 0.0; *ij_ptr != -1; ij_ptr++) sumCh1ByRow += ch1[ *ij_ptr ];
    double ch0_i = ch0[i];
    if ( sumCh1ByRow != 0.0 ) { 
      double cst = ch0[i] / sumCh1ByRow;
      for( ij_ptr = rowStart; *ij_ptr != -1; ij_ptr++) {
        projCh0[k] = ch1[ *ij_ptr ] * cst;
        minProj[k] = fmin( ch1[ *ij_ptr ], ch0_i );
        maxProj[k] = fmax( ch1[ *ij_ptr ], ch0_i );
        // Debug
        // printf(" i=%d, j=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", i, *ij_ptr, k, sumCh1ByRow, projCh0[k]);
        k++;
      }
    } else if (includeAlonePads){
      // Alone i-pad
      projCh0[k] = ch0[i];
      minProj[k] = ch0[i];
      maxProj[k] = ch0[i];      
      k++;      
    }
    // Move on a new row
    ij_ptr++;
  }
  // Just add alone pads of cathode 1
  if ( includeAlonePads ) {
    for( PadIdx_t j=0; j < N1; j++) {
      k = aloneJPads[j];
      if ( k >= 0 ){
        projCh0[k] = ch1[j];
        minProj[k] = ch1[j];
        maxProj[k] = ch1[j]; 
      }
    }
  }
  //
  // Computing charges of the projected pads
  // Ch1 part
  //
  k=0; 
  double sumCh0ByCol;
  ij_ptr = JInterI;
  PadIdx_t *colStart;
  for( PadIdx_t j=0; j < N1; j++) {
    // Save the starting index of the beginnig of the column
    colStart = ij_ptr;
    // sum of charge intercepting i-pad  
    for( sumCh0ByCol = 0.0; *ij_ptr != -1; ij_ptr++) sumCh0ByCol += ch0[ *ij_ptr ];
    if ( sumCh0ByCol != 0.0 ) { 
      double cst = ch1[j] / sumCh0ByCol;
      for( ij_ptr = colStart; *ij_ptr != -1; ij_ptr++) {
        PadIdx_t i = *ij_ptr;
        k = mapIJToK[i * N1 + j];
        projCh1[k] = ch0[i] * cst;
        // Debug
        // printf(" j=%d, i=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", j, i, k, sumCh0ByCol, projCh1[k]);
      }
    } else if (includeAlonePads){
      // Alone j-pad
      k = aloneJPads[j];
      if ( CHECK && (k < 0) ) printf("ERROR: Alone j-pad with negative index j=%d\n", j);
      // printf("Alone i-pad  i=%d, k=%d\n", i, k);
      projCh1[k] = ch1[j];
    }
    ij_ptr++;
  }

  // Just add alone pads of cathode 0
  if ( includeAlonePads ) {
    ij_ptr = IInterJ;
    for( PadIdx_t i=0; i < N0; i++) {
      k = aloneIPads[i];
      if ( k >= 0) {
        // printf("Alone i-pad  i=%d, k=%d\n", i, k);
        projCh1[k] = ch0[i];
      } 
    }
  }
  
  if (VERBOSE) printXYdXY( "Projection", projected_xyDxy, maxNbrOfProjPads, nbrOfProjPads, projCh0, projCh1);
  return nbrOfProjPads;
}

// ??? must be split: projected grid AND charge projection
int projectChargeOnOnePlaneWithTheta( 
        const double *xy0InfSup, const double *ch0, 
        const double *xy1InfSup, const double *ch1, 
        const double *chTheta0, const double *chTheta1, 
        PadIdx_t N0, PadIdx_t N1, int includeAlonePads, double *qProj) {
  // xy0InfSup, ch0, N0 : cathode 0
  // xy1InfSup, ch1, N1 : cathode 1
  // i: describes the cathode-0 pads
  // j: describes the cathode-1 pads
  // double maxFloat = DBL_MAX;
  double epsilon = 1.0e-4;
  const double *x0Inf = getConstXInf( xy0InfSup, N0);
  const double *y0Inf = getConstYInf( xy0InfSup, N0);
  const double *x0Sup = getConstXSup( xy0InfSup, N0);
  const double *y0Sup = getConstYSup( xy0InfSup, N0);
  const double *x1Inf = getConstXInf( xy1InfSup, N1);
  const double *y1Inf = getConstYInf( xy1InfSup, N1);
  const double *x1Sup = getConstXSup( xy1InfSup, N1);
  const double *y1Sup = getConstYSup( xy1InfSup, N1);
  /*
  // alocate in heap    
  intersectionMatrix = new PadIdx_t[N0*N1];
  // mapIJToK = (PadIdx_t *) malloc( N0*N1 * sizeof(PadIdx_t))
  mapIJToK = new PadIdx_t[N0*N1];
  PadIdx_t aloneIPads[N0], aloneJPads[N1];
  vectorSetZeroInt( intersectionMatrix, N0*N1);
  vectorSetInt( mapIJToK, -1, N0*N1);
  vectorSetInt( aloneIPads, -1, N0);
  vectorSetInt( aloneJPads, -1, N1);
  //
  // Looking for j pads, intercepting pad i
  // Build the intersection matrix
  //
  double xmin, xmax, ymin, ymax;
  PadIdx_t xInter, yInter;
  for( PadIdx_t i=0; i < N0; i++) {
    for( PadIdx_t j=0; j < N1; j++) {
      xmin = fmax( x0Inf[i], x1Inf[j] );
      xmax = fmin( x0Sup[i], x1Sup[j] );
      xInter = ( xmin <= (xmax - epsilon) );
      ymin = fmax( y0Inf[i], y1Inf[j] );
      ymax = fmin( y0Sup[i], y1Sup[j] );
      yInter = ( ymin <= (ymax - epsilon));
      intersectionMatrix[i*N1+j] =  (xInter & yInter);
    }
  }
  //  
  if (VERBOSE) printMatrixInt( "  Intersection Matrix", intersectionMatrix, N0, N1);
  //
  // Compute the max number of projected pads to make 
  // memory allocations 
  //
  maxNbrOfProjPads = vectorSumInt( intersectionMatrix, N0*N1 );
  int nbrOfAlonePads = 0;
  if (includeAlonePads) {
    // Add alone cath0-pads 
    for( PadIdx_t i=0; i < N0; i++) {
      if ( vectorSumRowInt( &intersectionMatrix[i*N1], N0, N1 ) == 0) nbrOfAlonePads++; 
    }
    // Add alone cath1-pads 
    for( PadIdx_t j=0; j < N1; j++) {
      if ( vectorSumColumnInt( &intersectionMatrix[j], N0, N1) == 0) nbrOfAlonePads++; 
    }  
  }
  // Add alone pas and row/column separators
  maxNbrOfProjPads += nbrOfAlonePads + fmax( N0, N1);
  if (VERBOSE) printf("  maxNbrOfProjPads %d\n", maxNbrOfProjPads);
  //
  //
  // Projected pad allocation
  // The limit maxNbrOfProjPads is alocated
  //
  projected_xyDxy = new double [4*maxNbrOfProjPads];
  projX  = getX ( projected_xyDxy, maxNbrOfProjPads);
  projY  = getY ( projected_xyDxy, maxNbrOfProjPads);
  projDX = getDX( projected_xyDxy, maxNbrOfProjPads);
  projDY = getDY( projected_xyDxy, maxNbrOfProjPads);
  //
  // Intersection Matrix Sparse representation
  //
  IInterJ = new PadIdx_t[maxNbrOfProjPads]; 
  JInterI = new PadIdx_t[maxNbrOfProjPads]; 
  int checkr = getIndexByRow( intersectionMatrix, N0, N1, IInterJ);

  int checkc = getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
  if (CHECK) { 
    if (( checkr > maxNbrOfProjPads) || (checkc > maxNbrOfProjPads)) {
      printf("Allocation pb for  IInterJ or JInterI: allocated=%d, needed for row=%d, for col=%d \n", 
          maxNbrOfProjPads, checkr, checkc);
      throw std::overflow_error(
         "Allocation pb for  IInterJ or JInterI" );
    }
  }
  if (VERBOSE) {
    printInterMap("  IInterJ", IInterJ, N0 );
    printInterMap("  JInterI", JInterI, N1 );
  }
  mapKToIJ = new MapKToIJ_t[maxNbrOfProjPads];

  //
  // Build the new pads
  //
  nbrOfProjPads = buildProjectedPads( xy0InfSup, xy1InfSup, N0, N1, 
                                      aloneIPads, aloneJPads, includeAlonePads );
  
  if (CHECK == 1) checkConsistencyMapKToIJ( aloneIPads, aloneJPads, N0, N1);
  
  //
  // Get the isolated new pads
  // (they have no neighborhing)
  //
  int thereAreIsolatedPads = 0;
  PadIdx_t *neigh = getFirstNeighbors( projected_xyDxy, nbrOfProjPads, maxNbrOfProjPads ); 
  // printNeighbors();
  MapKToIJ_t ij;
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    if (getTheFirstNeighborOf(neighbors, k) == -1) {
      // pad k is isolated     
      thereAreIsolatedPads = 1;
      ij = mapKToIJ[k];
      if( (ij.i >= 0) && (ij.j >= 0)) { 
        if (VERBOSE) printf(" Isolated pad: nul intersection i,j = %d %d\n", ij.i, ij.j);
        intersectionMatrix[ij.i*N1 + ij.j] = 0; 
      } else {
        throw std::overflow_error("I/j negative (alone pad)");
      }
    }
  }
  if (VERBOSE && thereAreIsolatedPads) printf("There are isolated pads %d\n", thereAreIsolatedPads);
  //
  if( thereAreIsolatedPads == 1) {
    // Recompute all
    getIndexByRow( intersectionMatrix, N0, N1, IInterJ);
    getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
    //
    // Build the new pads
    //
    nbrOfProjPads = buildProjectedPads( xy0InfSup, xy1InfSup, N0, N1, 
                                      aloneIPads, aloneJPads, includeAlonePads );
    getFirstNeighbors( projected_xyDxy, nbrOfProjPads, maxNbrOfProjPads );
  }
  */
  //
  // Computing charges of the projected pads
  // Ch0 part
  //
  double qProj0[nbrOfProjPads];
  double qProj1[nbrOfProjPads];
  PadIdx_t k=0;
  double sumCh1ByRow;
  PadIdx_t *ij_ptr = IInterJ;
  PadIdx_t *rowStart;
  for( PadIdx_t i=0; i < N0; i++) {
    // Save the starting index of the begining of the row
    rowStart = ij_ptr;
    // sum of charge with intercepting j-pad  
    // New formula for( sumCh1ByRow = 0.0; *ij_ptr != -1; ij_ptr++) sumCh1ByRow += ch1[ *ij_ptr ];
    for( sumCh1ByRow = 0.0; *ij_ptr != -1; ij_ptr++) sumCh1ByRow += ch1[ *ij_ptr ] * chTheta1[ *ij_ptr ] ;
    if ( sumCh1ByRow != 0.0 ) { 
      double cst = ch0[i] / sumCh1ByRow;
      for( ij_ptr = rowStart; *ij_ptr != -1; ij_ptr++) {
        // New formula qProj0[k] = ch1[ *ij_ptr ] * cst;
        qProj0[k] = ch1[ *ij_ptr ] * chTheta1[ *ij_ptr ] * cst;
        // Debug
        // printf(" i=%d, j=%d, k=%d, sumCh0ByCol = %g, qProj1[k]= %g \n", i, *ij_ptr, k, sumCh1ByRow, qProj0[k]);
        k++;
      }
    } else if (includeAlonePads){
      // Alone i-pad
      qProj0[k] = ch0[i];
      k++;      
    }
    // Move on a new row
    ij_ptr++;
  }
  // Just add alone pads of cathode 1
  if ( includeAlonePads ) {
    for( PadIdx_t j=0; j < N1; j++) {
      k = aloneJPads[j];
      if ( k >= 0 ){
        qProj0[k] = ch1[j];
      }
    }
  }
  //
  // Computing charges of the projected pads
  // Ch1 part
  //
  k=0; 
  double sumCh0ByCol;
  ij_ptr = JInterI;
  PadIdx_t *colStart;
  for( PadIdx_t j=0; j < N1; j++) {
    // Save the starting index of the beginnig of the column
    colStart = ij_ptr;
    // sum of charge intercepting i-pad  
    // New formula for( sumCh0ByCol = 0.0; *ij_ptr != -1; ij_ptr++) sumCh0ByCol += ch0[ *ij_ptr ];
    for( sumCh0ByCol = 0.0; *ij_ptr != -1; ij_ptr++) sumCh0ByCol += ch0[ *ij_ptr ] * chTheta0[ *ij_ptr ];
    if ( sumCh0ByCol != 0.0 ) { 
      double cst = ch1[j] / sumCh0ByCol;
      for( ij_ptr = colStart; *ij_ptr != -1; ij_ptr++) {
        PadIdx_t i = *ij_ptr;
        k = mapIJToK[i * N1 + j];
        // New formula qProj1[k] = ch0[i] * cst;
        qProj1[k] = ch0[i] * chTheta0[i] * cst;
        // Debug
        // printf(" j=%d, i=%d, k=%d, sumCh0ByCol = %g, qProj1[k]= %g \n", j, i, k, sumCh0ByCol, qProj1[k]);
      }
    } else if (includeAlonePads){
      // Alone j-pad
      k = aloneJPads[j];
      if ( CHECK && (k < 0) ) printf("ERROR: Alone j-pad with negative index j=%d\n", j);
      // printf("Alone i-pad  i=%d, k=%d\n", i, k);
      qProj1[k] = ch1[j];
    }
    ij_ptr++;
  }

  // Just add alone pads of cathode 0
  if ( includeAlonePads ) {
    ij_ptr = IInterJ;
    for( PadIdx_t i=0; i < N0; i++) {
      k = aloneIPads[i];
      if ( k >= 0) {
        // printf("Alone i-pad  i=%d, k=%d\n", i, k);
        qProj1[k] = ch0[i];
      } 
    }
  }
  
  if (VERBOSE) printXYdXY( "Projection with theta", projected_xyDxy, maxNbrOfProjPads, nbrOfProjPads, qProj0, qProj1);
  // Make the average
  for( int i=0; i< nbrOfProjPads; i++) {
    qProj[i] = 0.5 * (qProj0[i] + qProj1[i]);
  }
  // For debugging (inspect)
  for( int i=0; i< nbrOfProjPads; i++) {
    projCh0[i] = qProj0[i];
    projCh1[i] = qProj1[i];
  }
  return nbrOfProjPads;
}

int getConnectedComponentsOfProjPads( short *padGrp ) { 
  // Class from neighbors list of Projected pads, the pads in groups (connected components) 
  // padGrp is set to the group Id of the pad. 
  // If the group Id is zero, the the pad is unclassified
  // Return the number of groups
  int  N = nbrOfProjPads;
  PadIdx_t *neigh = neighbors;
  PadIdx_t neighToDo[N]; 
  vectorSetZeroShort( padGrp, N);  
  int nbrOfPadSetInGrp = 0;
  // Last padGrp to process
  short *curPadGrp = padGrp;
  short currentGrpId = 0;
  //
  int i, j, k;
  // printNeighbors();
  while (nbrOfPadSetInGrp < N) {
    // Seeking the first unclassed pad (padGrp[k]=0)
    for( ; (curPadGrp < &padGrp[N]) && *curPadGrp != 0; curPadGrp++ );
    k = curPadGrp - padGrp;
    // printf( "\nnbrOfPadSetInGrp %d\n", nbrOfPadSetInGrp);
    //
    // New group for k - then search all neighbours of k
    currentGrpId++;
    padGrp[k] = currentGrpId;
    nbrOfPadSetInGrp++;
    PadIdx_t startIdx = 0, endIdx = 1;
    neighToDo[ startIdx ] = k;
    // Labels k neighbors
    for( ; startIdx < endIdx; startIdx++) {
      i = neighToDo[startIdx];
      // printf("i %d neigh[i] %d: ", i, neigh[i] );
      //
      // Scan i neighbors
      for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh,i); *neigh_ptr != -1; neigh_ptr++) {
        j = *neigh_ptr;
        // printf("j %d\n, ", j);
        if (padGrp[j] == 0) {
          // Add the neighbors in the currentgroup  
          //
          // printf("found %d\n", j);
          padGrp[ j ] = currentGrpId;
          nbrOfPadSetInGrp++;
          // Append in the neighbor list to search
          neighToDo[ endIdx] = j;
          endIdx++;
        }
      }
    }
    // printf("make groups grpId=%d, nbrOfPadSetInGrp=%d\n", currentGrpId, nbrOfPadSetInGrp);
  }
  // return tne number of Grp
  return currentGrpId;
}

int getConnectedComponentsOfProjPadsWOIsolatedPads( short *padGrp ) { 
  // Class from neighbors list of Projected pads, the pads in groups (connected components) 
  // padGrp is set to the group Id of the pad. 
  // If the group Id is zero, the the pad is unclassified
  // Return the number of groups
  int  N = nbrOfProjPads;
  PadIdx_t *neigh = neighbors;
  PadIdx_t neighToDo[N]; 
  vectorSetZeroShort( padGrp, N);  
  int nbrOfPadSetInGrp = 0;
  // Last padGrp to process
  short *curPadGrp = padGrp;
  short currentGrpId = 0;
  //
  int i, j, k;
  // printNeighbors();
  printf("getConnectedComponentsOfProjPadsWOIsolatedPads\n");
  printf("-------------------aloneKPads %p \n", aloneKPads);

  while (nbrOfPadSetInGrp < N) {
    // Seeking the first unclassed pad (padGrp[k]=0)
    for( ; (curPadGrp < &padGrp[N]) && *curPadGrp != 0; curPadGrp++ );
    k = curPadGrp - padGrp;
    if (VERBOSE > 1) { 
      printf( "  k=%d, nbrOfPadSetInGrp g=%d: n=%d\n", k, currentGrpId, nbrOfPadSetInGrp);
    }
    //
    // New group for k - then search all neighbours of k
    // aloneKPads = 0 if only one cathode
    if (aloneKPads && (aloneKPads[k] != -1)) {
      // Alone Pad no group at the moment
      if (VERBOSE > 1) { 
        printf("  isolated pad %d\n", k);
      }
      padGrp[k] = -1;
      nbrOfPadSetInGrp++;  
      continue;
    }
    currentGrpId++;
    if (VERBOSE > 1) { 
      printf( "  NEW GRP, pad k=%d in new grp=%d\n", k, currentGrpId);
    }
    padGrp[k] = currentGrpId;
    nbrOfPadSetInGrp++;
    PadIdx_t startIdx = 0, endIdx = 1;
    neighToDo[ startIdx ] = k;
    // Labels k neighbors
    // Propagation of the group in all neighbour list
    for( ; startIdx < endIdx; startIdx++) {
      i = neighToDo[startIdx];
      if (VERBOSE > 1) { 
        printf("  propagate to neighbours of i=%d ", i );
      }
      //
      // Scan i neighbors
      for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh,i); *neigh_ptr != -1; neigh_ptr++) {
        j = *neigh_ptr;
        // printf("    neigh j %d\n, \n", j);
        if ((padGrp[j] == 0)  ) {
          // Add the neighbors in the currentgroup  
          //
          // aloneKPads = 0 if only one cathode
          if (aloneKPads && (aloneKPads[j] != -1)) {
            if (VERBOSE > 1) { 
              printf("isolated pad %d, ", j);
            }
            padGrp[j] = -1;
            nbrOfPadSetInGrp++;               
            continue;
          }
          if (VERBOSE > 1) { 
            printf("%d, ", j);
          }
          padGrp[ j ] = currentGrpId;
          nbrOfPadSetInGrp++;
          // Append in the neighbor list to search
          neighToDo[ endIdx] = j;
          endIdx++;
        }
      }
      if (VERBOSE > 1) { 
        printf("\n");
      }
    }
    // printf("make groups grpId=%d, nbrOfPadSetInGrp=%d\n", currentGrpId, nbrOfPadSetInGrp);
  }
  for ( int k=0; k < N; k++) {
    if ( padGrp[k] == -1) {
      padGrp[k] = 0; 
    }
  }
  // return tne number of Grp
  return currentGrpId;
}

int laplacian2D(const double *xyDxy, const double *q, PadIdx_t *neigh, int N, int chId, PadIdx_t *sortedLocalMax, int kMax, double *smoothQ) {
  // ??? Place somewhere
  double eps = 1.0e-7;
  double noise = 4. * 0.22875;
  double laplacianCutOff = noise;
  // ??? Inv int atLeastOneMax = -1;
  //
  const double *x  = getConstX( xyDxy, N);
  const double *y  = getConstY( xyDxy, N);
  const double *dx = getConstDX( xyDxy, N);
  const double *dy = getConstDY( xyDxy, N);

  // 
  // Laplacian allocation
  double lapl[N]; 
  // Locations not used as local max
  Mask_t unselected[N];
  vectorSetShort( unselected, 1, N );
  // printNeighbors(neigh, N);
  for (int i=0; i< N; i++) {
    /* ?? Inv 
    zi = z[i];
    nSupi = 0;
    nInfi = 0;
    */
    int nNeigh = 0;
    double sumNeigh = 0;
    int nNeighSmaller = 0;
    // printf("  Neighbors of i=%d [", i);
    //
    // For all neighbours of i
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      // printf("%d ,", j);
      // nNeighSmaller += (q[j] <= ((q[i] + noise) * unselected[i]));
      nNeighSmaller += (q[j] <= ((q[i] + noise) * unselected[j]));
      nNeigh++;
      sumNeigh += q[j];      
    }
    // printf("]");
    // printf(" nNeighSmaller %d / nNeigh %d \n", nNeighSmaller, nNeigh);
    lapl[i] = float(nNeighSmaller) / nNeigh;
    if (lapl[i] < laplacianCutOff) {
      lapl[i] = 0.0;
    }
    unselected[i] = (lapl[i] != 1.0);
    smoothQ[i] =  sumNeigh / nNeigh;
    if (1 || VERBOSE)
      printf("Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, smoothQ[i]=%6.3f, lapl[i]=%6.3f\n", i, x[i], y[i], q[i], smoothQ[i], lapl[i]);    
  } 
  //
  // Get local maxima
  Mask_t localMaxMask[N];
  vectorBuildMaskEqual( lapl, 1.0, N, localMaxMask);
  // Get the location in lapl[]
  // Inv ??? int nSortedIdx = vectorSumShort( localMaxMask, N );
  // ??? Inv int sortPadIdx[nSortedIdx];
  int nSortedIdx = vectorGetIndexFromMask( localMaxMask, N, sortedLocalMax );
  // Sort the slected laplacian (index sorting)
  // Indexes for sorting
  // Rq: Sometimes chage the order of max
  // std::sort( sortedLocalMax, &sortedLocalMax[nSortedIdx], [=](int a, int b){ return smoothQ[a] > smoothQ[b]; });
  std::sort( sortedLocalMax, &sortedLocalMax[nSortedIdx], [=](int a, int b){ return q[a] > q[b]; });
  if (1 || VERBOSE) {
    vectorPrint("  sort w", q, N);
    vectorPrintInt("  sorted q-indexes", sortedLocalMax, nSortedIdx);
  }
  
  ////
  // Filtering local max 
  ////
 
  printf("FILTERing Max\n");
  // At Least one locMax
  if ((nSortedIdx == 0) && (N!=0)) {
    // Take the first pad
    printf("-> No local Max, take the highest value < 1\n");
    sortedLocalMax[0] = 0;
    nSortedIdx = 1;
    return nSortedIdx;
  }
  
  // For a small number of pads 
  // limit the number of max to 1 local max
  // if the aspect ratio of the cluster 
  // is close to 1
  double aspectRatio=0;
  if ( (N > 0) && (N < 6) && (chId <= 6) ) {
    double xInf=DBL_MAX, xSup=DBL_MIN, yInf=DBL_MAX, ySup=DBL_MIN; 
    // Compute aspect ratio of the cluster
    for (int i=0; i <N; i++) {
      xInf = fmin( xInf, x[i] - dx[i]);
      xSup = fmax( xSup, x[i] + dx[i]);
      yInf = fmin( yInf, y[i] - dy[i]);
      ySup = fmax( ySup, y[i] + dy[i]);
    }
    // Nbr of pads in x-direction
    int nX = int((xSup - xInf) / dx[0] + eps);
    // Nbr of pads in y-direction
    int nY = int((ySup - yInf) / dy[0] + eps);
    aspectRatio = fmin( nX, nY) / fmax( nX, nY);
    if ( aspectRatio > 0.6 ) {
      // Take the max
      nSortedIdx = 1;
      printf("-> Limit to one local Max, nPads=%d, chId=%d, aspect ratio=%6.3f\n", N, chId, aspectRatio);
    }
  }
  
  // Suppress noisy peaks  when at least 1 peak 
  // is bigger than  
  if ( (N > 0) && (q[ sortedLocalMax[0]] > 2*noise) ) {
    int trunkIdx=nSortedIdx;
    for ( int ik=0; ik < nSortedIdx; ik++) {
      if ( q[ sortedLocalMax[ik]] <= 2*noise) {
        trunkIdx = ik;
      }
    }
    nSortedIdx = std::max(trunkIdx, 1);
    if (trunkIdx != nSortedIdx) {
      printf("-> Suppress %d local Max. too noisy (q < %6.3f),\n", nSortedIdx-trunkIdx, 2*noise);    
    }
    
  }
  // At most 
  // int nbrOfLocalMax = floor( (N + 1) / 3.0 );
  // if  ( nSortedIdx > nbrOfLocalMax) {
  //  printf("Suppress %d local Max. the limit of number of local max %d is reached (< %d)\n", nSortedIdx-nbrOfLocalMax, nSortedIdx, nbrOfLocalMax);    
  //  nSortedIdx = nbrOfLocalMax;
  //}
  

  
  if (VERBOSE) {
      
  }
  
  return nSortedIdx;
}

int findLocalMaxWithBothCathodes( const double *xyDxy0, const double *q0, int N0, 
        const double *xyDxy1, const double *q1, int N1, const double *xyDxyProj, int NProj, int chId, const PadIdx_t *mapGrpIdxToI, const PadIdx_t *mapGrpIdxToJ, int nbrCath0, int nbrCath1, double *thetaOut, int kMax ) {
// ???? Test if 
  // Max number of seeds 
  // ???? Do the filtering somewhere int kMax0 = ceil( (N0+N1 + 1)/3 );
  // int kMax1 = ceil( (N0+N1 + 1)/3 );
  int kMax0 = N0;
  int kMax1 = N1;
  // Number of seeds founds
  int k=0;
  // 
  // Pad indexes of local max.
  PadIdx_t localMax0[kMax0];
  PadIdx_t localMax1[kMax1];
  // Smothed values of q[0/1] with neighbours
  double smoothQ0[N0];
  double smoothQ1[N1];
  // Local Maximum for each cathodes
  // There are sorted with the lissed q[O/1] values
  printf("findLocalMaxWithBothCathodes N0=%d N1=%d\n", N0, N1);
  if ( N0 ) {
    grpNeighborsCath0 = getFirstNeighbors( xyDxy0, N0, N0 );
    // printNeighbors(grpNeighborsCath0, N0);
  }
  if ( N1 ) {
    grpNeighborsCath1 = getFirstNeighbors( xyDxy1, N1, N1 );
    // printNeighbors(grpNeighborsCath1, N1);
  }
  int K0 = laplacian2D( xyDxy0, q0, grpNeighborsCath0, N0, chId, localMax0, kMax0, smoothQ0);
  int K1 = laplacian2D( xyDxy1, q1, grpNeighborsCath1, N1, chId, localMax1, kMax1, smoothQ1);
  // Seed allocation
  double localXMax[K0+K1];
  double localYMax[K0+K1];
  double localQMax[K0+K1];
  //
  // Need an array to transform global index to the grp indexes
  PadIdx_t mapIToGrpIdx[nbrCath0];
  vectorSetInt(mapIToGrpIdx, -1, nbrCath0);

  for (int i=0; i<N0; i++) {
    // ??? printf("mapGrpIdxToI[%d]=%d\n", i, mapGrpIdxToI[i]);
    mapIToGrpIdx[ mapGrpIdxToI[i]] = i;  
  }
  PadIdx_t mapJToGrpIdx[nbrCath1];
  vectorSetInt(mapJToGrpIdx, -1, nbrCath1);
  for (int j=0; j<N1; j++) {
    // ??? printf("mapGrpIdxToJ[%d]=%d\n", j, mapGrpIdxToJ[j]);
    mapJToGrpIdx[ mapGrpIdxToJ[j]] = j;  
  }
  // ???
  // vectorPrintInt( "mapIToGrpIdx", mapIToGrpIdx, nbrCath0);
  // vectorPrintInt( "mapJToGrpIdx", mapJToGrpIdx, nbrCath1);
  if (VERBOSE) {
    vectorPrint("findLocalMax q0", q0, N0);
    vectorPrint("findLocalMax q1", q1, N1);
    vectorPrintInt("findLocalMax localMax0", localMax0, K0);
    vectorPrintInt("findLocalMax localMax1", localMax1, K1);
  }
  
  const double *x0  = getConstX( xyDxy0, N0);
  const double *y0  = getConstY( xyDxy0, N0);
  const double *dx0 = getConstDX( xyDxy0, N0);
  const double *dy0 = getConstDY( xyDxy0, N0);

  const double *x1  = getConstX( xyDxy1, N1);
  const double *y1  = getConstY( xyDxy1, N1);
  const double *dx1 = getConstDX( xyDxy1, N1);
  const double *dy1 = getConstDY( xyDxy1, N1);
  
  const double *xProj  = getConstX( xyDxyProj, NProj);
  const double *yProj = getConstY( xyDxyProj, NProj);
  const double *dxProj = getConstDX( xyDxyProj, NProj);
  const double *dyProj = getConstDY( xyDxyProj, NProj);
  
  //
  // Make the combinatorics between the 2 cathodes
  // - Take the maxOf( N0,N1) for the external loop
  //
  if (1 || VERBOSE) {
    printf("K0=%d, K1=%d\n", K0, K1);
  }
  bool K0GreaterThanK1 = ( K0 >= K1 );
  bool K0EqualToK1 = ( K0 == K1 );
  // Choose the highest last local max.
  bool highestLastLocalMax0;
  if (K0 == 0) {
    highestLastLocalMax0 = false;
  } else if (K1 == 0){
    highestLastLocalMax0 = true;
  } else {
    // highestLastLocalMax0 = (smoothQ0[localMax0[std::max(K0-1, 0)]] >= smoothQ1[localMax1[std::max(K1-1,0)]]);
    highestLastLocalMax0 = (q0[localMax0[std::max(K0-1, 0)]] >= q1[localMax1[std::max(K1-1,0)]]);
  }
  // Permute cathodes if necessary 
  int NU, NV;
  int KU, KV;
  PadIdx_t *localMaxU, *localMaxV;
  const double *qU, *qV;
  PadIdx_t *interUV;
  bool permuteIJ;
  const double *xu, *yu, *dxu, *dyu;
  const double *xv, *yv, *dxv, *dyv;
  const PadIdx_t *mapGrpIdxToU, *mapGrpIdxToV;
  PadIdx_t  *mapUToGrpIdx, *mapVToGrpIdx;

  // Do permutation between cath0/cath1 or not
  if( K0GreaterThanK1 || ( K0EqualToK1 && highestLastLocalMax0)) {
    NU = N0; NV=N1;
    KU = K0; KV = K1;
    xu=x0;  yu=y0; dxu=dx0;  dyu=dy0;
    xv=x1;  yv=y1; dxv=dx1;  dyv=dy1;
    localMaxU = localMax0; localMaxV = localMax1;
    // qU = smoothQ0; qV = smoothQ1;
    qU = q0; qV = q1;
    interUV = IInterJ;
    mapGrpIdxToU = mapGrpIdxToI;
    mapGrpIdxToV = mapGrpIdxToJ;
    mapUToGrpIdx = mapIToGrpIdx;
    mapVToGrpIdx = mapJToGrpIdx;
    permuteIJ = false;
  }
  else {
    NU = N1; NV = N0;
    KU = K1; KV = K0;
    xu=x1;  yu=y1; dxu=dx1;  dyu=dy1;
    xv=x0;  yv=y0; dxv=dx0;  dyv=dy0;
    localMaxU = localMax1; localMaxV = localMax0;
    // qU = smoothQ1; qV = smoothQ0;
    qU = q1; qV = q0;
    interUV = JInterI;  
    mapGrpIdxToU = mapGrpIdxToJ;
    mapGrpIdxToV = mapGrpIdxToI;
    mapUToGrpIdx = mapJToGrpIdx;
    mapVToGrpIdx = mapIToGrpIdx;
    permuteIJ = true;
  }
  // Keep the memory of the localMaxV already assigned
  Mask_t qvAvailable[KV];
  vectorSetShort( qvAvailable, 1, KV);
  // Compact intersection matrix
  PadIdx_t *UInterV;
  //
  // Cathodes combinatorics
  if (VERBOSE) {
    printf("Local max combinatorics: KU=%d KV=%d\n", KU, KV);
    // printXYdXY("Projection", xyDxyProj, NProj, NProj, 0, 0);
    printf("findLocalMaxWithBothCathodes mapIJToK=%p, N0=%d N1=%d\n", mapIJToK, N0, N1);
    for (int i=0; i <N0; i++) {
      int ii = mapGrpIdxToI[i];
      for (int j=0; j <N1; j++) {
        int jj = mapGrpIdxToJ[j];
        //if ( (mapIJToK[ii*nbrCath1+jj] != -1))
              printf(" %d inter %d, grp : %d inter %d yes=%d\n", ii, jj, i, j, mapIJToK[ii*nbrCath1+jj]);
      }
    }
  }
  for(int u=0; u<KU; u++) {
    // 
    PadIdx_t uPadIdx = localMaxU[u];
    double maxValue = 0.0;
    // Cathode-V pad index of the max (localMaxV)
    PadIdx_t maxPadVIdx = -1;
    // Index in the maxCathv
    PadIdx_t maxCathVIdx = -1;
    // Choose the best localMaxV
    // i.e. the maximum value among 
    // the unselected localMaxV
    //
    // uPadIdx in index in the Grp
    // need to get the cluster index
    // to checck the intersection
    int ug = mapGrpIdxToU[uPadIdx];
    printf("Cathode u=%d localMaxU[u]=%d, x,y= %6.3f,  %6.3f, q=%6.3f\n", u, localMaxU[u], xu[localMaxU[u]], yu[localMaxU[u]], qU[localMaxU[u]]);
    bool interuv;
    for(int v=0; v<KV; v++) {
      PadIdx_t vPadIdx = localMaxV[v];
      int vg = mapGrpIdxToV[vPadIdx];
      if (permuteIJ) {
         // printf("uPadIdx=%d,vPadIdx=%d, mapIJToK[vPadIdx*N0+uPadIdx]=%d permute\n",uPadIdx,vPadIdx, mapIJToK[vPadIdx*N0+uPadIdx]);
         interuv = (mapIJToK[vg*nbrCath1+ug] != -1);
      }
      else {
         //printf("uPadIdx=%d,vPadIdx=%d, mapIJToK[uPadIdx*N1+vPadIdx]=%d\n",uPadIdx,vPadIdx, mapIJToK[uPadIdx*N1+vPadIdx]);
         interuv = (mapIJToK[ug*nbrCath1+vg] != -1);
      }      
      if (interuv) {
        double val = qV[vPadIdx] * qvAvailable[v];
        if (val > maxValue) {
          maxValue = val;
          maxCathVIdx = v;
          maxPadVIdx = vPadIdx;
        }
      }
    }
    // A this step, we've got (or not) an 
    // intercepting pad v with u. This v is 
    // the maximum of all possible values
    // ??? printf("??? maxPadVIdx=%d, maxVal=%f\n", maxPadVIdx, maxValue);
    if (maxPadVIdx != -1) {
      // Found an intersevtion and a candidate
      // add in the list of seeds
      PadIdx_t kProj;
      int vg = mapGrpIdxToV[maxPadVIdx];
      if( permuteIJ) {
        kProj = mapIJToK[vg*nbrCath1 + ug];  
      }
      else {
        kProj = mapIJToK[ug*nbrCath1 + vg]; 
      }
      // mapIJToK and projection UNUSED ????
      localXMax[k] = xProj[kProj];
      localYMax[k] = yProj[kProj];
      // localQMax[k] = 0.5 * (qU[uPadIdx] + qV[maxPadVIdx]);
      localQMax[k] = qU[uPadIdx];
      /*
      double xyMin = fmax( xu[uPadIdx] - dxu[uPadIdx], xv[maxPadVIdx] - dxv[maxPadVIdx]);
      double xyMax = fmin( xu[uPadIdx] + dxu[uPadIdx], xv[maxPadVIdx] + dxv[maxPadVIdx]);
      localXMax[k] = 0.5 * (xyMin+xyMax);
      xyMin = fmax( yu[uPadIdx] - dyu[uPadIdx], yv[maxPadVIdx] - dyv[maxPadVIdx]);
      xyMax = fmin( yu[uPadIdx] + dyu[uPadIdx], yv[maxPadVIdx] + dyv[maxPadVIdx]);
      localYMax[k] = 0.5 * (xyMin+xyMax);    
      localQMax[k] = 0.5 * (qU[uPadIdx] + qV[maxPadVIdx]);
      */
      // Cannot be selected again as a seed
      qvAvailable[ maxCathVIdx ] = 0;
      if (1 || VERBOSE) {
        printf("  found intersection of u with v: u,v=(%d,%d) , x=%f, y=%f, w=%f\n", u, maxCathVIdx, localXMax[k],localYMax[k], localQMax[k]);
        // printf("Projection u=%d, v=%d, uPadIdx=%d, ,maxPadVIdx=%d, kProj=%d, xProj[kProj]=%f, yProj[kProj]=%f\n", u, maxCathVIdx, 
        //        uPadIdx, maxPadVIdx, kProj, xProj[kProj], yProj[kProj] );
        //kProj = mapIJToK[maxPadVIdx*N0 + uPadIdx]; 
        //printf(" permut kProj=%d xProj[kProj], yProj[kProj] = %f %f\n", kProj, xProj[kProj], yProj[kProj] );
      }
      k++;
    }
    else {
      // No intersection u with localMaxV set
      // Approximate the seed position
      //
      // Search v pads intersepting u
      PadIdx_t *uInterV;
      PadIdx_t uPad = 0;
      /*
      for( i=0; i < N0; i++) {
      // Nbr of j-pads intersepting  i-pad
      for( countIInterJ = 0; *ij_ptr != -1; countIInterJ++, ij_ptr++) {
      */
      printf("  No intersection between u=%d and v-set of , approximate the location\n", u);

      // Go to the mapGrpIdxToU[uPadIdx] (???? mapUToGrpIdx[uPadIdx])
      uInterV=interUV;
      if (NV != 0 ) {
        for( uInterV=interUV; uPad < ug; uInterV++) {
          if (*uInterV == -1) {
            uPad++;
          }
        }
      } 
      // if (uInterV) printf("??? uPad=%d, uPadIdx=%d *uInterV=%d\n", uPad, uPadIdx, *uInterV);
      // If intercepting pads or no V-Pad 
      if ((NV != 0) && (uInterV[0] != -1) ) {
        double vMin = 1.e+06; 
        double vMax = -1.e+06;        
        // Take the most precise direction
        if (dxu[u] < dyu[u]) {
          // x direction most precise
          // Find the y range intercepting pad u 
          for( ; *uInterV != -1; uInterV++) {
            PadIdx_t idx = mapVToGrpIdx[*uInterV];
            printf(" Global upad=%d intersect global vpad=%d grpIdx=%d\n", uPad, *uInterV, idx); 
            if (idx != -1) {
              vMin = fmin( vMin, yv[idx] - dyv[idx]);
              vMax = fmax( vMax, yv[idx] + dyv[idx]);
            }
          }
          localXMax[k] = xu[uPadIdx];
          localYMax[k] = 0.5*(vMin+vMax);
          localQMax[k] = qU[uPadIdx];
          if( localYMax[k] ==0 ) printf("WARNING localYMax[k] == 0, meaning no intersection");
        }
        else {
          // y direction most precise
          // Find the x range intercepting pad u 
          for( ; *uInterV != -1; uInterV++) {
            PadIdx_t idx = mapVToGrpIdx[*uInterV];
            printf(" Global upad=%d intersect global vpad=%d  grpIdx=%d \n", uPad, *uInterV, idx); 
            if (idx != -1) {
              // printf("y most precise, idx %d \n", idx);
              printf("xv[idx], yv[idx], dxv[idx], dyv[idx]: %6.3f %6.3f %6.3f %6.3f\n", xv[idx], yv[idx], dxv[idx], dyv[idx]);
              vMin = fmin( vMin, xv[idx] - dxv[idx]);
              vMax = fmax( vMax, xv[idx] + dxv[idx]);
            }
          }
          localXMax[k] = 0.5*(vMin+vMax);
          localYMax[k] = yu[uPadIdx];
          localQMax[k] = qU[uPadIdx];
          // printf(" uPadIdx = %d/%d\n", uPadIdx, KU);
          if( localXMax[k] ==0 ) printf("WARNING localXMax[k] == 0, meaning no intersection");
        }
        if (1 || VERBOSE) {
          printf("  solution found with all intersection of u=%d with all v, x more precise %d, position=(%f,%f), qU=%f\n", 
                    u, (dxu[u] < dyu[u]), localXMax[k], localYMax[k], localQMax[k]);
        }
        k++;
      }
      else {
        // No interception in the v-list 
        // or no V pads
        // Takes the u values
        // printf("No intersection of the v-set with u=%d, take the u location", u);

        localXMax[k] = xu[uPadIdx];
        localYMax[k] = yu[uPadIdx];
        localQMax[k] = qU[uPadIdx];
        if (1 || VERBOSE) {
          printf("  No intersection with u, u added in local Max: k=%d u=%d, position=(%f,%f), qU=%f\n", 
                 k, u, localXMax[k], localYMax[k], localQMax[k]);
        }
        k++;
      }
    }
  }
  // Proccess unselected localMaxV
  for(int v=0; v<KV; v++) {
    if (qvAvailable[v]) {
      int l = localMaxV[v];
      localXMax[k] = xv[l];
      localYMax[k] = yv[l];
      localQMax[k] = qV[l];
      if (1 || VERBOSE) {
        printf("Remaining VMax, v added in local Max:  v=%d, position=(%f,%f), qU=%f\n", 
                 v, localXMax[k], localYMax[k], localQMax[k]);
      }
      k++;    
    }
  }
  // k seeds
  double *varX  = getVarX(thetaOut, kMax);
  double *varY  = getVarY(thetaOut, kMax);
  double *muX   = getMuX(thetaOut, kMax);
  double *muY   = getMuY(thetaOut, kMax);
  double *w     = getW(thetaOut, kMax);  
  //
  double wRatio = 0;
  for ( int k_=0; k_ < k; k_++) {
    wRatio += localQMax[k_];
  }
  wRatio = 1.0/wRatio;
  printf("Local max found k=%d kmax=%d\n", k, kMax);
  for ( int k_=0; k_ < k; k_++) {
   muX[k_] = localXMax[k_];
   muY[k_] = localYMax[k_];
   w[k_] = localQMax[k_] * wRatio;  
   printf(" w=%6.3f, mux=%7.3f, muy=%7.3f\n", w[k_], muX[k_], muY[k_]);
   // printf(" localXMax=%6.3f, localMaxY=%7.3f, localQmax=%7.3f\n", localXMax[k_], localYMax[k_], localQMax[k_]);
  }
  if ( N0 ) delete [] grpNeighborsCath0;
  if ( N1 ) delete [] grpNeighborsCath1;
  return k;
}

int findLocalMaxWithLaplacian( const double *xyDxy, const double *z,
        Group_t *padToGrp,
        int nGroups,
        int N, int K, double *laplacian,  double *theta, PadIdx_t *thetaIndexes, Group_t *thetaToGrp) {
  //
  // ??? WARNING : theta must be allocated to N components (# of pads) 
  // ??? WARNING : use neigh of proj-pad
  //
  // Theta's
  double *varX  = getVarX(theta, K);
  double *varY  = getVarY(theta, K);
  double *muX   = getMuX(theta, K);
  double *muY   = getMuY(theta, K);
  double *w     = getW(theta, K);

  const double *X  = getConstX( xyDxy, N);
  const double *Y  = getConstY( xyDxy, N);
  const double *DX = getConstDX( xyDxy, N);
  const double *DY = getConstDY( xyDxy, N);
  //
  PadIdx_t *neigh = neighbors;
  // k is the # of seeds founds
  int k=0;
  double zi;
  int nSupi, nInfi;
  int nNeighi;
  // Group sum & max
  double sumW[nGroups+1];
  double chargeMax[nGroups+1];
  double chargeMin[nGroups+1];
  double maxLapl[nGroups+1];
  int kLocalMax[nGroups+1];
  // printf("??? nbrofGroups %d=\n", nGroups);
  for( int g=0; g < nGroups+1; g++) { sumW[g]=0; maxLapl[g]=0; kLocalMax[g]=0; chargeMin[g]=DBL_MAX; chargeMax[g]= 0.0;}
  //
  // Min/max
  for ( int i=0; i< N; i++) {
    Group_t g = padToGrp[i];
    if ( z[i] > chargeMax[g] ) chargeMax[g] = z[i];
    if ( z[i] < chargeMin[g] ) chargeMin[g] = z[i];
  }
  double zMin = vectorMin( chargeMin,  nGroups+1); 
  double zMax = vectorMax( chargeMax,  nGroups+1); 
  // Min charge wich can be a maximum
  double chargeMinForAMax = zMin + 0.05 * (zMax - zMin);
  //
  //
  int j;
  for ( int i=0; i< N; i++) {
    zi = z[i];
    nSupi = 0;
    nInfi = 0;
    nNeighi = 0;
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      j = *neigh_ptr;
      if( zi >= z[j] ) nSupi++; else  nInfi++;
    }
    nNeighi = nSupi + nInfi;
    if ( (nNeighi <= 2) && (N > 2) ) {
      // Low nbr of neighbors but the # Pads must be > 2
      laplacian[i] = 0;
    } else {
      // TODO Test ((double) nSup -Inf) / nNeigh;
      laplacian[i] = ((double) nSupi) / nNeighi;
    }
    Group_t g = padToGrp[i];
    if (laplacian[i] > maxLapl[g] ) {
      maxLapl[g] = laplacian[i];
    }
    // printf("???? lapl=%g %d %d %d group=%d maxLapl[g]=%g\n", laplacian[i], nSupi, nInfi, nNeighi, g, maxLapl[g]);
    // Strong max
    if ( (laplacian[i] >= 0.99) && (z[i] > chargeMinForAMax) ) {
    // if ( (laplacian[i] >= 0.80) && (z[i] > chargeMinForAMax) ) {
        // save the seed
        w[k] = z[i];
        muX[k] = X[i];
        muY[k] = Y[i];
        thetaToGrp[k] = g;
        sumW[g]  += w[k];
        kLocalMax[g] += 1;
        thetaIndexes[k] = i;
        k++;
    }
  }
  // If there is no max in a group
  for (int g=1; g <= nGroups; g++) {
    if ( VERBOSE) printf("findLocalMaxWithLaplacian: group=%d nLocalMax=%d, maxLaplacian=%7.3g\n", g,  kLocalMax[g], maxLapl[g]);
    // ????
    if ( 0 || kLocalMax[g] == 0 ) {
      double lMax = maxLapl[g];
      for ( int i=0; i< N; i++) {
        if ( (padToGrp[i] == g) && (laplacian[i] == lMax) )  {
          // save the seed
          w[k]   = z[i];
          muX[k] = X[i];
          muY[k] = Y[i];
          thetaToGrp[k] = g;
          sumW[g] += w[k];
          kLocalMax[g] += 1;
          thetaIndexes[k] = i;
          k++;
        }
      }
    }
  }
  //
  // w normalization
  double cst[nGroups];
  for (int g=1; g <= nGroups; g++) {
    if ( kLocalMax[g] != 0 ) {
      cst[g] = 1.0 / sumW[g];
    } else {
      cst[g] = 1.0;
    }
  }
  // 
  for( int l=0; l < k; l++) w[l] = w[l] * cst[ thetaToGrp[l]];
  //
  return k;
}

// With Groups. Not Used ???
int findLocalMaxWithLaplacianV0( const double *xyDxy, const double *z, const PadIdx_t *grpIdxToProjIdx, int N, int xyDxyAllocated, double *laplacian,  double *theta) {
  //
  // ??? WARNING : theta must be allocated to N components (# of pads) 
  // ??? WARNING : use neigh of proj-pad
  //
  // Theta's
  double *varX  = getVarX(theta, N);
  double *varY  = getVarY(theta, N);
  double *muX   = getMuX(theta, N);
  double *muY   = getMuY(theta, N);
  double *w     = getW(theta, N);

  // ??? allocated
  const double *X  = getConstX( xyDxy, xyDxyAllocated);
  const double *Y  = getConstY( xyDxy, xyDxyAllocated);
  const double *DX = getConstDX( xyDxy, xyDxyAllocated);
  const double *DY = getConstDY( xyDxy, xyDxyAllocated);
  //
  PadIdx_t *neigh = neighbors;
  // k is the # of seeds founds
  int k=0;
  double zi;
  int nSupi, nInfi;
  int nNeighi;
  double sumW=0;
  int j;
  for ( int i=0; i< N; i++) {
    zi = z[i];
    nSupi = 0;
    nInfi = 0;
    nNeighi = 0;
    int iProj = ( grpIdxToProjIdx != 0) ? grpIdxToProjIdx[i] : i;
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, iProj); *neigh_ptr != -1; neigh_ptr++) {
      j = *neigh_ptr;
      if (CHECK && ( j >= N )) { 
        printf("findLocalMaxWithLaplacian error: i=%d has the j=%d neighbor but j > N=%d \n", iProj, j, N);
      }
      if( zi >= z[j] ) nSupi++; else  nInfi++;
    }
    nNeighi = nSupi + nInfi;
    if ( nNeighi <= 2) {
      laplacian[i] = 0;
    } else {
      // TODO Test ((double) nSup -Inf) / nNeigh;
      laplacian[i] = ((double) nSupi) / nNeighi;
    }
    // Strong max
    if ( (laplacian[i] >= 0.99) && (z[i] > 5.0) ) {
        // save the seed
        w[k] = z[i];
        muX[k] = X[i];
        muY[k] = Y[i];
        sumW  += w[k];
        k++;
    }
  }
  if (k==0) {
    // No founded seed
    // Weak max/seed
    double lMax = vectorMax(laplacian, N); 
    for ( int i=0; i< N; i++) {
      if ( laplacian[i] == lMax ) {
        // save the seed
        w[k]   = z[i];
        muX[k] = X[i];
        muY[k] = Y[i];
        sumW  += w[k];
        k++;
      }      
    }
  }
  // w normalization
  // ??? should be zero
  if ( k != 0 ) {
    double cst = 1.0 / sumW;
    for( int l=0; l < k; l++) w[l] = w[l] * cst;
  }
  //
  return k;
}

void assignOneCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup) {
  cath0ToGrpFromProj = 0;
  cath1ToGrpFromProj = 0;
  if ( nCath0 != 0) {
    cath0ToGrpFromProj = new short[nCath0];
    vectorCopyShort( padGroup, nCath0, cath0ToGrpFromProj);
  } else {
    cath1ToGrpFromProj = new short[nCath1];
    vectorCopyShort( padGroup, nCath1, cath1ToGrpFromProj);   
  }
  vectorSetShort( wellSplitGroup, 1, nGrp+1);
}

/* ???
void forceSplitCathodes( double *newCath0, double *newcath1) {
    
  // need i/j intersection 
  // Cath0
  for ( int c=0; c < nCath0; c++ ) {
    if ( cath0ToGrp[c] <= 0 ) {
      // find conflicting pads with cath1 (j index)
      // .. i.e intersecting  j with an other group g
      // then separated charge from pads with different groups
      // To do that: project compute coef for group g
      //  S = sum z[u, g], u in same grp g and intercep i
      // For all g :
      //   zi' = zi/ S(g,i)
      // Matrix chargeRatio[ group, cathodes ]
    } else {
      // copy to new cath
         
    }   
  }
}
*/
// Assign a group to the original pads
// No group merging, the charge select the group
void assignPadsToGroupFromProjAndProjCharge( short *projPadGroup, double *chProj, int nProjPads, 
        const PadIdx_t *cath0ToPadIdx, const PadIdx_t *cath1ToPadIdx, 
        int nGrp, int nPads, short *padToCathGrp) {
// outputs:
//   - wellSplitGroup[ nGrp+1]: 1 if the group is well splitted,  0 if not
  vectorSetZeroShort( padToCathGrp, nPads);
  //
  // Max charge of the k-contribution to cathode I/J
  // The array is oveallocated
  double maxChI[ nPads];
  double maxChJ[ nPads];
  vectorSetZero( maxChI , nPads);
  vectorSetZero( maxChJ , nPads);
  //
  PadIdx_t i, j; 
  short g, prevGroup;
  for( int k=0; k < nProjPads; k++) {
    g = projPadGroup[k];
    // give the indexes of overlapping pads
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    if ( (i >= 0) && (cath0ToPadIdx !=0) ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      //
      // cath0ToPadIdx: map cathode-pad to the original pad
      PadIdx_t padIdx = cath0ToPadIdx[i];
      prevGroup = padToCathGrp[ padIdx ];
      if ( (prevGroup == 0) || (prevGroup == g) ) {
        // Case: no group before or same group
        //
        padToCathGrp[ padIdx ] = g;
        // Update the max-charge contribution for i/padIdx
        if( chProj[k] > maxChI[i] ) maxChI[i] = chProj[k]; 
      } else {
        // 
        if ( chProj[k] > maxChI[i] ) {
          padToCathGrp[ padIdx ] = g;
          maxChI[i] = chProj[k]; 
        }
      }
    }
    //
    // Cathode 1
    //
    if ( (j >= 0) && (cath1ToPadIdx != 0) ) {
      // Remark: if j is an alone pad (j<0)
      // j is processed as well
      //
      // cath1ToPadIdx: map cathode-pad to the original pad
      PadIdx_t padIdx = cath1ToPadIdx[j];
      prevGroup = padToCathGrp[padIdx];
      if ( (prevGroup == 0) || (prevGroup == g) ){
         // No group before
         padToCathGrp[padIdx] = g;
        // Update the max-charge contribution for j/padIdx
        if( chProj[k] > maxChJ[j] ) maxChJ[j] = chProj[k]; 
      } else {
        if ( chProj[k] > maxChJ[j] ) {
          padToCathGrp[ padIdx ] = g;
          maxChJ[j] = chProj[k]; 
        }
      }
    }
  }
}


// Assign a group to the original pads
int assignPadsToGroupFromProj( short *projPadGroup, int nProjPads, 
        const PadIdx_t *cath0ToPadIdx, const PadIdx_t *cath1ToPadIdx, 
        int nGrp, int nPads, short *padMergedGrp, short* grpToMergedGrp) {
// outputs:
//   - wellSplitGroup[ nGrp+1]: 1 if the group is well splitted,  0 if not
  // ??? To suppres nPads
  short padToGrp[nPads];
  short matGrpGrp[ (nGrp+1)*(nGrp+1)];
  vectorSetZeroShort( padToGrp, nPads);
  // 
  // vectorSetShort( wellSplitGroup, 1, nGrp+1);
  vectorSetZeroShort( matGrpGrp, (nGrp+1)*(nGrp+1) );
  //
  PadIdx_t i, j; 
  short g, prevGroup;
  for( int k=0; k < nProjPads; k++) {
    g = projPadGroup[k];
    // give the indexes of overlapping pads
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    if ( (i >= 0) && (cath0ToPadIdx !=0) ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      //
      // cath0ToPadIdx: map cathode-pad to the original pad
      PadIdx_t padIdx = cath0ToPadIdx[i];
      prevGroup = padToGrp[ padIdx ];
      if ( (prevGroup == 0) || (prevGroup == g) ) {
        // Case: no group before or same group
        //
        padToGrp[ padIdx ] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1; 
      } else {
        // Already a Grp which differs
        if ( prevGroup > 0) {
          // Invalid prev group
          // wellSplitGroup[ prevGroup ] = 0;
          // Store in the grp to grp matrix  
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1; 
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1; 
        }
        padToGrp[padIdx] = -g;
        // Invalid current group
        // wellSplitGroup[ g ] = 0;
      }
    }
    //
    // Cathode 1
    //
    if ( (j >= 0) && (cath1ToPadIdx != 0) ) {
      // Remark: if j is an alone pad (j<0)
      // j is processed as well
      //
      // cath1ToPadIdx: map cathode-pad to the original pad
      PadIdx_t padIdx = cath1ToPadIdx[j];
      prevGroup = padToGrp[padIdx];
      if ( (prevGroup == 0) || (prevGroup == g) ){
         // No group before
         padToGrp[padIdx] = g;
         matGrpGrp[ g*(nGrp+1) +  g ] = 1; 
      } else {
        // Already a Group 
        if ( prevGroup > 0) {
          // Invalid Old group
          // wellSplitGroup[ prevGroup ] = 0;
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1; 
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1; 
        }
        padToGrp[padIdx] = -g;
        // Invalid current group
        // wellSplitGroup[ g ] = 0;
      }
    }
  }
  if (VERBOSE) printMatrixShort("Group/Group matrix", matGrpGrp, nGrp+1, nGrp+1);
  printf( "AssignPadsToGroupFromProj\n");
  vectorPrintShort( "  padToGrp", padToGrp, nPads);
  //
  // Merge overlapping groups
  //
  // short grpToMergedGrp[nGrp+1];
  vectorSetZeroShort(grpToMergedGrp, nGrp+1);
  int newGroupID = 0;
  int curGroup;
  //
  int iNewGroup = 1;
  while ( iNewGroup < (nGrp+1)) {
    // Define the new group
    if ( grpToMergedGrp[iNewGroup] == 0 ) {
        newGroupID++;
        curGroup= newGroupID;
        grpToMergedGrp[iNewGroup] = newGroupID; 
    } else {
        curGroup = grpToMergedGrp[iNewGroup];
    } 
    
    // printf("new Group idx=%d, newGroupID=%d\n", iNewGroup, newGroupID);
    //
    // Propagate the new grp 
    /// ??? for (int i=iNewGroup; i < (nGrp+1); i++) {
    int i = iNewGroup;
      int ishift = i*(nGrp+1);
      // Check if there are an overlaping group
      for (int j=i+1; j < (nGrp+1); j++) {
        if ( matGrpGrp[ishift+j] ) {
          /*
          if (CHECK) {
            if ( (grpToGrp[j] != 0) && (grpToGrp[j] != i) ) {
              printf("The mapping grpTogrp can't have 2 group values (surjective) oldGrp=%d newGrp=%d\n", grpToGrp[j], i);
              throw std::overflow_error("The mapping grpTogrp can't have 2 group values");
            }
          }
          */
          // Merge the groups with the current one 
          grpToMergedGrp[j] = curGroup;        
        }
      }
      iNewGroup++;
    // ???? }
    // Go to the next index which have not a group
    /*
    int k;
    for( k = iNewGroup; k < (nGrp+1) && (grpToMergedGrp[k] > 0); k++);
    iNewGroup = k; 
    */
  }
  // vectorPrintShort( "grpToMergedGrp", grpToMergedGrp, nGrp+1);
  //
  // Perform the mapping group -> mergedGroups
  vectorPrintShort( "  grpToMergedGrp", grpToMergedGrp, nGrp+1);
  for (int p=0; p < nPads; p++) {
    padMergedGrp[p] = grpToMergedGrp[ abs( padToGrp[p] ) ];
  }
  // Integrate isolated pads
  /* ???
  Mask_t maskPadWithNoGrp[nPads];
  nPadWithNoGrp = vectorBuildMaskEqualShort( padMergedGrp, 0, nPads, maskPadWithNoGrp) - 1;
  for (int p=0; p < nPads; p++) {
    if ( padMergedGrp[p] == 0 ) {
      for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, p); *neigh_ptr != -1; neigh_ptr++) {
        j = *neigh_ptr; 
        if (padMergedGrp[j] != 0) {
          // propagate the grp
          grpStatusChanged = 1;
        }
          padMergedGrp[p] = padMergedGrp[j]
        }
    }
      getNeighborsOf( neighbours, p);
    }  
  */
  if (CHECK) {
    for (int p=0; p < nPads; p++) {
      if ( padMergedGrp[p] == 0)
        printf("  assignPadsToGroupFromProj: pad %d with no group\n", p);
    }
  }
  //
 
  return newGroupID;  
}

void assignCathPadsToGroupFromProj( short *projPadGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup, short *matGrpGrp) {
  cath0ToGrpFromProj = new short[nCath0];
  cath1ToGrpFromProj = new short[nCath1];
  vectorSetZeroShort( cath0ToGrpFromProj, nCath0);
  vectorSetZeroShort( cath1ToGrpFromProj, nCath1);
  vectorSetShort( wellSplitGroup, 1, nGrp+1);
  vectorSetZeroShort( matGrpGrp, (nGrp+1)*(nGrp+1) );
  //
  PadIdx_t i, j; 
  short g, prevGroup0, prevGroup1;
  for( int k=0; k < nPads; k++) {
    g = projPadGroup[k];
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    if ( i >= 0 ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      prevGroup0 = cath0ToGrpFromProj[i];
      if ( (prevGroup0 == 0) || (prevGroup0 == g) ) {
        // No group before or same group
        cath0ToGrpFromProj[i] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1; 
      } else  {
        // Already a Grp which differs
        if ( prevGroup0 > 0) {
          // Invalid Old group
          wellSplitGroup[ prevGroup0 ] = 0;
          matGrpGrp[ g*(nGrp+1) +  prevGroup0 ] = 1; 
          matGrpGrp[ prevGroup0*(nGrp+1) +  g ] = 1; 
        }
        cath0ToGrpFromProj[i] = -g;
        // Invalid current group
        wellSplitGroup[ g ] = 0;
      }
    }
    //
    // Cathode 1
    //
    if ( j >= 0) {
      // Remark: if j is an alone pad (i<0)
      // j is processed as well
      prevGroup1 = cath1ToGrpFromProj[j];
      if ( (prevGroup1 == 0) || (prevGroup1 == g) ){
         // No group before
         cath1ToGrpFromProj[j] = g;
         matGrpGrp[ g*(nGrp+1) +  g ] = 1; 
      } else {
        // Already a Group 
        if ( prevGroup1 > 0) {
          // Invalid Old group
          wellSplitGroup[ prevGroup1 ] = 0;
          matGrpGrp[ g*(nGrp+1) +  prevGroup1 ] = 1; 
          matGrpGrp[ prevGroup1*(nGrp+1) +  g ] = 1; 
        }
        cath1ToGrpFromProj[j] = -g;
        // Invalid current group
        wellSplitGroup[ g ] = 0;
      }
    }
  }
  if (VERBOSE) printMatrixShort("Group/Group matrix", matGrpGrp, nGrp+1, nGrp+1);
}
int renumberGroups( short *grpToGrp, int nGrp ) {
  // short renumber[nGrp+1];
  // vectorSetShort( renumber, 0, nGrp+1 );
  int maxIdx = vectorMaxShort( grpToGrp, nGrp+1);
  short counters[maxIdx+1];
  vectorSetShort( counters, 0, maxIdx+1 );

  for (int g = 1; g <= nGrp; g++) {
    /*
    if ( grpToGrp[g] != 0) {
      if (renumber[ grpToGrp[g]] == 0 ) {
        // has not be renumbered
        curGrp++;
        // ??? renumber[grpToGrp[g]] = curGrp;      
        renumber[g] = curGrp;
        // grpToGrp[g] = curGrp;
      } else {
        renumber[g] = grpToGrp[g];      
      } 
    */    
    if ( grpToGrp[g] != 0) {
      counters[ grpToGrp[g] ]++;
    }
  }
  int curGrp = 0;
  for (int g = 1; g <= maxIdx; g++) {
    if (counters[g] != 0) {
      curGrp++;
      counters[g] = curGrp;
    }
  }
  // Now counters contains the mapping oldGrp -> newGrp
  // ??? vectorMapShort( grpToGrp, )
  for (int g = 1; g <= nGrp; g++) {
    grpToGrp[g] = counters[ grpToGrp[g]];
  }
  // vectorCopyShort( renumber, nGrp+1, grpToGrp );
  return curGrp;
}

int assignGroupToCathPads( short *projPadGroup, int nPads, int nGrp, int nCath0, int nCath1, short *cath0ToGrp, short *cath1ToGrp) {
  cath0ToGrpFromProj = new short[nCath0];
  cath1ToGrpFromProj = new short[nCath1];
  vectorSetZeroShort( cath0ToGrpFromProj, nCath0);
  vectorSetZeroShort( cath1ToGrpFromProj, nCath1);
  vectorSetZeroShort( cath0ToGrp, nCath0);
  vectorSetZeroShort( cath1ToGrp, nCath1); 
  short projGrpToCathGrp[nGrp+1];
  vectorSetZeroShort( projGrpToCathGrp, nGrp+1); 
  int nCathGrp = 0; // return value, ... avoid a sum ... ????
  //
  printf("assignGroupToCathPads\n");
  //
  PadIdx_t i, j; 
  short g, prevGroup0, prevGroup1;
  if (nCath0 == 0) {
    vectorCopyShort( projPadGroup, nCath1, cath1ToGrp);
    return nGrp;
  } else if (nCath1==0) {
    vectorCopyShort( projPadGroup, nCath0, cath0ToGrp);
    return nGrp;
  }
  for( int k=0; k < nPads; k++) {
    g = projPadGroup[k];
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    if (VERBOSE) {
      printf("map k=%d g=%d to i=%d/%d, j=%d/%d\n", k, g, i, nCath0, j, nCath1);
    }
    //
    // Cathode 0
    //
    if ( (i >= 0) && (nCath0 != 0) ) {
      prevGroup0 = cath0ToGrpFromProj[i];
      if ( (prevGroup0 == 0) ) {
        if( (projGrpToCathGrp[g] == 0 ) && (g !=0 )) {
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
        } 
        cath0ToGrpFromProj[i] = projGrpToCathGrp[g];
      } else if ( prevGroup0 != projGrpToCathGrp[g] ) {
         projGrpToCathGrp[g] = prevGroup0;
      }
    }
    //
    // Cathode 1
    //
    if ( (j >= 0) && (nCath1 != 0) ) {
      prevGroup1 = cath1ToGrpFromProj[j];
      if ( (prevGroup1 == 0) ) {
        if(( projGrpToCathGrp[g] == 0 ) && (g !=0 ) ){
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
        } 
        cath1ToGrpFromProj[j] = projGrpToCathGrp[g];
      } else if ( prevGroup1 != projGrpToCathGrp[g] ) {
         projGrpToCathGrp[g] = prevGroup1;
      }
    }

    /*
    //
    // Cathode 0
    //
    if ( (i >= 0) && (nCath0 != 0) ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      prevGroup0 = cath0ToGrpFromProj[i];
      if ( (prevGroup0 == 0) || (prevGroup0 == g) ) {
        // No group before or same group
        cath0ToGrpFromProj[i] = g;
        if( projGrpToCathGrp[g] == 0 ) {
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
          printf("projGrpToCathGrp[g] = nCathGrp, g=%d,  nCathGrp=%d\n", g, nCathGrp);
          vectorPrintShort("projGrpToCathGrp", projGrpToCathGrp, nGrp+1);
        }
      } else  {
        // Already a Grp which differs
        // Set the same cath-group
        if( projGrpToCathGrp[g] == 0 ) {
          projGrpToCathGrp[g] = projGrpToCathGrp[prevGroup0];
          printf("projGrpToCathGrp[g] = projGrpToCathGrp[prevGroup0], g=%d,  projGrpToCathGrp[prevGroup0]=%d\n", g, projGrpToCathGrp[prevGroup0]);
          vectorPrintShort("projGrpToCathGrp", projGrpToCathGrp, nGrp+1);
        }
      }
    }
    //
    // Cathode 1
    //
    if ( (j >= 0) && (nCath1 != 0) ) {
      // Remark: if j is an alone pad (i<0)
      // j is processed as well
      prevGroup1 = cath1ToGrpFromProj[j];
      if ( (prevGroup1 == 0) || (prevGroup1 == g) ){
        // No group before or same group
        cath1ToGrpFromProj[j] = g;
        if( projGrpToCathGrp[g] == 0 ) {
            nCathGrp++;
            projGrpToCathGrp[g] = nCathGrp;
            printf("cath1 projGrpToCathGrp[g] = nCathGrp, g=%d,  nCathGrp=%d\n", g, nCathGrp);
            vectorPrintShort("projGrpToCathGrp", projGrpToCathGrp, nGrp+1);
        }
      } else  {
        // Already a Grp which differs
        // Set the same cath-group
        projGrpToCathGrp[g] = projGrpToCathGrp[prevGroup1];
        printf("cath1 projGrpToCathGrp[g] = projGrpToCathGrp[prevGroup1], g=%d,  projGrpToCathGrp[prevGroup1]=%d\n", g, projGrpToCathGrp[prevGroup1]);
        vectorPrintShort("projGrpToCathGrp", projGrpToCathGrp, nGrp+1);

      }
     */
  }
  printf("assignGroupToCathPads\n");
  vectorPrintShort( "  cath0ToGrpFromProj ??? ",cath0ToGrpFromProj,nCath0);
  vectorPrintShort( "  cath1ToGrpFromProj ??? ",cath1ToGrpFromProj,nCath1);
  vectorPrintShort( "  projGrpToCathGrp ??? ", projGrpToCathGrp, nGrp+1);
  // Renumering cathodes groups
  // ????
  /*
  short curGrp = 0;
  short renumber[nGrp+1];
  vectorSetShort( renumber, 0, nGrp+1 );
  for (int g = 1; g <= nGrp; g++) {
    if ( (projGrpToCathGrp[g] != 0) && (renumber[ projGrpToCathGrp[g]] == 0) ) {
      curGrp++;
      renumber[ projGrpToCathGrp[g]] = curGrp;      
    }
  }
  vectorPrintShort("Renumber", renumber, nGrp+1);
  vectorMapShort( projGrpToCathGrp, renumber, nGrp+1);
  */
  int nNewGrp = renumberGroups( projGrpToCathGrp, nGrp);
  // nGrp = nNewGrp;
  vectorPrintShort("  projGrpToCathGrp renumbered", projGrpToCathGrp, nGrp+1);  
  //
  vectorMapShort( cath0ToGrpFromProj, projGrpToCathGrp, nCath0);
  vectorCopyShort( cath0ToGrpFromProj, nCath0, cath0ToGrp);
  vectorMapShort( cath1ToGrpFromProj, projGrpToCathGrp, nCath1);
  vectorCopyShort( cath1ToGrpFromProj, nCath1, cath1ToGrp);
  
  for( i=0; i<nPads; i++) {
    projPadGroup[i] = projGrpToCathGrp[ projPadGroup[i] ];
  }
  if (1 || VERBOSE) {
    vectorPrintShort("  projPadGroup", projPadGroup, nPads);
    vectorPrintShort("  cath0ToGrp", cath0ToGrp, nCath0);
    vectorPrintShort("  cath1ToGrp", cath1ToGrp, nCath1);
  }
  return nNewGrp;
}

int assignCathPadsToGroup( short *matGrpGrp, int nGrp, int nCath0, int nCath1, short *grpToGrp ) {
// Merge the ovelaping groups, renumber them 
// and give the mapping OldGrp -> newMergedGroup in grpToGrp array
  cath0ToTGrp = new short[nCath0];
  cath1ToTGrp = new short[nCath1];
  vectorSetZeroShort(grpToGrp, nGrp+1);
  int newGroupID = 0;
  //
  int iNewGroup = 1;
  while ( iNewGroup < (nGrp+1)) {
    // Define the new group
    newGroupID++; 
    grpToGrp[iNewGroup] = newGroupID; 
    // printf("new Group idx=%d, newGroupID=%d\n", iNewGroup, newGroupID);

    for (int i=iNewGroup; i < (nGrp+1); i++) {
      // New Group
      int ishift = i*(nGrp+1);
      for (int j=i+1; j < (nGrp+1); j++) {
        if ( matGrpGrp[ishift+j] ) {
          /*
          if (CHECK) {
            if ( (grpToGrp[j] != 0) && (grpToGrp[j] != i) ) {
              printf("The mapping grpTogrp can't have 2 group values (surjective) oldGrp=%d newGrp=%d\n", grpToGrp[j], i);
              throw std::overflow_error("The mapping grpTogrp can't have 2 group values");
            }
          }
          */
          // Merge the groups 
          grpToGrp[j] = newGroupID;        
        }
      }
    }
    // Go to the next index which have not a group
    int k;
    for( k = iNewGroup; k < (nGrp+1) && (grpToGrp[k] > 0); k++);
    iNewGroup = k; 
  }
  // vectorPrintShort( "grpToGrp", grpToGrp, nGrp+1);
  //
  // Perform the mapping group -> mergedGroups
  for (int c=0; c < nCath0; c++) {
    cath0ToTGrp[c] = grpToGrp[ abs( cath0ToGrpFromProj[c] ) ];
  }
  for (int c=0; c < nCath1; c++) {
    cath1ToTGrp[c] = grpToGrp[ abs( cath1ToGrpFromProj[c] ) ];
  }
  // vectorPrintShort( "cath0ToTGrp", cath0ToTGrp, nCath0);
  // vectorPrintShort( "cath1ToTGrp", cath1ToTGrp, nCath1);
  return newGroupID;
}

void updateProjectionGroups ( short *projPadToGrp, int nProjPads, const short *cath0ToGrp, const short *cath1ToGrp, 
                             const PadIdx_t *mapPadToCathIdx, const short *grpToGrp) {
 
    printf("updateProjectionGroups\n");
    for (int k=0; k < nProjPads; k++) {
         MapKToIJ_t ij = mapKToIJ[k];
         PadIdx_t i = ij.i;
         PadIdx_t j = ij.j;
         if ((i > -1) && (j== -1)) {
           // int cath0Idx = mapPadToCathIdx[ i ];
           projPadToGrp[k] = cath0ToGrp[i];
           // printf("  projPadToGrp[k] = cath0ToGrp[cath0Idx], i=%d, j=%d, cath0Idx=%d, cath0ToGrp[cath0Idx]=%d\n", i, j, cath0Idx, cath0ToGrp[cath0Idx]);
         } else if ((i == -1) && (j > -1)) {
           // int cath1Idx = mapPadToCathIdx[ j ];
           projPadToGrp[k] = cath1ToGrp[j];  
           // printf("  projPadToGrp[k] = cath1ToGrp[cath1Idx], i=%d, j=%d, cath1Idx=%d, cath1ToGrp[cath1Idx]=%d\n", i, j, cath1Idx, cath1ToGrp[cath1Idx]);
         } else if ((i > -1) && (j > -1)) {
           projPadToGrp[k] = grpToGrp[ projPadToGrp[k] ];
           // printf("  projPadToGrp[k] = grpToGrp[ projPadToGrp[k] ], i=%d, j=%d, k=%d \n", i, j, k);
         } else {
           throw std::overflow_error("updateProjectionGroups i,j=-1");
         }
    }
    vectorPrintShort("  updated projGrp", projPadToGrp, nProjPads);

}
int addIsolatedPadInGroups( const double *xyDxy, Mask_t *cathToGrp, int nbrCath, int cath, Mask_t *grpToGrp, int nGroups) {
  PadIdx_t *neigh;
  if (nbrCath == 0) return nGroups;
  
  if (cath == 0 ) {
    neighborsCath0 = getFirstNeighbors( xyDxy, nbrCath, nbrCath);
    neigh = neighborsCath0;
  } else {
    neighborsCath1 = getFirstNeighbors( xyDxy, nbrCath, nbrCath );
    neigh = neighborsCath1;
  }
  printf("addIsolatedPadInGroups cath=%d, nGroups=%d\n", cath, nGroups);

  short newCathToGrp[nbrCath];
  vectorSetShort(newCathToGrp, 0, nbrCath );
  for ( int p=0; p < nbrCath; p++) {
    if( cathToGrp[p] == 0 ) {
      // Neighbors
      //
      int q = -1;
      for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, p); *neigh_ptr != -1; neigh_ptr++) {
        q = *neigh_ptr; 
        if (0 || VERBOSE) printf("  neigh of %d: %d\n", p, q);
        if ( cathToGrp[q] != 0) {
            if ( newCathToGrp[q] == 0 ) {
              // Propagate grp

              newCathToGrp[p] = cathToGrp[q];
              if (0||VERBOSE) printf("    propagate the old grp=%d to the isolated pad p=%d grp\n", cathToGrp[q], p);              
            } else {
              // neighbour already assigned to a group
              // Grp fusion
              Mask_t gMin = std::min( cathToGrp[q], newCathToGrp[p] );
              Mask_t gMax = std::max( cathToGrp[q], newCathToGrp[p] );
              grpToGrp[ gMax ] = gMin; 
              newCathToGrp[p] = gMin;
              if (0||VERBOSE) printf("    Neighbour q=%d  with a newgrp=%d, take this grp for p=%d\n", q, gMin, p);

            }
        } else if ( newCathToGrp[q] != 0 ) {
          // The grp of the neigbour is 0, 
          // cathToGrp[q] == 0,  ie is an "Alone Pad"  
          // But the neighbor pad  has 
          // just been assigned to a group
          if ( newCathToGrp[p] == 0 ) {
              // Propagate grp
              newCathToGrp[p] = cathToGrp[q];
              if (0||VERBOSE) printf("    propagate the new grp=%d of the neighbor to p=%d\n", cathToGrp[q], p);              
          } else {
              // neighbour already assigned to a group
              // Grp fusion
              Mask_t gMin = std::min( newCathToGrp[q], newCathToGrp[p] );
              Mask_t gMax = std::max( cathToGrp[q], newCathToGrp[p] );
              grpToGrp[ gMax ] = gMin; 
              newCathToGrp[p] = gMin;
              if (0||VERBOSE) printf("    Neighbour q=%d with a newgrp=%d, take this grp for p=%d\n", q, gMin, p);
          }
        } else {
          if ( newCathToGrp[p] == 0 ) {
            // The grp of the neigbour is 0, 
            // cathToGrp[q] == 0,  ie is an "Alone Pad"  
            // and  no grp assigne to the neighbours  
            // newCathToGrp[q] == 0
            // Create a new grp
            nGroups++;
            // Take care to not overwrite memeory
            grpToGrp[nGroups] = nGroups;
            newCathToGrp[p] = nGroups;
            newCathToGrp[q] = nGroups;
            if (0||VERBOSE) printf("    No group already assign to p and q pad, create a new group=%d and set it to p=%d, q=%d\n", nGroups, p, q);                        
          } else {
            // Propagate the new p-group to q 
            newCathToGrp[q] = newCathToGrp[p];
            if (0||VERBOSE) printf("    propagate the new grp=%d of p=%d to the neighbor q=%d\n", newCathToGrp[p], p, q);
          }
        }
      }
      if ((newCathToGrp[p] == 0)) {
          // No neighbours with a group
          // New group
          nGroups++;
          grpToGrp[nGroups] = nGroups;
          newCathToGrp[p] = nGroups;
          if (0||VERBOSE) printf("    Create a new group=%d and set it to p=%d\n", nGroups, p);                        

      }
    }
  }
  printf("  ..................................;;;;\n");   
  vectorPrintShort("  cathToGrp", cathToGrp, nbrCath);
  vectorPrintShort("  newCathToGrp", newCathToGrp, nbrCath);
  for( int p=0; p < nbrCath; p++) {
      if( cathToGrp[p] == 0) {
        cathToGrp[p] = newCathToGrp[p];
      }
  }
  vectorPrintShort("  grpToGrp before renumbering", grpToGrp, nGroups+1);
  /*
  int nNewGroups = renumberGroups( grpToGrp, nGroups);
  vectorPrintShort("  grpToGrp", grpToGrp, nGroups+1);
  vectorMapShort( cathToGrp, grpToGrp, nbrCath);
  vectorPrintShort("  cathToGrp", cathToGrp, nbrCath);
  */
  return vectorMaxShort( cathToGrp, nbrCath);
}

void copyCathToGrpFromProj( short *cath0Grp, short *cath1Grp, int nCath0, int nCath1) {
  vectorCopyShort( cath0ToGrpFromProj, nCath0, cath0Grp);
  vectorCopyShort( cath1ToGrpFromProj, nCath1, cath1Grp);
}

void getMaskCathToGrpFromProj( short g, short* mask0, short *mask1, int nCath0, int nCath1) {
    vectorBuildMaskEqualShort( cath0ToGrpFromProj, g, nCath0, mask0);
    vectorBuildMaskEqualShort( cath1ToGrpFromProj, g, nCath1, mask1);
}

void freeMemoryPadProcessing() {
  //
  // Intersection matrix
  if( IInterJ != 0) {
    delete IInterJ;
    IInterJ = 0;
  }
  if( JInterI != 0) {
    delete JInterI;
    JInterI = 0;
  }
  if( intersectionMatrix != 0) {
    delete intersectionMatrix;
    intersectionMatrix = 0;
  }
  // Isolated pads
  if( aloneIPads != 0 ) {
    delete [] aloneIPads;
    aloneIPads = 0;
  }
  if( aloneJPads != 0 ) {
    delete [] aloneJPads;
    aloneJPads = 0;
  }
  if( aloneKPads != 0 ) {
    delete [] aloneKPads;
    aloneKPads = 0;
  }
  //
  // Maps
  if( mapKToIJ != 0) {
    delete mapKToIJ;
    mapKToIJ = 0;
  }
  if( mapIJToK != 0) {
    delete mapIJToK;
    mapIJToK = 0;
  }
  //
  // Neighbors
  if( neighbors != 0) {
    delete neighbors;
    neighbors = 0;
  }
  // Neighbors
  if( neighborsCath0 != 0) {
    delete neighborsCath0;
    neighborsCath0 = 0;
  }
  // Neighbors
  if( neighborsCath1 != 0) {
    delete neighborsCath1;
    neighborsCath1 = 0;
  }
  // Grp Neighbors
  /*
  if( grpNeighborsCath0 != 0) {
    delete grpNeighborsCath0;
    grpNeighborsCath0 = 0;
  }
  if( grpNeighborsCath1 != 0) {
    delete grpNeighborsCath1;
    grpNeighborsCath1 = 0;
  }
  */
  // Projected Pads
  if( projected_xyDxy != 0) {
    delete projected_xyDxy;
    projected_xyDxy = 0;
  }
  // Charge on the projected pads
  if( projCh0 != 0) {
    delete projCh0;
    projCh0 = 0;
  }
  if( projCh1 != 0) {
    delete projCh1;
    projCh1 = 0;
  }
  if( minProj != 0) {
    delete minProj;
    minProj = 0;
  }
  if( maxProj != 0) {
    delete maxProj;
    maxProj = 0;
  }
  if( cath0ToGrpFromProj != 0) {
    delete cath0ToGrpFromProj;
    cath0ToGrpFromProj = 0;
  }
  if( cath1ToGrpFromProj != 0) {
    delete cath1ToGrpFromProj;
    cath1ToGrpFromProj = 0;
  }
  if( cath0ToTGrp != 0) {
    delete cath0ToTGrp;
    cath0ToTGrp = 0;
  }
  if( cath1ToTGrp != 0) {
    delete cath1ToTGrp;
    cath1ToTGrp = 0;
  }

}