// Unused ???
/*
void Cluster::prepareDataForFitting0( Mask_t* maskFit[2],
        double *xyDxyFit, double *qFit, Mask_t *cathFit, Mask_t *notSaturatedFit, double *zCathTotalCharge, int nFits[2]) {
    int nFit = nFits[0] + nFits[1];
    int nbrCath0 = pads[0]->nPads;
    int nbrCath1 = pads[1]->nPads;
    int nbrCaths[2] = { nbrCath0, nbrCath1};
    int n0 = nFits[0]; int n1 = nFits[1];
    zCathTotalCharge[0] = 0; zCathTotalCharge[1] = 0;
    if (nbrOfCathodePlanes == 2) {
        // Build xyDxyFit  in group g
        //
        // Extract from cath0 the pads which belong to the group g
        maskedCopyToXYdXY( *pads[0], maskFit[0], nbrCath0, xyDxyFit, nFit );
        // Extract from cath1 the pads which belong to the group g
        maskedCopyToXYdXY( *pads[1], maskFit[1], nbrCath1, &xyDxyFit[n0], nFit );
        // Saturated pads
        // vectorPrintShort(" notSaturatedFit 0 ??? ", notSaturatedFit, nFit);
        vectorGatherShort( pads[0]->saturate, maskFit[0], nbrCath0, &notSaturatedFit[0]);
        vectorGatherShort( pads[1]->saturate, maskFit[1], nbrCath1, &notSaturatedFit[n0]);
        // vectorPrintShort(" notSaturatedFit 0 ??? ", notSaturatedFit, nFit);
        vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
        // vectorPrintShort(" MaskFit0 ??? ", maskFit0, nbrCath0);
        // vectorPrintShort(" MaskFit1 ??? ", maskFit1, nbrCath1);
        // vectorPrintShort(" saturatedFit ??? ", saturated, nFit);
        // vectorPrintShort(" notSaturatedFit ??? ", notSaturatedFit, nFit);
        // Chargei in group g
        vectorGather( pads[0]->q, maskFit[0], nbrCath0, qFit);
        vectorGather( pads[1]->q, maskFit[1], nbrCath1, &qFit[n0]);
        // saturated pads are ignored
        // ??? Don't Set to zero the sat. pads
        // vectorMaskedMult( zFit, notSaturatedFit, nFit, zFit);
        // Total Charge on both cathodes
        zCathTotalCharge[0] = vectorMaskedSum( qFit,      &notSaturatedFit[0], n0);
        zCathTotalCharge[1] = vectorMaskedSum( &qFit[n0], &notSaturatedFit[n0], n1);
        // Merge the 2 Cathodes
        vectorSetShort( cathFit,      0, nFit);
        vectorSetShort( &cathFit[n0], 1, n1);
    } else {
        // In that case: there are only one cathode
        // It is assumed that there is no subcluster
        //
        // Extract from  all pads which belong to the group g
        // printf("??? nPads, nFit, n0, n1 %d %d %d %d\n", nPads, nFit, n0, n1);
        // vectorPrintShort( "maskFit0", maskFit0, nbrCath0);
        // vectorPrintShort( "maskFit1", maskFit1, nbrCath1);

        //
        // GG ??? Maybe to shrink with the 2 cathodes processing
        // Total Charge on cathodes & cathode mask
        maskedCopyToXYdXY( *pads[singleCathPlaneID], maskFit[singleCathPlaneID], nbrCaths[singleCathPlaneID], xyDxyFit, nFit );
        vectorGatherShort( pads[singleCathPlaneID]->saturate, maskFit[singleCathPlaneID], nbrCaths[singleCathPlaneID], &notSaturatedFit[0]);
        vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
        vectorGather( pads[singleCathPlaneID]->q, maskFit[singleCathPlaneID], nbrCaths[singleCathPlaneID], qFit);
        zCathTotalCharge[singleCathPlaneID] = vectorMaskedSum( qFit, notSaturatedFit, nFit);
        vectorSetShort(cathFit, singleCathPlaneID, nFit);

        // Don't take into account saturated pads
        vectorMaskedMult(  qFit, notSaturatedFit, nFit, qFit);
    }
}
*/

//
void Cluster::maskedCopyToXYdXY(const Pads &pads, const Mask_t* mask, int nMask,
                     double* xyDxyMasked, int nxyDxyMasked)
{
  /* ?? Inv
  const double* X = getConstX(xyDxy, nxyDxy);
  const double* Y = getConstY(xyDxy, nxyDxy);
  const double* DX = getConstDX(xyDxy, nxyDxy);
  const double* DY = getConstDY(xyDxy, nxyDxy);
  */
  const double* X = pads.getX();
  const double* Y = pads.getY();
  const double* DX = pads.getDX();
  const double* DY = pads.getDY();
  double* Xm = getX(xyDxyMasked, nxyDxyMasked);
  double* Ym = getY(xyDxyMasked, nxyDxyMasked);
  double* DXm = getDX(xyDxyMasked, nxyDxyMasked);
  double* DYm = getDY(xyDxyMasked, nxyDxyMasked);
  vectorGather(X, mask, nMask, Xm);
  vectorGather(Y, mask, nMask, Ym);
  vectorGather(DX, mask, nMask, DXm);
  vectorGather(DY, mask, nMask, DYm);
}

// ??? To remove - unused
double *Cluster::getProjPadsAsXYdXY( Groups_t group, const Mask_t* maskGrp, int nbrProjPadsInTheGroup) {
    double *xyDxyGProj = new double[nbrProjPadsInTheGroup*4];
    // double *qGProj = new double[nbrProjPadsInTheGroup*4];

    maskedCopyToXYdXY( *projectedPads, maskGrp, projectedPads->getNbrOfPads(), xyDxyGProj, nbrProjPadsInTheGroup );
    //maskedCopy(qProject)
    return xyDxyGProj;
}

// ??? To integrate in Pads or else where
Pads *Pads::addBoundaryPads( const double *x_, const double *y_, const double *dx_, const double *dy_, const double *q_, const Mask_t *cath_, const Mask_t *sat_, int chamberId, int N) {

  // TODO: Remove duplicate pads
  double eps = 1.0e-4;
  //
  std::vector<double> bX;
  std::vector<double> bY;
  std::vector<double> bdX;
  std::vector<double> bdY;
  std::vector<int> bCath;
  int n1 = vectorSumShort(cath_, N);
  int nPads_[2] = {N-n1, n1};

  for (int c=0; c < 2; c++) {
    int nc = nPads_[c];
    Mask_t mask[N];
    double x[nc], y[nc], dx[nc], dy[nc], q[nc], sat[nc];
    vectorBuildMaskEqualShort( cath_, c, N, mask);
    vectorGather(x_, mask, N, x);
    vectorGather(y_, mask, N, y);
    vectorGather(dx_, mask, N, dx);
    vectorGather(dy_, mask, N, dy);

    PadIdx_t *neighC = Pads::buildFirstNeighbors( x, y, dx, dy, nc, VERBOSE );

    for (int i=0; i < nc; i++) {
      bool east = true, west = true, north = true, south = true;
      for( const PadIdx_t *neigh_ptr = getNeighborListOf(neighC, i); *neigh_ptr != -1; neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        double xDelta = (x[v] - x[i]);
        if (fabs(xDelta) > eps) {
          if (xDelta > 0) {
            east = false;
          } else {
            west = false;
          }
        }
        double yDelta = (y[v] - y[i]);
        if (fabs(yDelta) > eps) {
          if (yDelta > 0) {
            north = false;
          } else {
            south = false;
          }
        }
      }
      // Add new pads
      if ( east ) {
        bX.push_back( x[i]+2*dx[i] );
        bY.push_back( y[i]);
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
      if ( west ) {
        bX.push_back( x[i]-2*dx[i] );
        bY.push_back( y[i]);
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
      if (north) {
        bX.push_back( x[i] );
        bY.push_back( y[i]+2*dy[i] );
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
      if (south) {
        bX.push_back( x[i] );
        bY.push_back( y[i]-2*dy[i] );
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
    }
    delete neighC;
  }

  int nPadToAdd = bX.size();
  // ??? int nTotalPads = nPadToAdd + N;
  int nTotalPads =  N+nPadToAdd;
  if (VERBOSE > 2) {
      printf("nTotalPads=%d, nPads=%d,  nPadToAdd=%d\n", nTotalPads, N, nPadToAdd);
  }
  o2::mch::Pads *padsWithBoundaries = new o2::mch::Pads( nTotalPads, chamberId );
  o2::mch::Pads *newPads = padsWithBoundaries;
  for (int i=0; i < N; i++) {
    newPads->x[ i ] = x_[i];
    newPads->y[ i ] = y_[i];
    newPads->dx[ i ] = dx_[i];
    newPads->dy[ i ] = dy_[i];
    newPads->q[ i ] = q_[i];
    newPads->cath[ i ] = cath_[i];
    newPads->saturate[ i ] = sat_[i];
  }
  for (int i=N, k=0; i < nTotalPads; i++,k++) {
      newPads->x[ i ] = bX[k];
      newPads->y[ i ] = bY[k];
      newPads->dx[ i ] = bdX[k];
      newPads->dy[ i ] = bdY[k];
      newPads->cath[ i ] = bCath[k];

      newPads->q[ i ] = 0.0;
      // Not saturated
      newPads->saturate[ i ] = 0;
    }
    //
    return padsWithBoundaries;
}


void fitMathieson0(double* thetai,
                  double* xyDxy, double* z, Mask_t* cath, Mask_t* notSaturated,
                  double* zCathTotalCharge,
                  int KMax, int N, int chamberId, int process,
                  double* thetaf,
                  double* khi2,
                  double* pError)
{
  int status;

  // process
  int p = process;
  int verbose = p & 0x3;
  p = p >> 2;
  int doJacobian = p & 0x1;
  p = p >> 1;
  int computeKhi2 = p & 0x1;
  p = p >> 1;
  int computeStdDev = p & 0x1;
  if (verbose) {
    printf("Fitting \n");
    printf("  mode: verbose, doJacobian, computeKhi2, computeStdDev %d %d %d %d\n", verbose, doJacobian, computeKhi2, computeStdDev);
  }
  //
  double* muAndWi = getMuAndW(thetai, KMax);
  //
  // Check if fitting is possible
  double* muAndWf = getMuAndW(thetaf, KMax);
  if (3 * KMax - 1 > N) {
    muAndWf[0] = NAN;
    muAndWf[KMax] = NAN;
    muAndWf[2 * KMax] = NAN;
    return;
  }


  funcDescription_t mathiesonData;
  double cathMax[2] = { 0.0, 0.0};
  double *cathWeights;
  o2::mch::Pads *pads = nullptr;


  // Add boundary Pads
  pads = Pads::addBoundaryPads( getX(xyDxy, N), getY(xyDxy, N), getDX(xyDxy, N), getDY(xyDxy, N),
          z, cath, notSaturated, chamberId, N);
  // inspectSavePixels( 3, *pads);
  N = pads->getNbrOfPads();
    // Function description (extra data nor parameters)
  mathiesonData.N = N;
  mathiesonData.K = KMax;
  mathiesonData.x_ptr = pads->getX();
  mathiesonData.y_ptr = pads->getY();
  mathiesonData.dx_ptr = pads->getDX();
  mathiesonData.dy_ptr = pads->getDY();
  mathiesonData.cath_ptr = pads->getCathodes();
  mathiesonData.zObs_ptr = pads->getCharges();
  // ??? mathiesonData.notSaturated_ptr = pads->getSaturates();
    // Init the weights
  cathWeights = new double[N];
  const Mask_t *sat = pads->getSaturates();
  const double *c = pads->getCharges();
  for (int i = 0; i < N; i++) {
    cathWeights[i] = (cath[i] == 0) ? zCathTotalCharge[0] : zCathTotalCharge[1];
    cathMax[cath[i]] = fmax( cathMax[cath[i]], sat[i] * c[i] );
  }
  /* Invalid before add boudbary pads
  // Function description (extra data nor parameters)
  mathiesonData.N = N;
  mathiesonData.K = KMax;
  mathiesonData.x_ptr = getX(xyDxy, N);
  mathiesonData.y_ptr = getY(xyDxy, N);
  mathiesonData.dx_ptr = getDX(xyDxy, N);
  mathiesonData.dy_ptr = getDY(xyDxy, N);
  mathiesonData.cath_ptr = cath;
  mathiesonData.zObs_ptr = z;
  mathiesonData.notSaturated_ptr = notSaturated;
  // Init the weights
  cathWeights = new double[N];
  for (int i = 0; i < N; i++) {
    cathWeights[i] = (cath[i] == 0) ? zCathTotalCharge[0] : zCathTotalCharge[1];
    cathMax[cath[i]] = fmax( cathMax[cath[i]], notSaturated[i]*z[i] );
  }
  }
  */

  mathiesonData.cathWeights_ptr = cathWeights;
  mathiesonData.cathMax_ptr = cathMax;
  mathiesonData.chamberId = chamberId;
  mathiesonData.zCathTotalCharge_ptr = zCathTotalCharge;
  mathiesonData.verbose = verbose;
  //
  // Define Function, jacobian
  gsl_multifit_function_fdf f;
  f.f = &f_ChargeIntegral;
  f.df = nullptr;
  f.fdf = nullptr;
  f.n = N;
  f.p = 3 * KMax - 1;
  f.params = &mathiesonData;

  bool doFit = true;
  // K test
  int K = KMax;
  // Sort w
  int maxIndex[KMax];
  for( int k=0; k<KMax; k++) { maxIndex[k]=k; }
  double *w = &muAndWi[2 * KMax];
  std::sort( maxIndex, &maxIndex[KMax], [=](int a, int b){ return(w[a] > w[b]); });

  while (doFit) {
    // Select the best K's
    // Copy kTest max
    double muAndWTest [3*K];
    // Mu part
    for (int k=0; k < K; k++) {
      // Respecttively mux, muy, w
      muAndWTest[k] = muAndWi[maxIndex[k]];
      muAndWTest[k+K] = muAndWi[maxIndex[k]+KMax];
      muAndWTest[k+2*K] = muAndWi[maxIndex[k]+2*KMax];
    }
    if( verbose > 0) {
      vectorPrint( "  Selected w", &muAndWTest[2*K], K);
      vectorPrint( "  Selected mux", &muAndWTest[0], K);
      vectorPrint( "  Selected muy", &muAndWTest[K], K);
    }
    mathiesonData.K = K;
    f.p = 3 * K - 1;
    // Set initial parameters
    // Inv ??? gsl_vector_view params0 = gsl_vector_view_array(muAndWi, 3 * K - 1);
    gsl_vector_view params0 = gsl_vector_view_array( muAndWTest, 3 * K - 1);

    // Fitting method
    gsl_multifit_fdfsolver* s = gsl_multifit_fdfsolver_alloc(gsl_multifit_fdfsolver_lmsder, N, 3 * K - 1);
    // associate the fitting mode, the function, and the starting parameters
    gsl_multifit_fdfsolver_set(s, &f, &params0.vector);

    if (verbose > 1) {
      printState(-1, s, K);
    }
    // double initialResidual = gsl_blas_dnrm2(s->f);
    double initialResidual = 0.0;
    // Fitting iteration
    status = GSL_CONTINUE;
    double residual = DBL_MAX;;
    double prevResidual = DBL_MAX;;
    double prevTheta[3*K-1];
    // ??? for (int iter = 0; (status == GSL_CONTINUE) && (iter < 500); iter++) {
    for (int iter = 0; (status == GSL_CONTINUE) && (iter < 50); iter++) {
      // TODO: to speed if possible
      for (int k = 0; k < (3 * K - 1); k++) {
        prevTheta[k] = gsl_vector_get(s->x, k);
      }
      // printf("  Debug Fitting iter=%3d |f(x)|=%g\n", iter, gsl_blas_dnrm2(s->f));
      status = gsl_multifit_fdfsolver_iterate(s);
      if (verbose > 1) {
        printf("  Solver status = %s\n", gsl_strerror(status));
      }
      if (verbose > 0) {
        printState(iter, s, K);
      }
      /* ???? Inv
      if (status) {
        printf("  ???? End fitting \n");
        break;
      };
      */
      // GG TODO ???: adjust error in fct of charge
      status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
      if (verbose > 1) {
        printf("  Status multifit_test_delta = %d %s\n", status, gsl_strerror(status));
      }
      // Residu
      prevResidual = residual;
      residual = gsl_blas_dnrm2(s->f);
      // vectorPrint(" prevtheta", prevTheta, 3*K-1);
      // vectorPrint(" theta", s->dx->data, 3*K-1);
      // printf(" prevResidual, residual %f %f\n", prevResidual, residual );
      if (fabs(prevResidual - residual) < 1.0e-2) {
        // Stop iteration
        // Take the previous value of theta
        if ( verbose > 0) {
          printf("  Stop iteration (dResidu~0), prevResidual=%f residual=%f\n", prevResidual, residual );
        }
        for (int k = 0; k < (3 * K - 1); k++) {
          gsl_vector_set(s->x, k, prevTheta[k]);
        }
        status = GSL_SUCCESS;
      }
    }
    double finalResidual=gsl_blas_dnrm2(s->f);
    bool keepInitialTheta = fabs(finalResidual - initialResidual) / initialResidual < 1.0e-1;

          // Khi2
      if (computeKhi2 && (khi2 != nullptr)) {
        // Khi2
        double chi = gsl_blas_dnrm2(s->f);
        double dof = N - (3 * K - 1);
        double c = fmax(1.0, chi / sqrt(dof));
        if (verbose > 0) {
          printf("K=%d, chi=%f, chisq/dof = %g\n",  K, chi*chi, chi*chi / dof);
        }
        khi2[0] = chi * chi / dof;
      }

    // ???? if (keepInitialTheta) {
    if (0) {
      // Keep the result of EM (GSL bug when no improvemebt)
      copyTheta( thetai, K, thetaf, K, K);
    } else {
      // Fitted parameters
      /* Invalid ???
      for (int k = 0; k < (3 * K - 1); k++) {
        muAndWf[k] = gsl_vector_get(s->x, k);
      }
      */

      // Mu part
      for (int k=0; k < K; k++) {
        muAndWf[k] = gsl_vector_get(s->x, k);
        muAndWf[k+KMax] = gsl_vector_get(s->x, k+K);
      }
      // w part
      double sumW = 0;
      for (int k=0; k < K - 1; k++) {
        double w = gsl_vector_get(s->x, k+2*K);
        sumW += w;
        muAndWf[k+2*KMax] = w;
      }
      // Last w : 1.0 - sumW
      muAndWf[3 * KMax - 1] = 1.0 - sumW;



      // Parameter error
      if (computeStdDev && (pError != nullptr)) { //
        // Covariance matrix an error
        gsl_matrix* covar = gsl_matrix_alloc(3 * K - 1, 3 * K - 1);
        gsl_multifit_covar(s->J, 0.0, covar);
        for (int k = 0; k < (3 * K - 1); k++) {
          pError[k] = sqrt(gsl_matrix_get(covar, k, k));
        }
        gsl_matrix_free(covar);
      }
    }
    if (verbose >= 2) {
      printf("  status parameter error = %s\n", gsl_strerror(status));
    }
    gsl_multifit_fdfsolver_free(s);
    K = K-1;
    // doFit = (K < 3) && (K > 0);
    doFit = false;
  } // while(doFit)
  // Release memory
  delete [] cathWeights;
  if (pads != nullptr) delete pads;
  //
  return;
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

void getIndexInPadProjGrp( const Mask_t *maskThetaGrp, const int *thetaPadProjIdx, const int *mapProjIdxToProjGrpIdx,
                           int KProj, int *thetaPadProjGrpIdx ) {
// maskThetaGrp    : Mask of projPads belonging to the current group
// thetaPadProjIdx : index in projPad array of a seed theta[k]
//                   In other words map a seed theta[k] on a projPad
// mapProjIdxToProjGrpIdx : map projPads idx to projPadGrp (subset projPads of the group)
// KProj : nbr of projPads
// thetaPadProjGrpIdx : index in projPadGroup array of a seed theta[k]
//                      In other words map a seed theta[k] on a projPadGroup
  int ii =0;
  for (int kk=0; kk < KProj; kk++) {
    if (maskThetaGrp[kk]) {
        // A max (or seed) belonging to the current grp
        // Get the pad-index in the whole set of projPad
        int idxProj = thetaPadProjIdx[kk];
        // Get the location in padProjGrp set
        // get the order projPadGrp
        int uu = mapProjIdxToProjGrpIdx[idxProj];
        if (uu < 0) {
          printf("Index in projPad %d padIdxInGrp %d \n", idxProj, uu);
          printf("#### Bad index %d %d\n", idxProj, kk);
          throw std::overflow_error("Bad Allocation");
        }
        thetaPadProjGrpIdx[ii] = uu;
        ii++;
    }
  }
}

void computeMathiesonResidual( const double *xyDxy, const Mask_t *cath, const double *zObs, const double *theta, int chId, int K, int N, double *residual) {

  // GG duplicated code with EM module ???
  // define x, y, dx, dy description
  const double *x  = getConstX( xyDxy, N);
  const double *y  = getConstY( xyDxy, N);
  const double *dX = getConstDX( xyDxy, N);
  const double *dY = getConstDY( xyDxy, N);

  // Compute boundary of each pads
  double xyInfSup[4*N];
  vectorAddVector( x, -1.0, dX, N, getXInf(xyInfSup, N) );
  vectorAddVector( y, -1.0, dY, N, getYInf(xyInfSup, N) );
  vectorAddVector( x, +1.0, dX, N, getXSup(xyInfSup, N) );
  vectorAddVector( y, +1.0, dY, N, getYSup(xyInfSup, N) );
  compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, N, K, chId, residual );
  // generateMixedGaussians2D( xyInfSup, theta, K, N, residual);
  Mask_t cath0[N];
  vectorNotShort( cath, N, cath0);
  double sumCh0 = vectorMaskedSum( zObs, cath0, N);
  double sumCh1 = vectorMaskedSum( zObs, cath, N);
  vectorMaskedMultScalar( residual, cath0, sumCh0, sumCh1, N);
  vectorAddVector( zObs, -1.0, residual, N, residual);
 }

/* Not Used ???
void finalizeSolution( &seedList ) {
  if ( seedList.size() == 1) return;

  double sumCharge = 0;
  for (int g=0;  < seedList.size(); g++) {
    int k = seedList[g].first;
    double theta = getW( seedList[g].second, k);
    double *w = getW(theta, k )
    for( int l=0; l<k; l++) {
      // w[k] contains the charge
      sumCharge += w[l];
    }
  }
  // Filter on relative weight (Charge)
  for (int g=0;  < seedList.size(); g++) {
    int k = seedList[g].first;
    double theta = getW( seedList[g].second, k);
    double *w = getW(theta, k )
    for( int l=0; l<k; l++) {
      // w[k] contains the charge
      w[l] = w[l]/sumCharge;
    }
    // Buid Mask
    vectorBuidMaskGreater( w, 0)
  }
  maskedCopyTheta(const double* theta, int K, const Mask_t* mask, int nMask, double* maskedTheta, int maskedK)
}
*/

void cleanClusterProcessVariables( ) {
    // To verify the alloc/dealloc ????
    // if (uniqueCath == -1) {
      // Two cathodes case
      // if ( xy0Dxy != 0 ) { delete[] xy0Dxy; xy0Dxy = 0; }
      // if ( xy1Dxy != 0 ) { delete[] xy1Dxy; xy1Dxy = 0; }
      // if ( ch0 != 0) { delete[] ch0; ch0 = 0; }
      // if ( ch1 != 0) { delete[] ch1; ch1 = 0; }
      // if ( satPads0 != 0) { delete[] satPads0; satPads0 = 0; }
      // if ( satPads1 != 0) { delete[] satPads1; satPads1 = 0; }
      // if ( cath0ToPadIdx != 0) { delete[] cath0ToPadIdx; cath0ToPadIdx = 0; }
      // if ( cath1ToPadIdx != 0) { delete[] cath1ToPadIdx; cath1ToPadIdx = 0; }
    // }
    // if ( xyDxyProj != 0 ) { delete[] xyDxyProj; xyDxyProj =0; }
    // if ( chProj != 0 ) { delete[] chProj; chProj = 0; }
    // if ( saturatedProj != 0 ) { delete[] saturatedProj; saturatedProj = 0; }
    // if ( wellSplitGroup != 0) { delete[] wellSplitGroup; wellSplitGroup=0; };
    // if ( padToMergedGrp != 0) { delete[] padToMergedGrp; padToMergedGrp=0; };
    //deleteDouble( xyDxyGrp );
    // deleteDouble( chGrp );
    nMergedGrp = 0;
  }

bool orderFct(double i, double j) { return i<j; }
