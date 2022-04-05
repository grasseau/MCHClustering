// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "MCHClustering/dataStructure.h"
#include "mathUtil.h"
#include "MCHClustering/mathieson.h"

namespace o2
{
namespace mch
{
// Chamber 1, 2
const double sqrtK3x1_2 = 0.7000; // Pitch= 0.21 cm
const double sqrtK3y1_2 = 0.7550; // Pitch= 0.21 cm
const double pitch1_2 = 0.21;
// Chamber 3, 10
const double sqrtK3x3_10 = 0.7131; // Pitch= 0.25 cm
const double sqrtK3y3_10 = 0.7642; // Pitch= 0.25 cm
const double pitch3_10 = 0.25;

int mathiesonType; // 0 for Station 1 or 1 for station 2-5
static double K1x[2], K1y[2];
static double K2x[2], K2y[2];
static const double sqrtK3x[2] = {sqrtK3x1_2, sqrtK3x3_10},
                    sqrtK3y[2] = {sqrtK3y1_2, sqrtK3y3_10};
static double K4x[2], K4y[2];
static double pitch[2] = {pitch1_2, pitch3_10};
static double invPitch[2];
// ??? The Pad Integrals are store here:
// double *I;

void initMathieson()
{
  //
  for (int i = 0; i < 2; i++) {
    K2x[i] = M_PI * 0.5 * (1.0 - sqrtK3x[i] * 0.5);
    K2y[i] = M_PI * 0.5 * (1.0 - sqrtK3y[i] * 0.5);
    K1x[i] = K2x[i] * sqrtK3x[i] * 0.25 / (atan(sqrtK3x[i]));
    K1y[i] = K2y[i] * sqrtK3y[i] * 0.25 / (atan(sqrtK3y[i]));
    K4x[i] = K1x[i] / K2x[i] / sqrtK3x[i];
    K4y[i] = K1y[i] / K2y[i] / sqrtK3y[i];
    invPitch[i] = 1.0 / pitch[i];
  }
}

void compute2DPadIntegrals(const double *xInf, const double *xSup, const double* yInf, const double*ySup,
                           int N, int chamberId, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  //
  if (chamberId <= 2) {
    mathiesonType = 0;
  } else {
    mathiesonType = 1;
  }
  //
  // Select Mathieson coef.
  double curK2x = K2x[mathiesonType];
  double curK2y = K2y[mathiesonType];
  double curSqrtK3x = sqrtK3x[mathiesonType];
  double curSqrtK3y = sqrtK3y[mathiesonType];
  double curK4x = K4x[mathiesonType];
  double curK4y = K4y[mathiesonType];
  double curInvPitch = invPitch[mathiesonType];
  double cst2x = curK2x * curInvPitch;
  double cst2y = curK2y * curInvPitch;
  double cst4 = 4.0 * curK4x * curK4y;
  double uInf, uSup, vInf, vSup;

  for (int i = 0; i < N; i++) {
    // x/u
    uInf = curSqrtK3x * tanh(cst2x * xInf[i]);
    uSup = curSqrtK3x * tanh(cst2x * xSup[i]);
    // y/v
    vInf = curSqrtK3y * tanh(cst2y * yInf[i]);
    vSup = curSqrtK3y * tanh(cst2y * ySup[i]);
    //
    Integrals[i] = cst4 * (atan(uSup) - atan(uInf)) * (atan(vSup) - atan(vInf));
    // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g\n", i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i]);
  }
  // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1], Integrals[N-1]);
  return;
}

void compute1DPadIntegrals( const double *xInf, const double *xSup,
                           int N, int chamberId, bool xAxe, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  //
  if (chamberId <= 2) {
    mathiesonType = 0;
  } else {
    mathiesonType = 1;
  }
  //
  // Select Mathieson coef.
  double curInvPitch = invPitch[mathiesonType];
  double curK2, curSqrtK3, curK4, cst2;
  if (xAxe) {
    curK2 = K2x[mathiesonType];
    curSqrtK3 = sqrtK3x[mathiesonType];
    curK4 = K4x[mathiesonType];
    cst2 = curK2 * curInvPitch;
  } else {
    curK2 = K2y[mathiesonType];
    curSqrtK3 = sqrtK3y[mathiesonType];
    curK4 = K4y[mathiesonType];
    cst2 = curK2 * curInvPitch;
  }
  double cst4 = 2.0 * curK4;

  double uInf, uSup, vInf, vSup;

  for (int i = 0; i < N; i++) {
    // x/u
    uInf = curSqrtK3 * tanh(cst2 * xInf[i]);
    uSup = curSqrtK3 * tanh(cst2 * xSup[i]);
    //
    Integrals[i] = cst4 * (atan(uSup) - atan(uInf));
    // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g\n", i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i]);
  }
  // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1], Integrals[N-1]);
  return;
}

void compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0, const double* theta,
                                           int N, int K, int chamberId, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  // Remarks:
  // - This fct is a cumulative one, as a result it should be set to zero
  //    before calling it
  vectorSetZero(Integrals, N);
  const double* xInf0 = getConstXInf(xyInfSup0, N);
  const double* yInf0 = getConstYInf(xyInfSup0, N);
  const double* xSup0 = getConstXSup(xyInfSup0, N);
  const double* ySup0 = getConstYSup(xyInfSup0, N);
  //
  const double* muX = getConstMuX(theta, K);
  const double* muY = getConstMuY(theta, K);
  const double* w = getConstW(theta, K);

  double z[N];
  double xyInfSup[4 * N];
  double* xInf = getXInf(xyInfSup, N);
  double* yInf = getYInf(xyInfSup, N);
  double* xSup = getXSup(xyInfSup, N);
  double* ySup = getYSup(xyInfSup, N);
  for (int k = 0; k < K; k++) {
    vectorAddScalar(xInf0, -muX[k], N, xInf);
    vectorAddScalar(xSup0, -muX[k], N, xSup);
    vectorAddScalar(yInf0, -muY[k], N, yInf);
    vectorAddScalar(ySup0, -muY[k], N, ySup);
    compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId, z);
    // printf("Vector Sum %g\n", vectorSum(z, N) );
    vectorAddVector(Integrals, w[k], z, N, Integrals);
  }
}

void computeFastCij( const Pads &pads, const Pads &pixel, double Cij[] ) {
    // Compute the Charge Integral Cij of pads (j index), considering the
    // center of the Mathieson fct on a pixel (i index)
    // Use the fact that the charge integral CI(x,y) = CI(x) * CI(y)
    // to reduce the computation cost
    // CI(x) is store in PadIntegralX
    // CI(y) is store in PadIntegralY
    // A subsampling of CI(x_i + k*minDx) (or CI(y_i + l*minDY)) is used
    // by taking the mininimun of pads.dx(pads.dy) to discretize the x/y space
    //
    // CI(x)/CI(y) are computed if they are requested.
    //
    // Returning array: Charge Integral on all the pads Cij[]

    if ((pads.mode != Pads::xyInfSupMode) || (pixel.mode != Pads::xydxdyMode)) {
      printf("Warning: bad representation (mode) of pads in computeCij (padMode=%d, pixelMode=%d)\n", pads.mode, pixel.mode);
      throw std::overflow_error("Bad mode");
      return;
    }
    int N = pads.nPads;
    int K = pixel.nPads;
    // Pads
    int chId = pads.chamberId;
    const double *xInf0 = pads.xInf;
    const double *yInf0 = pads.yInf;
    const double *xSup0 = pads.xSup;
    const double *ySup0 = pads.ySup;
    // Pixels
    const double *muX = pixel.x;
    const double *muY = pixel.y;
    //
    double xPixMin = vectorMin( pixel.x, K);
    double xPixMax = vectorMax( pixel.x, K);
    double yPixMin = vectorMin( pixel.y, K);
    double yPixMax = vectorMax( pixel.y, K);
    double dxMinPix = 2*vectorMin( pixel.dx, K);
    double dyMinPix = 2*vectorMin( pixel.dy, K);
    // Sampling of PadIntegralX/PadIntegralY
    int nXPixels =  (int) ( (xPixMax -  xPixMin) / dxMinPix + 0.5) + 1;
    int nYPixels =  (int) ( (yPixMax -  yPixMin) / dyMinPix + 0.5) + 1;
    printf(" ??? nXPixels nYPixels %d %d \n", nXPixels, nYPixels);
    //
    // PadIntegralX/PadIntegralY allocation and init with -1
    double PadIntegralX[nXPixels][N];
    double PadIntegralY[nYPixels][N];
    vectorSet( (double *) PadIntegralX, -1.0, nXPixels*N );
    vectorSet( (double *) PadIntegralY, -1.0, nYPixels*N );
    double zInf[N];
    double zSup[N];
    bool xAxe;
    /*
    for (int kx=0; kx < nXPixels; kx++) {
      double x = xPixMin + kx * dxPix;
      vectorAddScalar( xInf0, - x, N, zInf );
      vectorAddScalar( xSup0, - x, N, zSup );
      compute1DPadIntegrals( zInf, zSup, N, chId, xAxe, PadIntegralX[kx] );
    }
    xAxe = false;
    for (int ky=0; ky < nYPixels; ky++) {
      double y = yPixMin + ky * dyPix;
      vectorAddScalar( yInf0, - y, N, zInf );
      vectorAddScalar( ySup0, - y, N, zSup );
      compute1DPadIntegrals( zInf, zSup, N, chId, xAxe, PadIntegralY[ky] );
    }
    */

    // Loop on Pixels
    for (int k=0; k < K; k++) {
      // Calculate the indexes in the 1D charge integral
      // PadIntegralX:PadIntegralY
      int xIdx = (int) ((muX[k] - xPixMin) / dxMinPix + 0.5 );
      int yIdx = (int) ((muY[k] - yPixMin) / dyMinPix + 0.5 );
      // compute2DPadIntegrals( xInf, xSup, yInf, ySup, N, chId, &Cij[N*k] );
      // Cij[ N*k + p] = PadIntegralX( k, xIdx) * PadIntegralY( k, yIdx);
      // printf("k=%d, mu[k]=(%f, %f) Sum_pads Ck = %g\n", k, muX[k], muY[k], vectorSum( &Cij[N*k], N) );
      if ( PadIntegralX[xIdx][0] == -1 ) {
        // Not yet computed
        vectorAddScalar( xInf0, - muX[k], N, zInf );
        vectorAddScalar( xSup0, - muX[k], N, zSup );
        xAxe = true;
        compute1DPadIntegrals( zInf, zSup, N, chId, xAxe, PadIntegralX[xIdx] );
      }
      if ( PadIntegralY[yIdx][0] == -1 ) {
        // Not yet computed
        vectorAddScalar( yInf0, - muY[k], N, zInf );
        vectorAddScalar( ySup0, - muY[k], N, zSup );
        xAxe = false;
        compute1DPadIntegrals( zInf, zSup, N, chId, xAxe, PadIntegralY[yIdx] );
      }
      // Compute IC(xy) = IC(x) * IC(y)
      vectorMultVector( PadIntegralX[xIdx], PadIntegralY[yIdx], N,  &Cij[N*k]);
    }
}

void computeCij( const Pads &pads, const Pads &pixel, double Cij[] ) {
    // Compute the Charge Integral Cij of pads (j index), considering the
    // center of the Mathieson fct on a pixel (i index)
    //
    // Returning array: Charge Integral on all the pads Cij[]

    if ((pads.mode != Pads::xyInfSupMode) || (pixel.mode != Pads::xydxdyMode)) {
      printf("Warning: bad representation (mode) of pads in computeCij (padMode=%d, pixelMode=%d)\n", pads.mode, pixel.mode);
      throw std::overflow_error("Bad mode");
      return;
    }
    int N = pads.nPads;
    int K = pixel.nPads;
    int chId = pads.chamberId;
    const double *xInf0 = pads.xInf;
    const double *yInf0 = pads.yInf;
    const double *xSup0 = pads.xSup;
    const double *ySup0 = pads.ySup;

    //
    const double *muX = pixel.x;
    const double *muY = pixel.y;

    double xInf[N];
    double yInf[N];
    double xSup[N];
    double ySup[N];

    for (int k=0; k < K; k++) {
      vectorAddScalar( xInf0, - muX[k], N, xInf );
      vectorAddScalar( xSup0, - muX[k], N, xSup );
      vectorAddScalar( yInf0, - muY[k], N, yInf );
      vectorAddScalar( ySup0, - muY[k], N, ySup );
      compute2DPadIntegrals( xInf, xSup, yInf, ySup, N, chId, &Cij[N*k] );
      // printf("k=%d, mu[k]=(%f, %f) Sum_pads Ck = %g\n", k, muX[k], muY[k], vectorSum( &Cij[N*k], N) );
    }
}
} // namespace mch
} // namespace o2

// C Wrapper
void o2_mch_initMathieson() {
    o2::mch::initMathieson();
}

void o2_mch_compute2DPadIntegrals(const double* xInf, const double* xSup, const double* yInf, const double* ySup,
                           int N, int chamberId,
                           double Integrals[]) {
    o2::mch::compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId, Integrals);
}

void o2_mch_computeCij( const double *xyInfSup0, const double *pixel,
                            int N, int K, int chamberId, double Cij[] ) {
    // Returning array: Charge Integral on all the pads
    // Remarks:
    // - This fct is a cumulative one, as a result it should be set to zero
    //    before calling it
    const double *xInf0 = o2::mch::getConstXInf( xyInfSup0, N);
    const double *yInf0 = o2::mch::getConstYInf( xyInfSup0, N);
    const double *xSup0 = o2::mch::getConstXSup( xyInfSup0, N);
    const double *ySup0 = o2::mch::getConstYSup( xyInfSup0, N);
    //
    const double *muX = o2::mch::getConstMuX( pixel, K);
    const double *muY = o2::mch::getConstMuY( pixel, K);
    const double *w = o2::mch::getConstW( pixel, K);

    double z[N];
    double xyInfSup[4*N];
    double *xInf = o2::mch::getXInf( xyInfSup, N);
    double *yInf = o2::mch::getYInf( xyInfSup, N);
    double *xSup = o2::mch::getXSup( xyInfSup, N);
    double *ySup = o2::mch::getYSup( xyInfSup, N);
    for (int k=0; k < K; k++) {
      o2::mch::vectorAddScalar( xInf0, - muX[k], N, xInf );
      o2::mch::vectorAddScalar( xSup0, - muX[k], N, xSup );
      o2::mch::vectorAddScalar( yInf0, - muY[k], N, yInf );
      o2::mch::vectorAddScalar( ySup0, - muY[k], N, ySup );
      o2_mch_compute2DPadIntegrals( xInf, xSup, yInf, ySup, N, chamberId, &Cij[N*k] );
      // printf("Vector Sum %g\n", vectorSum(z, N) );
      // ??? vectorMultScalar( &Cij[N*k], w[k], N,  &Cij[N*k]);
    }
}

void o2_mch_compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0, const double* theta,
                                           int N, int K, int chamberId,
                                           double Integrals[]) {
  o2::mch::compute2DMathiesonMixturePadIntegrals(xyInfSup0, theta,
                                           N, K, chamberId, Integrals);
}
