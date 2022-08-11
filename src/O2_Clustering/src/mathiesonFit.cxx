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
#include <gsl/gsl_multifit_nlin.h>

#include "MCHClustering/PadsPEM.h"
#include "mathUtil.h"
#include "mathieson.h"
#include "mathiesonFit.h"

using namespace o2::mch;

double chargeNormalization( const Mask_t* cath, const Mask_t* notSaturated, const double* cathMaxObs, int N, double* z, double* coefNorm) {
  double zMax[2] = {0, 0};
  for (int i = 0; i < N; i++) {
    zMax[cath[i]] = std::fmax(zMax[cath[i]], notSaturated[i] * z[i]);
  }
  // Avoid dividing by 0
  for (int c = 0; c < 2; c++) {
    if (zMax[c] < 1.0e-6) {
      // In this case cathMax[c] must be 0
      zMax[c] = 1.0;
    }
  }
  //
  // Normalization coefficient
  //
  // Use the max charge cathode for each cathode
  coefNorm[0] = cathMaxObs[0] / zMax[0];
  coefNorm[1] = cathMaxObs[1] / zMax[1];
  // Perform the normalization
  for (int i = 0; i < N; i++) {
    z[i] = z[i] * coefNorm[cath[i]];
    // To have traces about the fitting
    // chargePerCath[cath[i]] += z[i];
  }
  // Use to weight the penalization
  double meanCoef = (coefNorm[0] + coefNorm[1]) /
                    ((coefNorm[0] > 1.0e-6) + (coefNorm[1] > 1.0e-6));

  if (ClusterConfig::fittingLog >= ClusterConfig::debug) {
    printf(
      "    Max of unsaturated (observed) pads (cathMax0/1)= %f, %f, "
      "maxThZ (computed)  %f, %f\n",
      cathMaxObs[0], cathMaxObs[1], zMax[0], zMax[1]);
  }

  return  meanCoef;
}

int f_ChargeIntegral(const gsl_vector* gslParams, void* dataFit,
                     gsl_vector* residuals)
{
  funcDescription_t* dataPtr = (funcDescription_t*)dataFit;
  int N = dataPtr->N;
  int K = dataPtr->K;
  const double* x = dataPtr->x_ptr;
  const double* y = dataPtr->y_ptr;
  const double* dx = dataPtr->dx_ptr;
  const double* dy = dataPtr->dy_ptr;
  const Mask_t* cath = dataPtr->cath_ptr;
  const double* zObs = dataPtr->zObs_ptr;
  Mask_t* notSaturated = dataPtr->notSaturated_ptr;
  int chamberId = dataPtr->chamberId;
  double* cathWeights = dataPtr->cathWeights_ptr;
  double* cathMax = dataPtr->cathMax_ptr;
  double* zCathTotalCharge = dataPtr->zCathTotalCharge_ptr;
  double* cathCoefNorm = dataPtr->cathCoefNorm_ptr;
  // ??? int verbose = dataPtr->verbose;
  // Parameters
  const double* params = gsl_vector_const_ptr(gslParams, 0);
  // Note:
  //  mux = mu[0:K-1]
  //  muy = mu[K:2K-1]
  const double* mu = &params[0];
  double* w = (double*)&params[2 * K];

  // Set constrain: sum_(w_k) = 1
  double lastW = 1.0 - vectorSum(w, K - 1);
  //
  // Display paramameters (w, mu_x, mu_x
  if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
    printf("  Function evaluation at:\n");
    for (int k = 0; k < K; k++) {
      printf("    mu_k[%d] = %g %g \n", k, mu[k], mu[K + k]);
    }
    for (int k = 0; k < K - 1; k++) {
      printf("    w_k[%d] = %g \n", k, w[k]);
    }
    // Last W
    printf("    w_k[%d] = %g \n", K - 1, lastW);
  }

  // Charge Integral on Pads
  double z[N];
  vectorSetZero(z, N);
  double zTmp[N];
  //
  double xyInfSup[4 * N];
  double* xInf = getXInf(xyInfSup, N);
  double* xSup = getXSup(xyInfSup, N);
  double* yInf = getYInf(xyInfSup, N);
  double* ySup = getYSup(xyInfSup, N);

  // Compute the pads charge considering the
  // Mathieson set w_k, mu_x, mu_y
  // TODO: Minor optimization  avoid to
  // compute  x[:] - dx[:]  i.E use xInf / xSup
  for (int k = 0; k < K; k++) {
    // xInf[:] = x[:] - dx[:] - muX[k]
    vectorAddVector(x, -1.0, dx, N, xInf);
    vectorAddScalar(xInf, -mu[k], N, xInf);
    // xSup = xInf + 2.0 * dxy[0]
    vectorAddVector(xInf, 2.0, dx, N, xSup);
    // yInf = xy[1] - dxy[1] - mu[k,1]
    // ySup = yInf + 2.0 * dxy[1]
    vectorAddVector(y, -1.0, dy, N, yInf);
    vectorAddScalar(yInf, -mu[K + k], N, yInf);
    // ySup = yInf + 2.0 * dxy[0]
    vectorAddVector(yInf, 2.0, dy, N, ySup);
    //
    compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId, zTmp);
    // Multiply by the weight w[k]
    double wTmp = (k != K - 1) ? w[k] : lastW;
    vectorAddVector(z, wTmp, zTmp, N, z);
  }
  // ??? vectorPrint("z", z, N);
  // ??? vectorPrint("zObs", zObs, N);

  //
  // To Normalize each cathode with the charge sum
  // of unsaturated pads
  // NOT USED in this residual computation
  double sumNormalizedZ[2];
  if (ClusterConfig::fittingLog >= ClusterConfig::debug) {
    for (int i = 0; i < N; i++) {
      if (cath[i] == 0) {
        sumNormalizedZ[0] += notSaturated[i] * z[i];
      } else {
        sumNormalizedZ[1] += notSaturated[i] * z[i];
      }
    }
  }

  // Charge normalization
  // Get the max charge of unsaturated pads for each cathodes
  double meanCoef = chargeNormalization( cath, notSaturated, cathMax, N, z, cathCoefNorm );
  //
  // printf("maxCath: %f %f\n", cathMax[0], cathMax[1]);
  // printf("coefNorm: %f %f\n", coefNorm[0], coefNorm[1]);
  // printf("meaCoef: %f \n", meanCoef);
  //

  //
  // Cathode Penalization
  //
  // Consider the charge sum for each cathode
  // Tested but NOT USED
  // To be removed for perf
  double chargePerCath[2] = {0., 0.};
  for (int i = 0; i < N; i++) {
    // To have traces about the fitting
    chargePerCath[cath[i]] += z[i];
  }
  double cathPenal = 0;
  if (ClusterConfig::fittingLog >= ClusterConfig::debug) {
    cathPenal = fabs(zCathTotalCharge[0] - chargePerCath[0]) +
                fabs(zCathTotalCharge[1] - chargePerCath[1]);
  }

  //
  // w-Penalization
  //
  // Each w, must be 0 < w < 1
  double wPenal = 0.0;
  for (int k = 0; k < (K - 1); k++) {
    if (w[k] < 0.0) {
      wPenal += (-w[k]);
    } else if (w[k] > 1.0) {
      wPenal += (w[k] - 1.0);
    }
  }
  // ... and the w-sum must be equal to 1
  wPenal = wPenal + fabs(1.0 - vectorSum(w, K - 1) - lastW);
  if (ClusterConfig::fittingLog >= ClusterConfig::debug) {
    printf("    wPenal: %f\n", wPenal);
  }
  // Compute residual
  for (int i = 0; i < N; i++) {
    // Don't consider saturated pads (notSaturated[i] = 0)
    double mask = notSaturated[i];
    if ((notSaturated[i] == 0) && (z[i] < zObs[i])) {
      // Except those charge < Observed charge
      mask = 1.0;
    }
    //
    // Residuals with penalization
    //
    gsl_vector_set(residuals, i, mask * ((z[i] - zObs[i]) + meanCoef * wPenal));
    //
    // Without penalization
    // gsl_vector_set(residuals, i, mask * (zObs[i] - z[i]) + 0 * wPenal);
    //
    // Other studied penalization
    // gsl_vector_set(residuals, i, (zObs[i] - z[i]) * (1.0 + cathPenal) +
    // wPenal);
  }
  if (ClusterConfig::fittingLog >= ClusterConfig::debug) {
    printf("    Observed sumCath0=%15.8f, sumCath1=%15.8f,\n",
           zCathTotalCharge[0], zCathTotalCharge[1]);
    // printf("  fitted   sumCath0=%15.8f, sumCath1=%15.8f,\n", chargePerCath,
    // chargePerCath);
    printf("    Penalties cathPenal=%5.4g wPenal=%5.4g \n", 1.0 + cathPenal,
           wPenal);
    printf("    Residues\n");
    printf("  %15s  %15s  %15s %15s %15s %15s\n", "zObs", "z", "cathWeight",
           "norm. factor", "notSaturated", "residual");
    for (int i = 0; i < N; i++) {
      printf("  %15.8f  %15.8f  %15.8f  %15.8f         %d  %15.8f\n", zObs[i],
             z[i], cathWeights[i], sumNormalizedZ[cath[i]] * cathWeights[i],
             notSaturated[i], gsl_vector_get(residuals, i));
    }
    printf("\n");
  }
  if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
    printf("    |f| = %g \n", gsl_blas_dnrm2(residuals));
  }
  return GSL_SUCCESS;
}

// Derivate of the Charge Integral i.e. mathieson
int df_ChargeIntegral(const gsl_vector* gslParams, void* dataFit,
                     gsl_matrix* J)
{
  funcDescription_t* dataPtr = (funcDescription_t*)dataFit;
  int N = dataPtr->N;
  int K = dataPtr->K;
  const double* x = dataPtr->x_ptr;
  const double* y = dataPtr->y_ptr;
  const double* dx = dataPtr->dx_ptr;
  const double* dy = dataPtr->dy_ptr;
  const Mask_t* cath = dataPtr->cath_ptr;
  const double* zObs = dataPtr->zObs_ptr;
  Mask_t* notSaturated = dataPtr->notSaturated_ptr;
  int chamberId = dataPtr->chamberId;
  double* cathWeights = dataPtr->cathWeights_ptr;
  double* cathMax = dataPtr->cathMax_ptr;
  double* zCathTotalCharge = dataPtr->zCathTotalCharge_ptr;
  double* cathCoefNorm = dataPtr->cathCoefNorm_ptr;
  // ??? int verbose = dataPtr->verbose;
  // Parameters
  const double* params = gsl_vector_const_ptr(gslParams, 0);
  // Note:
  //  mux = mu[0:K-1]
  //  muy = mu[K:2K-1]
  const double* mux = &params[0];
  const double* muy = &params[K];
  double* w = (double*)&params[2 * K];

  // Compute mathieson on x/y
  // and charge integral on x/y
  double xCI[N], yCI[N];
  double xMath[N], yMath[N], xyMath[N];
  double xyInf[N], xySup[N];
  double xyVar[N];
  // Set constrain: sum_(w_k) = 1
  double lastW = 1.0 - vectorSum(w, K - 1);

  if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
    printf("  df evaluation at:\n");
    for (int k = 0; k < K; k++) {
      printf("    mu_k[%d] = %g %g \n", k, mux[k], muy[k]);
    }
    for (int k = 0; k < K - 1; k++) {
      printf("    w_k[%d] = %g \n", k, w[k]);
    }
    // Last W
    printf("    w_k[%d] = %g \n", K - 1, lastW);
  }

  for (int k = 0; k < K; k++) {
    double w_k = (k < (K-1))? w[k]: lastW;
    //
    // X components for CI and mathieson
    //
    // xyVar = x - mux[k]
    vectorAddScalar(x, -mux[k], N, xyVar);
    // xInf = xyVar - dx
    vectorAddVector(xyVar, -1.0, dx, N, xyInf);
    // xSup = xInf + 2.0 * dx
    vectorAddVector(xyInf, 2.0, dx, N, xySup);
    // Compute the derivate : mathieson(xSup) - mathieson(xInf)
    compute1DMathieson( xyInf, N, 0, chamberId, xyMath);
    compute1DMathieson( xySup, N, 0, chamberId, xMath);
    vectorAddVector( xMath, -1, xyMath, N, xMath);
    vectorMultScalar(xMath, 4, N, xMath);
    // Compute the 1D Charge integral on x
    compute1DPadIntegrals(xyInf, xySup, N, 0, chamberId, xCI);
    //
    // Y components for CI and mathieson
    //
    // xyVar = y - muy[k]
    vectorAddScalar(y, -muy[k], N, xyVar);
    // Mathieson at  xyVar
    compute1DMathieson( xyVar, N, 1, chamberId, yMath);
    // yInf = xyVar - dy
    vectorAddVector(xyVar, -1.0, dy, N, xyInf);
    // ySup = yInf + 2.0 * dy
    vectorAddVector(xyInf, 2.0, dy, N, xySup);
    // Compute the derivate : mathieson(ySup) - mathieson(yInf)
    compute1DMathieson( xyInf, N, 1, chamberId, xyMath);
    compute1DMathieson( xySup, N, 1, chamberId, yMath);
    vectorAddVector( yMath, -1, xyMath, N, yMath);
    vectorMultScalar(yMath, 4, N, yMath);
    // Compute the 1D Charge integral on y
    compute1DPadIntegrals(xyInf, xySup, N, 1, chamberId, yCI);

    // Normalization factor
    // double meanCoef = chargeNormalization( cath, notSaturated, cathMax, N, z);
    //
    //  Jacobian matrix
    //
    // d / dmux_k component

    for (int i = 0; i < N; i++) {
      gsl_matrix_set (J, i, k,  -0.5*w_k*cathCoefNorm[cath[i]]*xMath[i]*yCI[i]);
    }
    // d / dmuy_k component
    for (int i = 0; i < N; i++) {
      gsl_matrix_set (J, i, k+K,  -0.5*w_k*cathCoefNorm[cath[i]]*xCI[i]*yMath[i]);
    }
    // d / dw_k component
    if (k < K-1) {
      for (int i = 0; i < N; i++) {
        gsl_matrix_set (J, i, 2*K+k,  -0.5*cathCoefNorm[cath[i]]*xCI[i]*yCI[i]);
      }
    }
    // ??? vectorPrint("xMath", xMath, N);
    // vectorPrint("yMath", yMath, N);
    // vectorPrint("xCI", xCI, N);
    // vectorPrint("yCI", yCI, N);
  }
  if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
    double sumdParam[3*K];
    vectorSet( sumdParam, 0.0, 3*K);
    for (int k=0; k < 3*K - 1; k++) {
      printf("%2d: ", k);
      for (int i=0; i < N; i++) {
         sumdParam[k] += gsl_matrix_get( J, i, k);
         printf("%g ", gsl_matrix_get(J, i, k) );
      }
      printf("\n");
    }
    printf("  Sum_i d/dparam :\n");
    for (int k = 0; k < K; k++) {
      printf("    mux/y[%d] = %g %g \n", k, sumdParam[k], sumdParam[K + k]);
    }
    for (int k = 0; k < K; k++) {
      printf("    w_k[%d] = %g \n", k, sumdParam[2*K + k]);
    }
  }
  return GSL_SUCCESS;
}

// Invalid version
int f_ChargeIntegral0(const gsl_vector* gslParams, void* dataFit,
                      gsl_vector* residuals)
{
  funcDescription_t* dataPtr = (funcDescription_t*)dataFit;
  int N = dataPtr->N;
  int K = dataPtr->K;
  const double* x = dataPtr->x_ptr;
  const double* y = dataPtr->y_ptr;
  const double* dx = dataPtr->dx_ptr;
  const double* dy = dataPtr->dy_ptr;
  const Mask_t* cath = dataPtr->cath_ptr;
  const double* zObs = dataPtr->zObs_ptr;
  const Mask_t* notSaturated = dataPtr->notSaturated_ptr;
  int chamberId = dataPtr->chamberId;
  double* cathWeights = dataPtr->cathWeights_ptr;
  double* cathMax = dataPtr->cathMax_ptr;
  double* zCathTotalCharge = dataPtr->zCathTotalCharge_ptr;
  int verbose = dataPtr->verbose;
  // Parameters
  const double* params = gsl_vector_const_ptr(gslParams, 0);
  // Note:
  //  mux = mu[0:K-1]
  //  muy = mu[K:2K-1]
  const double* mu = &params[0];
  double* w = (double*)&params[2 * K];

  // Set constrain: sum_(w_k) = 1
  // Rewriten ??? w[K-1] = 1.0 - vectorSum( w, K-1 );
  double lastW = 1.0 - vectorSum(w, K - 1);
  if (verbose > 1) {
    printf("  Function evaluation at:\n");
    for (int k = 0; k < K; k++) {
      printf("    mu_k[%d] = %g %g \n", k, mu[k], mu[K + k]);
    }
    for (int k = 0; k < K - 1; k++) {
      printf("    w_k[%d] = %g \n", k, w[k]);
    }
    // Last W
    printf("    w_k[%d] = %g \n", K - 1, lastW);
  }
  int i;

  // Charge Integral on Pads
  double z[N];
  // Not used
  double z_k[N];
  double zTmp[N];
  // TODO: optimize compute before
  double xyInfSup[4 * N];
  //
  double* xInf = getXInf(xyInfSup, N);
  double* xSup = getXSup(xyInfSup, N);
  double* yInf = getYInf(xyInfSup, N);
  double* ySup = getYSup(xyInfSup, N);

  vectorSetZero(z, N);
  for (int k = 0; k < K; k++) {
    // xInf[:] = x[:] - dx[:] - muX[k]
    vectorAddVector(x, -1.0, dx, N, xInf);
    vectorAddScalar(xInf, -mu[k], N, xInf);
    // xSup = xInf + 2.0 * dxy[0]
    vectorAddVector(xInf, 2.0, dx, N, xSup);
    // yInf = xy[1] - dxy[1] - mu[k,1]
    // ySup = yInf + 2.0 * dxy[1]
    vectorAddVector(y, -1.0, dy, N, yInf);
    vectorAddScalar(yInf, -mu[K + k], N, yInf);
    // ySup = yInf + 2.0 * dxy[0]
    vectorAddVector(yInf, 2.0, dy, N, ySup);
    //
    // z[:] +=  w[k] * cathWeight
    //       * computeMathieson2DIntegral( xInf[:], xSup[:], yInf[:], ySup[:], N
    //       )
    // Inv. ??? compute2DPadIntegrals(xyInfSup, N, chamberId, zTmp);
    compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId, zTmp);
    // GG Note: if require compute the charge of k-th component
    // z_k[k] = vectorSum( zTmp, N )
    // Done after the loop: vectorMultVector( zTmp, cathWeights, N, zTmp);
    double wTmp = (k != K - 1) ? w[k] : lastW;
    // saturated pads
    // ??? ?
    // vectorMaskedMult( zTmp, notSaturated, N, zTmp);
    vectorAddVector(z, wTmp, zTmp, N, z);
  }
  // Normalize each cathodes
  double sumNormalizedZ[2];
  for (int i = 0; i < N; i++) {
    if (cath[i] == 0) {
      sumNormalizedZ[0] += notSaturated[i] * z[i];
    } else {
      sumNormalizedZ[1] += notSaturated[i] * z[i];
    }
  }
  if (0) {
    // Normalize with the charge sum
    double var[2] = {1.0 / sumNormalizedZ[0], 1. / sumNormalizedZ[1]};
    for (int i = 0; i < N; i++) {
      if (cath[i] == 0) {
        z[i] = z[i] * var[0];
      } else {
        z[i] = z[i] * var[1];
      }
    }
    if (verbose > 1) {
      printf("  sum mathiesons %f %f\n", sumNormalizedZ[0], sumNormalizedZ[1]);
    }
    vectorMultVector(z, cathWeights, N, z);
    // vectorMultScalar( z, 2.0/sumNormalizedZ, N, z);

  } else {
    // Normalize with the max
    double maxThZ[2];
    for (int i = 0; i < N; i++) {
      maxThZ[cath[i]] = fmax(maxThZ[cath[i]], notSaturated[i] * z[i]);
    }
    double var[2] = {cathMax[0] / maxThZ[0], cathMax[1] / maxThZ[1]};
    for (int i = 0; i < N; i++) {
      z[i] = z[i] * var[cath[i]];
    }
  }

  // ??? why abs value for penalties
  double wPenal = abs(1.0 - vectorSum(w, K - 1) + lastW);
  // Not Used: zPenal
  // double zPenal = abs( 1.0 - vectorSum( z, N ) );
  //
  // Charge on cathodes penalisation
  double zCath0 = 0.0, zCath1 = 0.0;
  for (int i = 0; i < N; i++) {
    if (cath[i] == 0) {
      zCath0 += (z[i] * notSaturated[i]);
    } else {
      zCath1 += (z[i] * notSaturated[i]);
    }
  }
  if (verbose > 1) {
    printf("  non saturated sum zCath0/1 %f %f\n", zCath0, zCath1);
  }
  double cathPenal =
    fabs(zCathTotalCharge[0] - zCath0) + fabs(zCathTotalCharge[1] - zCath1);
  // ??? vectorAdd( zObs, -1.0, residual );
  // TODO Optimize (elementwise not a good solution)
  for (i = 0; i < N; i++) {
    // gsl_vector_set(residuals, i, (zObs[i] - z[i]) * (1.0 + cathPenal) +
    // wPenal);
    double mask;
    if ((notSaturated[i] == 0) && (z[i] < zObs[i])) {
      mask = 1;
    } else {
      mask = notSaturated[i];
    }
    gsl_vector_set(residuals, i, mask * (zObs[i] - z[i]) + 0. * wPenal);
  }
  if (verbose > 1) {
    printf("  observed sumCath0=%15.8f, sumCath1=%15.8f,\n",
           zCathTotalCharge[0], zCathTotalCharge[1]);
    printf("  fitted   sumCath0=%15.8f, sumCath1=%15.8f,\n", zCath0, zCath1);
    printf("  cathPenal=%5.4g wPenal=%5.4g \n", 1.0 + cathPenal, wPenal);
    printf("  residual\n");
    printf("  %15s  %15s  %15s %15s %15s %15s\n", "zObs", "z", "cathWeight",
           "norm. factor", "notSaturated", "residual");
    for (i = 0; i < N; i++) {
      printf("  %15.8f  %15.8f  %15.8f  %15.8f         %d  %15.8f\n", zObs[i],
             z[i], cathWeights[i], sumNormalizedZ[cath[i]] * cathWeights[i],
             notSaturated[i], gsl_vector_get(residuals, i));
    }
    printf("\n");
  }
  return GSL_SUCCESS;
}

void printState(int iter, gsl_multifit_fdfsolver* s, int K, int N)
{
  printf("  Fitting iter=%3d |f(x)| =%g\n", iter, gsl_blas_dnrm2(s->f));
  printf("    mu (x,y):");
  int k = 0;
  for (; k < 2 * K; k++) {
    printf(" % 7.3f", gsl_vector_get(s->x, k));
  }
  printf("\n");
  double sumW = 0;
  printf("    w:");
  for (; k < 3 * K - 1; k++) {
    double w = gsl_vector_get(s->x, k);
    sumW += w;
    printf(" %7.3f", gsl_vector_get(s->x, k));
  }
  // Last w : 1.0 - sumW
  printf(" %7.3f", 1.0 - sumW);

  printf("\n");
  k = 0;
  printf("    dx:");
  for (; k < 2 * K; k++) {
    printf(" % 7.3f", gsl_vector_get(s->dx, k));
  }
  printf("\n");
  printf("    Jacobian");
  double sum = 0.0;
  for (int k=0; k < K; k++) {
    printf("    k:");
    for (int i=0; i < N; i++) {
      printf(" % 7.3f",  gsl_matrix_get (s->J, i, k) );
    }
    printf("\n");
  }
  printf("\n");
}

// Notes :
//  - the intitialization of Mathieson module must be done before
//  (initMathieson)
// zCathTotalCharge = sum excludind saturated pads
namespace o2
{
namespace mch
{
void printState(int iter, gsl_multifit_fdfsolver* s, int K, int N)
{
  printf("  Fitting iter=%3d |f(x)|=%g\n", iter, gsl_blas_dnrm2(s->f));
  printf("    mu (x,y):");
  int k = 0;
  for (; k < 2 * K; k++) {
    printf(" % 7.3f", gsl_vector_get(s->x, k));
  }
  printf("\n");
  double sumW = 0;
  printf("    w:");
  for (; k < 3 * K - 1; k++) {
    double w = gsl_vector_get(s->x, k);
    sumW += w;
    printf(" %7.3f", gsl_vector_get(s->x, k));
  }
  // Last w : 1.0 - sumW
  printf(" %7.3f", 1.0 - sumW);

  printf("\n");
  k = 0;
  printf("    dx:");
  for (; k < 2 * K; k++) {
    printf(" % 7.3f", gsl_vector_get(s->dx, k));
  }
  printf("\n");
  printf("    Jacobian\n");
  double sum = 0.0;
  for (int k=0; k < K; k++) {
    printf("    k=%2d mux:", k);
    for (int i=0; i < N; i++) {
      printf(" % 7.3f",  gsl_matrix_get (s->J, i, k) );
    }
    printf("\n");
    printf("    k=%2d muy:", k);
    for (int i=0; i < N; i++) {
      printf(" % 7.3f",  gsl_matrix_get (s->J, i, k+K) );
    }
    printf("\n");
    if (k < K-1) {
      printf("    k=%2d w  :", k);
      for (int i=0; i < N; i++) {
        printf(" % 7.3f",  gsl_matrix_get (s->J, i, k+2*K) );
      }
    }
    printf("\n");
  }
  printf("\n");
}

void fitMathieson(const Pads& iPads, double* thetaInit, int kInit, int mode,
                  double* thetaFinal, double* khi2, double* pError)
{
  int status;

  // process / mode
  int p = mode;
  // verbose unused ???
  int verbose = p & 0x3;
  p = p >> 2;
  int doJacobian = p & 0x1;
  p = p >> 1;
  int computeKhi2 = p & 0x1;
  p = p >> 1;
  int computeStdDev = p & 0x1;
  if (ClusterConfig::fittingLog >= ClusterConfig::info) {
    printf("\n> [fitMathieson] Fitting \n");
    printf(
      "  mode: verbose, doJacobian, computeKhi2, computeStdDev %d %d %d %d\n",
      verbose, doJacobian, computeKhi2, computeStdDev);
  }
  //
  int N = iPads.getNbrOfPads();
  //
  double* muAndWi = getMuAndW(thetaInit, kInit);
  //
  // Check if fitting is possible
  double* muAndWf = getMuAndW(thetaFinal, kInit);
  if (3 * kInit - 1 > N) {
    muAndWf[0] = NAN;
    muAndWf[kInit] = NAN;
    muAndWf[2 * kInit] = NAN;
    return;
  }

  funcDescription_t mathiesonData;
  double cathMax[2] = {0.0, 0.0};
  double* cathWeights;

  if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
    vectorPrintShort("  iPads.cath", iPads.getCathodes(), N);
    vectorPrint("  iPads.q", iPads.getCharges(), N);
  }

  // if( 1 ) {
  // Add boundary Pads
  // pads = Pads::addBoundaryPads( iPads.x, iPads.y, iPads.dx, iPads.dy,
  //        iPads.q, iPads.cath, iPads.saturate, iPads.chamberId, iPads.nPads );

  // Function description (extra data nor parameters)
  mathiesonData.N = N;
  mathiesonData.K = kInit;
  mathiesonData.x_ptr = iPads.getX();
  mathiesonData.y_ptr = iPads.getY();
  mathiesonData.dx_ptr = iPads.getDX();
  mathiesonData.dy_ptr = iPads.getDY();
  mathiesonData.cath_ptr = iPads.getCathodes();
  mathiesonData.zObs_ptr = iPads.getCharges();
  Mask_t notSaturated[N];
  vectorCopyShort(iPads.getSaturates(), N, notSaturated);
  vectorNotShort(notSaturated, N, notSaturated);
  mathiesonData.notSaturated_ptr = notSaturated;
  //} else {
  /*
  // Function description (extra data nor parameters)
  mathiesonData.N = N;
  mathiesonData.K = kInit;
  mathiesonData.x_ptr = iPads.x;
  mathiesonData.y_ptr = iPads.y;
  mathiesonData.dx_ptr = iPads.dx;
  mathiesonData.dy_ptr = iPads.dy;
  mathiesonData.cath_ptr = iPads.cath;
  mathiesonData.zObs_ptr = iPads.q;
  Mask_t notSaturated[N];
  vectorCopyShort( iPads.saturate, N, notSaturated);
  vectorNotShort( notSaturated, N, notSaturated );
  mathiesonData.notSaturated_ptr = notSaturated;
  }
  */
  // Total Charge per cathode plane
  double zCathTotalCharge[2];
  double cathCoefNorm[2] = {0.0};
  Mask_t mask[N];
  // Cath 1
  vectorCopyShort(mathiesonData.cath_ptr, N, mask);
  // Logic And operation
  vectorMultVectorShort(mathiesonData.notSaturated_ptr, mask, N, mask);
  zCathTotalCharge[0] = vectorMaskedSum(mathiesonData.zObs_ptr, mask, N);
  // cath 0
  vectorCopyShort(mathiesonData.cath_ptr, N, mask);
  vectorNotShort(mask, N, mask);
  // Logic And operation
  vectorMultVectorShort(mathiesonData.notSaturated_ptr, mask, N, mask);
  zCathTotalCharge[1] = vectorMaskedSum(mathiesonData.zObs_ptr, mask, N);

  // Init the weights
  cathWeights = new double[N];
  for (int i = 0; i < N; i++) {
    cathWeights[i] = (mathiesonData.cath_ptr[i] == 0) ? zCathTotalCharge[0]
                                                      : zCathTotalCharge[1];
    cathMax[mathiesonData.cath_ptr[i]] = std::fmax(
      cathMax[mathiesonData.cath_ptr[i]],
      mathiesonData.notSaturated_ptr[i] * mathiesonData.zObs_ptr[i]);
  }
  if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
    vectorPrintShort("mathiesonData.cath_ptr", mathiesonData.cath_ptr, N);
    vectorPrintShort("mathiesonData.notSaturated_ptr",
                     mathiesonData.notSaturated_ptr, N);
    vectorPrint("mathiesonData.zObs_ptr", mathiesonData.zObs_ptr, N);
  }
  mathiesonData.cathWeights_ptr = cathWeights;
  mathiesonData.cathMax_ptr = cathMax;
  mathiesonData.chamberId = iPads.getChamberId();
  mathiesonData.zCathTotalCharge_ptr = zCathTotalCharge;
  mathiesonData.cathCoefNorm_ptr = cathCoefNorm;
  mathiesonData.verbose = verbose;
  //
  // Define Function, jacobian
  gsl_multifit_function_fdf f;
  f.f = &f_ChargeIntegral;
  f.df = nullptr;
  //f.df = df_ChargeIntegral;
  f.fdf = nullptr;
  f.n = N;
  f.p = 3 * kInit - 1;
  f.params = &mathiesonData;

  bool doFit = true;
  // K test
  int K = kInit;
  // Sort w
  int maxIndex[kInit];
  for (int k = 0; k < kInit; k++) {
    maxIndex[k] = k;
  }
  double* w = &muAndWi[2 * kInit];
  std::sort(maxIndex, &maxIndex[kInit],
            [=](int a, int b) { return (w[a] > w[b]); });

  while (doFit) {
    // Select the best K's
    // Copy kTest max
    double muAndWTest[3 * K];
    // Mu part
    for (int k = 0; k < K; k++) {
      // Respecttively mux, muy, w
      muAndWTest[k] = muAndWi[maxIndex[k]];
      muAndWTest[k + K] = muAndWi[maxIndex[k] + kInit];
      muAndWTest[k + 2 * K] = muAndWi[maxIndex[k] + 2 * kInit];
    }
    if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
      vectorPrint("  Selected w", &muAndWTest[2 * K], K);
      vectorPrint("  Selected mux", &muAndWTest[0], K);
      vectorPrint("  Selected muy", &muAndWTest[K], K);
    }
    mathiesonData.K = K;
    f.p = 3 * K - 1;
    // Set initial parameters
    // Inv ??? gsl_vector_view params0 = gsl_vector_view_array(muAndWi, 3 * K -
    // 1);
    gsl_vector_view params0 = gsl_vector_view_array(muAndWTest, 3 * K - 1);

    // Fitting method
    gsl_multifit_fdfsolver* s = gsl_multifit_fdfsolver_alloc(
      gsl_multifit_fdfsolver_lmsder, N, 3 * K - 1);
    // associate the fitting mode, the function, and the starting parameters
    gsl_multifit_fdfsolver_set(s, &f, &params0.vector);

    if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
      o2::mch::printState(-1, s, K, N);
    }
    // double initialResidual = gsl_blas_dnrm2(s->f);
    double initialResidual = 0.0;
    // Fitting iteration
    status = GSL_CONTINUE;
    double residual = DBL_MAX;
    ;
    double prevResidual = DBL_MAX;
    ;
    double prevTheta[3 * K - 1];
    // ??? for (int iter = 0; (status == GSL_CONTINUE) && (iter < 500); iter++)
    // {
    for (int iter = 0; (status == GSL_CONTINUE) && (iter < 50); iter++) {
      // TODO: to speed if possible
      for (int k = 0; k < (3 * K - 1); k++) {
        prevTheta[k] = gsl_vector_get(s->x, k);
      }
      // printf("  Debug Fitting iter=%3d |f(x)|=%g\n", iter,
      // gsl_blas_dnrm2(s->f));
      status = gsl_multifit_fdfsolver_iterate(s);
      if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
        printf("  Solver status = %s\n", gsl_strerror(status));
      }
      if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
        o2::mch::printState(iter, s, K, N);
      }
      /* ???? Inv
      if (status) {
        printf("  ???? End fitting \n");
        break;
      };
      */
      // GG TODO ???: adjust error in fct of charge
      status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
      if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
        printf("  Status multifit_test_delta = %d %s\n", status,
               gsl_strerror(status));
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
        if (ClusterConfig::fittingLog >= ClusterConfig::info) {
          printf("  Stop iteration (dResidu~0), prevResidual=%f residual=%f\n",
                 prevResidual, residual);
        }
        for (int k = 0; k < (3 * K - 1); k++) {
          gsl_vector_set(s->x, k, prevTheta[k]);
        }
        status = GSL_SUCCESS;
      }
    }
    double finalResidual = gsl_blas_dnrm2(s->f);
    bool keepInitialTheta =
      fabs(finalResidual - initialResidual) / initialResidual < 1.0e-1;

    // Khi2
    if (computeKhi2 && (khi2 != nullptr)) {
      // Khi2
      double chi = gsl_blas_dnrm2(s->f);
      double dof = N - (3 * K - 1);
      double c = fmax(1.0, chi / sqrt(dof));
      if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
        printf("  K=%d, chi=%f, chisq/dof = %g\n", K, chi * chi,
               chi * chi / dof);
      }
      khi2[0] = chi * chi / dof;
    }

    // ???? if (keepInitialTheta) {
    if (0) {
      // Keep the result of EM (GSL bug when no improvemebt)
      copyTheta(thetaInit, kInit, thetaFinal, kInit, kInit);
    } else {
      // Fitted parameters
      /* Invalid ???
      for (int k = 0; k < (3 * K - 1); k++) {
        muAndWf[k] = gsl_vector_get(s->x, k);
      }
      */

      // Mu part
      for (int k = 0; k < K; k++) {
        muAndWf[k] = gsl_vector_get(s->x, k);
        muAndWf[k + kInit] = gsl_vector_get(s->x, k + K);
      }
      // w part
      double sumW = 0;
      for (int k = 0; k < K - 1; k++) {
        double w = gsl_vector_get(s->x, k + 2 * K);
        sumW += w;
        muAndWf[k + 2 * kInit] = w;
      }
      // Last w : 1.0 - sumW
      muAndWf[3 * kInit - 1] = 1.0 - sumW;

      // Parameter error
      /* Pb Mac compilation
      if (computeStdDev && (pError != nullptr)) { //
        // Covariance matrix an error
        gsl_matrix* covar = gsl_matrix_alloc(3 * K - 1, 3 * K - 1);
        gsl_multifit_covar(s->J, 0.0, covar);
        for (int k = 0; k < (3 * K - 1); k++) {
          pError[k] = sqrt(gsl_matrix_get(covar, k, k));
        }
        gsl_matrix_free(covar);
      }
      */
    }
    if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
      printf("  status parameter error = %s\n", gsl_strerror(status));
    }
    gsl_multifit_fdfsolver_free(s);
    K = K - 1;
    // doFit = (K < 3) && (K > 0);
    doFit = false;
  } // while(doFit)
  // Release memory
  delete[] cathWeights;
  //
  return;
}

} // namespace mch
} // namespace o2

void fitMathieson( const double* x, const double* y, const double* dx, const double* dy, const double* q,
                   const o2::mch::Mask_t* cath, const o2::mch::Mask_t* sat, int chId, int nPads,
                   double* thetaInit, int kInit,
                   double* thetaFinal, double* khi2, double* pError) {
  //
  Pads pads = o2::mch::Pads( x, y, dx, dy, q, cath, sat, chId, nPads );
  int mode = 0;
  o2::mch::fitMathieson( pads, thetaInit, kInit, mode,
                  thetaFinal, khi2, pError);
}