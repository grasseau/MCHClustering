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

#ifndef _MATHIESON_H
#define _MATHIESON_H

#include <cstddef>
#include <cmath>

#include "PadsPEM.h"

namespace o2
{
namespace mch
{
void initMathieson();

void compute1DPadIntegrals( const double *xInf, const double *xSup,
                           int N, int chamberId, bool xAxe, double Integrals[]);

void compute2DPadIntegrals(const double* xInf, const double* xSup, const double* yInf, const double* ySup,
                           int N, int chamberId,
                           double Integrals[]);

void compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0, const double* theta,
                                           int N, int K, int chamberId,
                                           double Integrals[]);

void computeFastCij( const Pads &pads, const Pads &theta,
                            double Cij[] );
void computeCij( const Pads &pads, const Pads &theta,
                            double Cij[] );
} // namespace mch
} // namespace o2

// For InspectModel (if required)
extern "C" {
  void o2_mch_initMathieson();
  void o2_mch_compute2DPadIntegrals(const double* xInf, const double* xSup, const double* yInf, const double* ySup,
                           int N, int chamberId,
                           double Integrals[]);
  void o2_mch_compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0, const double* theta,
                                           int N, int K, int chamberId,
                                           double Integrals[]);
  void o2_mch_computeCij( const double *xyInfSup0, const double *theta,
                            int N, int K, int chamberId, double Cij[] );
}

#endif
