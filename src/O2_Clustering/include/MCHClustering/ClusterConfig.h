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

/// \file ClusterConfig.h
/// \brief Clustering and fifting parameters
/// \author Gilles Grasseau, Subatech

#ifndef ALICEO2_MCH_CLUSTERCONFIG_H_
#define ALICEO2_MCH_CLUSTERCONFIG_H_

namespace o2
{
namespace mch
{

struct ClusterConfig {
  // Physical-Numerical parameters 
  static constexpr double minChargeOfClusterPerCathode = 1.1;    // Lowest Charge of a Group
  
  // Logs
  enum VerboseMode {
   no = 0x0,      ///< No message
   info = 0x1,    ///< Describes main steps and high level behaviors
   detail = 0x2,  ///< Describes in detail
   debug = 0x3    ///< Ful details
  };
  static constexpr VerboseMode fittingLog = info;
  static constexpr VerboseMode groupsLog = info;
  static constexpr VerboseMode EMLocalMaxLog = info;
  static constexpr VerboseMode laplacianLocalMaxLog = no;
  static constexpr VerboseMode inspectModelLog = no;
  // TODO ???
  // Check, Stat
  // Check
  static constexpr bool groupsCheck = true;
  static constexpr bool padMappingCheck = true;
};

} // namespace mch
} // end namespace o2

#endif // ALICEO2_MCH_CLUSTERCONFIG_H_
