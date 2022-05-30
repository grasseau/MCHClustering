/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

// Propagate the proj-groups to the cath-pads
int Cluster::assignGroupToCathPads( ) {
  //
  // From the cathode group found with the projection,
  int nCath0 = (pads[0]) ? pads[0]->getNbrOfPads() : 0;
  int nCath1 = (pads[1]) ? pads[1]->getNbrOfPads() : 0;
  int nGrp = nbrOfProjGroups;
  // Groups obtain with the projection
  Group_t cath0ToGrpFromProj[nCath0];
  Group_t cath1ToGrpFromProj[nCath1];
  vectorSetZeroShort( cath0ToGrpFromProj, nCath0);
  vectorSetZeroShort( cath1ToGrpFromProj, nCath1);
  vectorSetZeroShort( cathGroup[0], nCath0);
  vectorSetZeroShort( cathGroup[1], nCath1);
  // Mapping proj-groups to cath-groups
  Group_t projGrpToCathGrp[nGrp+1];
  vectorSetZeroShort( projGrpToCathGrp, nGrp+1);
  int nCathGrp = 0;
  //
  if (VERBOSE > 0) {
    printf("  assignGroupToCathPads\n");
  }
  //
  PadIdx_t i, j;
  short g, prevGroup0, prevGroup1;
  if (nbrOfCathodePlanes == 1) {
    // Single cathode plane
    vectorCopyShort( projPadToGrp, pads[singleCathPlaneID]->getNbrOfPads(), cathGroup[singleCathPlaneID]);
    return nGrp;
  }
  int nProjPads = projectedPads->getNbrOfPads();
  for( int k=0; k < nProjPads; k++) {
    // Group of the projection k
    g = projPadToGrp[k];
    // Intersection indexes of the 2 cath
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    if (VERBOSE > 1) {
      printf("map k=%d g=%d to i=%d/%d, j=%d/%d\n", k, g, i, nCath0, j, nCath1);
    }
    //
    // Cathode 0
    //
    if ( (i >= 0) && (nCath0 != 0) ) {
      // if the pad has already been set
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
  }
  if (VERBOSE > 2) {
    printf("assignGroupToCathPads\n");
    vectorPrintShort( "  cath0ToGrpFromProj ??? ",cath0ToGrpFromProj,nCath0);
    vectorPrintShort( "  cath1ToGrpFromProj ??? ",cath1ToGrpFromProj,nCath1);
    vectorPrintShort( "  projGrpToCathGrp ??? ", projGrpToCathGrp, nGrp+1);
  }
  //
  // Renumering cathodes groups
  // Desactivated ???
  // int nNewGrp = renumberGroups( projGrpToCathGrp, nGrp);
  // Test if renumbering is necessary
  // if( nNewGrp != nGrp) throw std::overflow_error("Divide by zero exception");
  // vectorMapShort( cath0ToGrpFromProj, projGrpToCathGrp, nCath0);
  // vectorMapShort( cath1ToGrpFromProj, projGrpToCathGrp, nCath1);
  //
  //
  int nNewGrp = nGrp;
  //
  // Set/update the cath/proj Groups
  vectorCopyShort( cath0ToGrpFromProj, nCath0, cathGroup[0]);
  vectorCopyShort( cath1ToGrpFromProj, nCath1, cathGroup[1]);
  //
  for( i=0; i<nProjPads; i++) {
    projPadToGrp[i] = projGrpToCathGrp[ projPadToGrp[i] ];
  }
  if (VERBOSE > 1) {
    vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    vectorPrintShort("  cath0ToGrp", cathGroup[0], nCath0);
    vectorPrintShort("  cath1ToGrp", cathGroup[1], nCath1);
  }
  //
  return nNewGrp;
}

// Assign a group to the original pads
// Update the pad group and projected-pads group
int Cluster::assignPadsToGroupFromProj(
        // const PadIdx_t *cath0ToPadIdx, const PadIdx_t *cath1ToPadIdx,
        // int nGrp, int nPads, short *padMergedGrp ) {
        int nGrp ) {
// cath0ToPadIdx : pad indices of cath0 (cath0ToPadIdx[0..nCath0] -> i-pad
// outputs:
  short matGrpGrp[ (nGrp+1)*(nGrp+1)];
  //
  // vectorSetShort( wellSplitGroup, 1, nGrp+1);
  vectorSetZeroShort( matGrpGrp, (nGrp+1)*(nGrp+1) );
  //
  PadIdx_t i, j;
  short g, prevGroup;
  if (VERBOSE > 1) {
    printf( "[AssignPadsToGroupFromProj]\n");
  }
  // Expand the projected Groups
  // 'projPadToGrp' to the pad groups 'padToGrp'
  // If there are conflicts, fuse the groups
  // Build the Group-to-Group matrix matGrpGrp
  // which describe how to fuse Groups
  // with the projected Groups
  // projPadToGrp
  int nProjPads = projectedPads->getNbrOfPads();
  for( int k=0; k < nProjPads; k++) {
    g = projPadToGrp[k];
    // give the indexes of overlapping pads
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    // Inv ??? if ( (i >= 0) && (cath0ToPadIdx !=0) ) {
    if ( i >= 0 ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      //
      // cath0ToPadIdx: map cathode-pad to the original pad
      /*
      PadIdx_t padIdx = cath0ToPadIdx[i];
      prevGroup = padToGrp[ padIdx ];
      */
      prevGroup = cathGroup[0][i];
      if ( (prevGroup == 0) || (prevGroup == g) ) {
        // Case: no group before or same group
        //
        // ???? padToGrp[ padIdx ] = g;
        cathGroup[0][i] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1;
      } else {
        // Already a grp (Conflict)
        // if ( prevGroup > 0) {
          // Invalid prev group
          // wellSplitGroup[ prevGroup ] = 0;
          // Store in the grp to grp matrix
          // Group to fuse
          cathGroup[0][i] = g;
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1;
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1;
        //}
        // padToGrp[padIdx] = -g;
      }
    }
    //
    // Cathode 1
    //
    // ??? if ( (j >= 0) && (cath1ToPadIdx != 0) ) {
    if ( (j >= 0) ) {
      // Remark: if j is an alone pad (j<0)
      // j is processed as well
      //
      // cath1ToPadIdx: map cathode-pad to the original pad
      // ??? PadIdx_t padIdx = cath1ToPadIdx[j];
      // ??? prevGroup = padToGrp[padIdx];
      prevGroup = cathGroup[1][j];

      if ( (prevGroup == 0) || (prevGroup == g) ){
        // No group before
        // padToGrp[padIdx] = g;
        cathGroup[1][j] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1;
      } else {
        // Already a Group (Conflict)
        // if ( prevGroup > 0) {
          cathGroup[1][j] = g;
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1;
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1;
        // }
        // padToGrp[padIdx] = -g;
      }
    }
  }
  if (VERBOSE > 0) {
    printMatrixShort("  Group/Group matrix", matGrpGrp, nGrp+1, nGrp+1);
    vectorPrintShort("  cathToGrp[0]", cathGroup[0], pads[0]->getNbrOfPads());
    vectorPrintShort("  cathToGrp[1]", cathGroup[1], pads[1]->getNbrOfPads());
  }
  //
  // Merge the groups (build the mapping grpToMergedGrp)
  //
  Group_t grpToMergedGrp[nGrp+1]; // Mapping old groups to new merged groups
  vectorSetZeroShort(grpToMergedGrp, nGrp+1);
  //
  int iGroup = 1; // Describe the current group
  int curGroup;   // Describe the mapping grpToMergedGrp[iGroup]
  while ( iGroup < (nGrp+1)) {
    // Define the new group to process
    if ( grpToMergedGrp[iGroup] == 0 ) {
        // newGroupID++;
        // grpToMergedGrp[iGroup] = newGroupID;
        grpToMergedGrp[iGroup] = iGroup;
    }
    curGroup = grpToMergedGrp[iGroup];
    // printf( "  current iGroup=%d -> grp=%d \n", iGroup, curGroup);
    //
      // Look for other groups in matGrpGrp
      int ishift = iGroup*(nGrp+1);
      // Check if there are an overlaping group
      for (int j=iGroup+1; j < (nGrp+1); j++) {
        if ( matGrpGrp[ishift+j] ) {
          // Merge the groups with the current one
          if ( grpToMergedGrp[j] == 0) {
            // printf( "    newg merge grp=%d -> grp=%d\n", j, curGroup);
            // No group assign before, merge the groups with the current one
            grpToMergedGrp[j] = curGroup;
          } else {
            // Fuse grpToMergedGrp[j] with
            // Merge curGroup and grpToMergedGrp[j]
            // printf( "    oldg merge grp=%d -> grp=%d\n", curGroup, grpToMergedGrp[j]);

            // A group is already assigned, the current grp takes the grp of ???
            // Remark : curGroup < j
            // Fuse and propagate
            grpToMergedGrp[ curGroup ] = grpToMergedGrp[j];
            for( int g=1; g < nGrp+1; g++) {
                if (grpToMergedGrp[g] == curGroup) {
                    grpToMergedGrp[g] = grpToMergedGrp[j];
                }
            }
          }
        }
      }
      iGroup++;
  }

  // Perform the mapping group -> mergedGroups
  if (VERBOSE >0 ) {
    vectorPrintShort( "  grpToMergedGrp", grpToMergedGrp, nGrp+1);
  }
  //
  // Renumber the fused groups
  //
  int newGroupID = 0;
  Mask_t map[nGrp+1];
  vectorSetZeroShort( map, (nGrp+1) );
  for (int g=1; g < (nGrp+1); g++) {
    int gm = grpToMergedGrp[g];
    if ( map[gm] == 0) {
      newGroupID++;
      map[gm] = newGroupID;
    }
  }
  // vectorPrintShort( "  map", map, nGrp+1);
  // Apply the renumbering
  for (int g=1; g < (nGrp+1); g++) {
    grpToMergedGrp[g] = map[ grpToMergedGrp[g] ];
  }

  // Perform the mapping grpToMergedGrp
  if ( VERBOSE >0 ) {
    vectorPrintShort( "  grpToMergedGrp", grpToMergedGrp, nGrp+1);
  }
  for ( int c=0; c<2; c++) {
    for ( int p=0; p< pads[c]->getNbrOfPads(); p++) {
      // ??? Why abs() ... explain
      cathGroup[c][p]= grpToMergedGrp[ std::abs(cathGroup[c][p]) ];
    }
  }

  if (CHECK) {
    for ( int c=0; c<2; c++) {
      for ( int p=0; p< pads[c]->getNbrOfPads(); p++) {
        // ??? Why abs() ... explain
        // ??? cathGroup[c][p]= grpToMergedGrp[ std::abs(cathGroup[c][p]) ];
        if (cathGroup[c][p] == 0) {
          printf("  Warning  assignPadsToGroupFromProj: pad %d with no group\n", p);
        }
      }
    }
  }

  // Update the group of the proj-pads
  vectorMapShort(projPadToGrp, grpToMergedGrp, nProjPads);

  //
  return newGroupID;
}
