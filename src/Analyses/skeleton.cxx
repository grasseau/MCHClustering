processPreCluster() {
  // test suivant le nbre de pads (pb cluster pas connexes- ex en forme de H)
  if nPads[x/y] < 4 {
    // processSimple : MLEM WITHOUT refinement, force to ONE seed
    processSimple()
  } else {
    // findLocalMaxima: with histogram
    // no solution if the nbr of local max exceed > 99
    findLocalMaxima()
    if (nPads < 50 ) {
      // process: MLEM with refinement, split ( fitting in split method)
      process()
    } else {
      for (k in localMaxima) {
        // restrictPreCluster: select the part of the precluster that is around the local maximum
        restrictPreCluster(localMaxima[k]);
        process()
      }
    }
  }
}

Rq: si nPads <= 2, pas de fit
Fit sur nSeeds, choisi le meilleur chi2 parmi ???