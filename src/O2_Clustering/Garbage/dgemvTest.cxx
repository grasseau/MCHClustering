/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <stdio.h>
#include <gsl/gsl_blas.h>

int
main (void)
{
  double a[] = { 0.1, 0.2,
                 0.1, 0.2,
                 0.1, 0.2};

  double b[] = { 1, 2, 3 };

  double c[] = { 5.00, 8.00, 9.00 };

  gsl_matrix_view A = gsl_matrix_view_array(a, 3, 2);
  gsl_vector_view B = gsl_vector_view_array(b, 3);
  gsl_vector_view C = gsl_vector_view_array(c, 2);

  /* Compute C = A B */

  gsl_blas_dgemv (CblasTrans,
                  1.0, &A.matrix, &B.vector,
                  0.0, &C.vector);

  printf ("[ %g, %g %g\n", c[0], c[1], c[2]);
  // printf ("  %g, %g ]\n", c[2], c[3]);

  return 0;
}
