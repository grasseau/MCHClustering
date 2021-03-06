#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

#include "mathiesonFit.c"

#define N 40

void print_state (size_t iter, gsl_multifit_fdfsolver * s);

int main (void) {
  // const gsl_multifit_fdfsolver_type *T;

  int status;
  unsigned int i, iter = 0;
  const size_t n = N;
  const size_t p = 3;

  // Data to fit & std-error on y
  double y[N], sigma[N];

 
  // RNG
  const gsl_rng_type * type;
  gsl_rng * r;
  gsl_rng_env_setup();
  type = gsl_rng_default;
  r = gsl_rng_alloc (type);

  // Define Function, jacobian
  gsl_multifit_function_fdf f;
  f.f = &expb_f;
  // f.df = &expb_df;
  // f.fdf = &expb_fdf;
  f.df = 0;
  f.fdf = 0;
  f.n = n;
  f.p = p;
  
  // y & Parameters
  struct data d = { n, y, sigma};
  f.params = &d;
  // Initial parameters
  double params_init[3] = { 1.0, 0.0, 0.0 };
  gsl_vector_view params = gsl_vector_view_array (params_init, p);
  

  // Input data to be fitted
  for (i = 0; i < n; i++) {
      double t = i;
      y[i] = 1.0 + 5 * exp (-0.1 * t) + gsl_ran_gaussian (r, 0.1);
      sigma[i] = 0.1;
      // printf ("data: %u %g %g\n", i, y[i], sigma[i]);
    };

  // Fitting method
  gsl_multifit_fdfsolver *s = gsl_multifit_fdfsolver_alloc ( gsl_multifit_fdfsolver_lmsder, n, p);
  // associate the fitting mode, the function, and the starting parameters
  gsl_multifit_fdfsolver_set (s, &f, &params.vector);

  // Covariance matrix
  gsl_matrix *covar = gsl_matrix_alloc (p, p);
  print_state (iter, s);

  // Do the fitting
  do {
      iter++;
      status = gsl_multifit_fdfsolver_iterate ( s );
      printf ("status = %s\n", gsl_strerror (status));
      print_state (iter, s);
      if (status)
        break;
      status = gsl_multifit_test_delta (s->dx, s->x, 1e-4, 1e-4);
  } while (status == GSL_CONTINUE && iter < 500);

  // Compute and fill the covariance
  gsl_multifit_covar (s->J, 0.0, covar);

#define FIT(i) gsl_vector_get(s->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

  { 
    double chi = gsl_blas_dnrm2(s->f);
    double dof = n - p;
    double c = GSL_MAX_DBL(1, chi / sqrt(dof)); 

    printf("chisq/dof = %g\n",  pow(chi, 2.0) / dof);

    printf ("A      = %.5f +/- %.5f\n", FIT(0), c*ERR(0));
    printf ("lambda = %.5f +/- %.5f\n", FIT(1), c*ERR(1));
    printf ("b      = %.5f +/- %.5f\n", FIT(2), c*ERR(2));
  }

  printf ("status = %s\n", gsl_strerror (status));

  // Release memory
  gsl_multifit_fdfsolver_free (s);
  gsl_matrix_free (covar);
  gsl_rng_free (r);
  return 0;
}

void print_state (size_t iter, gsl_multifit_fdfsolver * s) {
  printf ("iter: %3u params = % 15.8f % 15.8f % 15.8f "
          "|f(x)| = %g\n",
          iter,
          gsl_vector_get (s->x, 0), 
          gsl_vector_get (s->x, 1),
          gsl_vector_get (s->x, 2), 
          gsl_blas_dnrm2 (s->f));
}

