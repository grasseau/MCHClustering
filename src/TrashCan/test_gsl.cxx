
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

struct data {
  size_t n;
  double * y;
  double * sigma;
};

int
expb_f (const gsl_vector * x, void *params,
        gsl_vector * f)
{
  printf("expb_f\n");
  size_t n = ((struct data *)params)->n;
  double *y = ((struct data *)params)->y;
  double *sigma = ((struct data *) params)->sigma;

  double A = gsl_vector_get (x, 0);
  double lambda = gsl_vector_get (x, 1);
  double b = gsl_vector_get (x, 2);
  printf("  at A, lambda, b = %8.5f %8.5f %8.5f\n", A, lambda, b);
  size_t i;

  for (i = 0; i < n; i++)
    {
      /* Model Yi = A * exp(-lambda * i) + b */
      double t = i;
      double Yi = A * exp (-lambda * t) + b;
      gsl_vector_set (f, i, (Yi - y[i])/sigma[i]);
    }

  return GSL_SUCCESS;
}

int
expb_df (const gsl_vector * x, void *params,
         gsl_matrix * J)
{
  printf("expb_df\n");
  size_t n = ((struct data *)params)->n;
  double *sigma = ((struct data *) params)->sigma;

  double A = gsl_vector_get (x, 0);
  double lambda = gsl_vector_get (x, 1);

  double b = gsl_vector_get (x, 2);
  printf("  at A, lambda, b = %8.5f %8.5f %8.5f\n", A, lambda, b);

  size_t i;

  for (i = 0; i < n; i++)
    {
      /* Jacobian matrix J(i,j) = dfi / dxj, */
      /* where fi = (Yi - yi)/sigma[i],      */
      /*       Yi = A * exp(-lambda * i) + b  */
      /* and the xj are the parameters (A,lambda,b) */
      double t = i;
      double s = sigma[i];
      double e = exp(-lambda * t);
      gsl_matrix_set (J, i, 0, e/s);
      gsl_matrix_set (J, i, 1, -t * A * e/s);
      gsl_matrix_set (J, i, 2, 1/s);
      printf("  Jacobian n=%d: A, lambda, b = %8.5f %8.5f %8.5f\n", i, e/s, -t * A * e/s, 1/s);

    }
  return GSL_SUCCESS;
}

int
expb_fdf (const gsl_vector * x, void *params,
          gsl_vector * f, gsl_matrix * J)
{
  printf("expb_fdf\n");
  expb_f (x, params, f);
  expb_df (x, params, J);

  return GSL_SUCCESS;
}
void
print_state (size_t iter, gsl_multifit_fdfsolver * s)
{
  printf ("iter: %3u x = % 15.8f % 15.8f % 15.8f "
          "|f(x)| = %g\n",
          iter,
          gsl_vector_get (s->x, 0),
          gsl_vector_get (s->x, 1),
          gsl_vector_get (s->x, 2),
          gsl_blas_dnrm2 (s->f));
}

int
main (void)
{
  const gsl_multifit_fdfsolver_type *T;
  gsl_multifit_fdfsolver *s;

  int status;
  size_t i, iter = 0;

  const size_t n = 40;
  const size_t p = 3;

  gsl_matrix *covar = gsl_matrix_alloc (p, p);

  double y[n], sigma[n];

  struct data d = { n, y, sigma};

  gsl_multifit_function_fdf f;

  double x_init[3] = { 1.0, 0.0, 0.0 };

  gsl_vector_view x = gsl_vector_view_array (x_init, p);

  const gsl_rng_type * type;
  gsl_rng * r;

  gsl_rng_env_setup();

  type = gsl_rng_default;
  r = gsl_rng_alloc (type);

  f.f = &expb_f;
  f.df = &expb_df;
  // f.df = nullptr;
  f.fdf = &expb_fdf;
  // f.fdf = nullptr;
  f.n = n;
  f.p = p;
  f.params = &d;

  /* This is the data to be fitted */

  for (i = 0; i < n; i++)
    {
      double t = i;
      y[i] = 1.0 + 5 * exp (-0.1 * t)
                 + gsl_ran_gaussian(r, 0.1);
      sigma[i] = 0.1;
      printf("data: %d %g %g\n", i, y[i], sigma[i]);
    };

  T = gsl_multifit_fdfsolver_lmsder;
  s = gsl_multifit_fdfsolver_alloc (T, n, p);
  gsl_multifit_fdfsolver_set (s, &f, &x.vector);

  print_state (iter, s);

  do
    {
      iter++;
      status = gsl_multifit_fdfsolver_iterate (s);

      printf ("status = %s\n", gsl_strerror (status));

      print_state (iter, s);

      if (status)
        break;

      status = gsl_multifit_test_delta (s->dx, s->x,
                                        1e-4, 1e-4);
    }
  while (status == GSL_CONTINUE && iter < 500);

  gsl_multifit_covar (s->J, 0.0, covar);

  gsl_matrix_fprintf (stdout, covar, "%g");

#define FIT(i) gsl_vector_get(s->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

  printf("A      = %.5f +/- %.5f\n", FIT(0), ERR(0));
  printf("lambda = %.5f +/- %.5f\n", FIT(1), ERR(1));
  printf("b      = %.5f +/- %.5f\n", FIT(2), ERR(2));

  printf ("status = %s\n", gsl_strerror (status));

  gsl_multifit_fdfsolver_free (s);
  return 0;
}

