#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef double (*fun)(double, double);

typedef struct meta {
  int64_t size;
  double h;

  double eps;
  int64_t block_size;

  int64_t iter;

  double **u;
  double **f;
} meta;

double **init_matrix(int size) {
  double **foo = calloc(size, sizeof(*foo));
  for (int i = 0; i < size; i++)
    foo[i] = calloc(size, sizeof(*foo[i]));
  return foo;
}

void init_meta(meta *meta, int64_t size, fun f, fun u, int64_t bs, double eps) {
  meta->size = size;
  meta->h = 1.0 / (size - 1);

  meta->block_size = bs;
  meta->eps = eps;

  meta->iter = 0;

  meta->u = init_matrix(size);
  meta->f = init_matrix(size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if ((i == 0) || (i == (size - 1)) || (j == 0) || (j == (size - 1))) {
        meta->u[i][j] = u(meta->h * i, meta->h * j);
      } else {
        meta->u[i][j] = 0;
      }
      meta->f[i][j] = f(meta->h * i, meta->h * j);
    }
  }
}

void free_matrix(double **mtx, int64_t size) {
  for (int64_t i = 0; i < size; i++) {
    free(mtx[i]);
  }
  free(mtx);
}

void free_meta(meta *meta) {
  free_matrix(meta->u, meta->size);
  free_matrix(meta->f, meta->size);
  free(meta);
}

double block_processing(int bi, int bj, meta *meta) {
  double dm = 0, temp;

  for (int i = 1 + bi * meta->block_size;
       i <= MIN((bi + 1) * meta->block_size, meta->size - 2); i++) {
    for (int j = 1 + bj * meta->block_size;
         j <= MIN((bj + 1) * meta->block_size, meta->size - 2); j++) {
      temp = meta->u[i][j];
      meta->u[i][j] =
          0.25 * (meta->u[i - 1][j] + meta->u[i + 1][j] + meta->u[i][j - 1] +
                  meta->u[i][j + 1] - meta->h * meta->h * meta->f[i][j]);
      dm = MAX(dm, fabs(temp - meta->u[i][j]));
    }
  }
  return dm;
}

void blocks_processing(meta *meta) {
  int64_t N = meta->size - 2;
  int64_t NB = (N + meta->block_size - 1) / meta->block_size;
  double dmax;
  double *dm = calloc(NB, sizeof(*dm));

  do {
    dmax = 0;
    for (int64_t nx = 0; nx < NB; nx++) {
      dm[nx] = 0;
      double d;
      int64_t i, j;
#pragma omp parallel for shared(nx, dm, meta) private(i, j, d)
      for (i = 0; i < nx + 1; i++) {
        j = nx - i;

        d = block_processing(i, j, meta);
        dm[i] = MAX(dm[i], d);
      }
    }

    for (int64_t nx = NB - 2; nx >= 0; nx--) {
      double d;
      int64_t i, j;
#pragma omp parallel for shared(nx, dm, meta) private(i, j, d)
      for (i = NB - nx - 1; i < NB; i++) {
        j = 2 * (NB - 1) - nx - i;

        d = block_processing(i, j, meta);
        dm[i] = MAX(dm[i], d);
      }
    }

    for (int64_t i = 0; i < NB; i++) {
      dmax = MAX(dmax, dm[i]);
    }
    meta->iter++;
  } while (dmax > meta->eps);

  free(dm);
}

meta *approximate(int64_t size, fun f, fun u, int64_t block_size, double eps) {
  meta *mt = calloc(1, sizeof(meta));
  init_meta(mt, size, f, u, block_size, eps);
  blocks_processing(mt);
  return mt;
}

double d_kx3_p_2ky3(double x, double y) { return 6000 * x + 12000 * y; }
double kx3_p_2ky3(double x, double y) {
  return 1000 * pow(x, 3) + 2000 * pow(y, 3);
}
double sinx_p_cosy3(double x, double y) { return sin(x) + pow(cos(y), 3); }
double d_sinx_p_cosy3(double x, double y) {
  return -sin(x) - 3 * (1 - 3 * pow(sin(y), 2)) * cos(y);
}

int main() {
  FILE *fpt;
  fpt = fopen("results.csv", "w");
  fprintf(fpt, "Thread number, Size, Block size, Epsilon, Iteration number, "
               "Time, Boost, Maximal error, Average error\n");

  int num_threads[] = {1, 8, 12};
  int size[] = {10, 60, 100, 500, 1000, 2000};
  int block_size[] = {16, 64, 128, 256};
  double eps[] = {0.0001, 0.00001};

  int seq[6][4][2];

  fun df = d_sinx_p_cosy3;
  fun f = sinx_p_cosy3;

  int sz, bs, nt, e;

  for (nt = 0; nt < 3; nt++) {
    omp_set_num_threads(num_threads[nt]);
    for (sz = 0; sz < 6; sz++)

    {
      for (bs = 0; bs < 4; bs++) {
        for (e = 0; e < 2; e++) {

          double t_start = omp_get_wtime();
          meta *mt = approximate(size[sz], df, f, block_size[bs], eps[e]);
          double t_end = omp_get_wtime();

          double d = 0, average_err = 0, cnt = 0;
          for (int i = 0; i < mt->size; i++) {
            for (int j = 0; j < mt->size; j++) {
              cnt++;
              double dif = fabs(mt->u[i][j] - f(i * mt->h, j * mt->h));
              average_err = (average_err * (cnt - 1) + dif) / cnt;
              d = MAX(d, dif);
            }
          }

          double time = t_end - t_start;
          if (nt == 0) {
            fprintf(fpt, "1, %d, %d, %f, %ld, %7.4f, -, %7.2f, %7.2f\n",
                    size[sz], block_size[bs], eps[e], mt->iter, time, d,
                    average_err);
            seq[sz][bs][e] = time;
          } else {
            double boost = seq[sz][bs][e] / time;
            fprintf(fpt, "%d, %d, %d, %f, %ld, %7.4f, %7.2f, %7.2f, %7.2f\n",
                    num_threads[nt], size[sz], block_size[bs], eps[e], mt->iter,
                    time, boost, d, average_err);
          }
          free_meta(mt);
        }
      }
    }
  }
}
