
/* C source code is found in dgemm_example.c */

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>
#include <malloc.h>
#include <omp.h>

#ifdef GEM5ROI
#include "gem5/m5ops.h"
#endif


double get_seconds() {
    struct timeval now;
    gettimeofday(&now, NULL);

    const double seconds = (double) now.tv_sec;
    const double usec    = (double) now.tv_usec;

    return seconds + (usec * 1.0e-6);
}

int main(int argc, char *argv[])
{
    if (argc!= 5){
        printf("Run: dgemm.x <min_size> <max_size> <size_step> <niterations>");
        return 0;
    }
    printf("# OpenBLAS::config=%s\n", openblas_get_config());

    printf("Number of OpenMP threads: %d", omp_get_num_threads());
#pragma omp parallel 
    {

        /* Obtain thread number */
        int tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        /* Only master thread does this */
        if (tid == 0) 
        {
            int nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }

    } /* All threads join master thread and disband */


    printf ("\nSTARTING DGEMM BENCHMARKS.\n");

    char *a = argv[1];
    int size_start = atoi(a);
    char *b = argv[2];
    int size_end = atoi(b);
    char *c = argv[3];
    int stride = atoi(c);
    char *d = argv[4];
    int iters = atoi(d);
    double *A, *B, *C;

    A = (double *) malloc(size_end*size_end*sizeof( double ));
    B = (double *) malloc(size_end*size_end*sizeof( double ));
    C = (double *) malloc(size_end*size_end*sizeof( double ));

    /* A = (double *) memalign(64, size_end*size_end*sizeof( double )); */
    /* B = (double *) memalign(64, size_end*size_end*sizeof( double )); */
    /* C = (double *) memalign(64, size_end*size_end*sizeof( double )); */

    /* posix_memalign((void**)&A, 64, size_end*size_end*sizeof( double )); */
    /* posix_memalign((void**)&B, 64, size_end*size_end*sizeof( double )); */
    /* posix_memalign((void**)&C, 64, size_end*size_end*sizeof( double )); */

    printf ("\nSuccessfully allocated memory for the matrices A, B, C.\n");

    int N = size_end;

#pragma omp parallel for
        for(int i = 0; i < size_end; i++) {
            for(int j = 0; j < size_end; j++) {
                A[i*N + j] = 2.0;
                B[i*N + j] = 0.5;
                C[i*N + j] = 1.0;
            }
        }



    for (int s = size_start; s <= size_end; s+=stride) {


        int64_t m, n, k, i, j;
        double alpha, beta;

        m = s, k = s, n = s;


        alpha = 2.5; beta = 1.5;

        double sum = 0.0f;
        double min = 1e8;
        double max = 0.0f;
        for (int iter = 0; iter < iters; iter++) {
#ifdef GEM5ROI
            m5_dump_reset_stats(0,0);
#endif
            double start = get_seconds();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);

            double end = get_seconds();
#ifdef GEM5ROI
            m5_dump_reset_stats(0,0);
#endif
            double time = end-start;
            if (time<min) min = time;
            if (time>max) max = time;
            sum += time;
        }
        double flops = n*n*n*2.0 + n*n*2;

        printf ("\n\nComputations completed for size M=N=K= %ld and %d iterations.\n", m, iters);
        printf("Average time:            %f\n", sum/iters);
        printf("MIN time:            %f\n", min);
        printf("MAX time:            %f\n", max);
        printf("FLOPs computed:          %f\n", flops);
        printf("Max GFLOP/s rate:        %f GF/s\n", (flops / min) / 1000000000.0);



    }
        free(A);
        free(B);
        free(C);

    return 0;
}
