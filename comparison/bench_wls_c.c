/*
 * C benchmark for ActiveSetCtlAlloc — times QR_NAIVE and QR solvers on all
 * 1000 test cases. Outputs CSV to stdout.
 *
 * Build:
 *   make -f Makefile.bench
 *
 * Run:
 *   ./bench_c > results_c.csv
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "solveActiveSet.h"
#include "setupWLS.h"
#include "test_cases.h"

#define N_WARMUP  50
#define N_ITERS   1000

static TestCase test_cases[N_CASES];

static double bench_one(
    activeSetAlgoChoice algo,
    const TestCase *tc,
    num_t *us_out,
    int *exit_code_out,
    int *iter_out)
{
    int n_u = tc->n_u;
    int n_v = tc->n_v;

    /* Setup WLS problem */
    num_t Wu[AS_N_U];
    memcpy(Wu, tc->Wu, sizeof(num_t) * n_u);

    num_t A[AS_N_C * AS_N_U];
    num_t b[AS_N_C];
    num_t gamma;

    setupWLS_A(tc->JG, tc->Wv, Wu, n_v, n_u, 2.0e-9, 4e5, A, &gamma);
    setupWLS_b(tc->v, tc->up, tc->Wv, Wu, n_v, n_u, gamma, b);

    /* Warm up */
    for (int w = 0; w < N_WARMUP; w++) {
        num_t us[AS_N_U];
        memcpy(us, tc->u0, sizeof(num_t) * n_u);
        int8_t Ws[AS_N_U];
        memset(Ws, 0, sizeof(Ws));
        int iter, n_free;
        num_t costs[1];
        solveActiveSet(algo)(A, b, tc->lb, tc->ub, us, Ws, 100, n_u, n_v,
                             &iter, &n_free, costs);
    }

    /* Timed runs */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int r = 0; r < N_ITERS; r++) {
        num_t us[AS_N_U];
        memcpy(us, tc->u0, sizeof(num_t) * n_u);
        int8_t Ws[AS_N_U];
        memset(Ws, 0, sizeof(Ws));
        int iter, n_free;
        num_t costs[1];
        activeSetExitCode ec = solveActiveSet(algo)(
            A, b, tc->lb, tc->ub, us, Ws, 100, n_u, n_v,
            &iter, &n_free, costs);

        /* Capture last iteration results */
        if (r == N_ITERS - 1) {
            memcpy(us_out, us, sizeof(num_t) * n_u);
            *exit_code_out = (int)ec;
            *iter_out = iter;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                (double)(t1.tv_nsec - t0.tv_nsec);
    return ns / N_ITERS;
}

int main(void)
{
    fill_cases(test_cases);

    /* CSV header */
    printf("case,solver,exit_code,iterations,time_ns,"
           "us0,us1,us2,us3,us4,us5\n");

    const char *names[] = {"qr_naive", "qr"};
    activeSetAlgoChoice algos[] = {AS_QR_NAIVE, AS_QR};

    for (int s = 0; s < 2; s++) {
        for (int c = 0; c < N_CASES; c++) {
            num_t us[AS_N_U] = {0};
            int ec = 0, iter = 0;
            double ns = bench_one(algos[s], &test_cases[c], us, &ec, &iter);

            printf("%d,%s,%d,%d,%.1f,"
                   "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                   c, names[s], ec, iter, ns,
                   us[0], us[1], us[2], us[3], us[4], us[5]);
        }
    }

    return 0;
}
