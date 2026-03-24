/*
 * C benchmark for indiflight rlsParallelNewSample — times the standard
 * RLS on representative data for head-to-head comparison with Rust.
 *
 * Build:
 *   make -f Makefile.bench
 *
 * Run:
 *   ./bench_c
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* ── Inline indiflight RLS (self-contained, no external deps) ──────── */

#define RLS_MAX_N 8
#define RLS_MAX_P 6
#define RLS_COV_MAX 1e+10f
#define RLS_COV_MIN 1e-10f
#define RLS_MAX_P_ORDER_DECREMENT 0.1f

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

typedef struct {
    int n;
    int p;
    float X[RLS_MAX_N * RLS_MAX_P];
    float P[RLS_MAX_N * RLS_MAX_N];
    float lambda;
    uint32_t samples;
} rls_parallel_t;

static void rls_init(rls_parallel_t *rls, int n, int p, float gamma, float lambda) {
    memset(rls, 0, sizeof(*rls));
    rls->n = n;
    rls->p = p;
    rls->lambda = lambda;
    for (int i = 0; i < n; i++)
        rls->P[i * n + i] = gamma;
}

static void rls_update(rls_parallel_t *rls, const float *aT, const float *yT) {
    rls->samples++;
    int n = rls->n;
    int p = rls->p;

    float lam = rls->lambda;
    float diagP[RLS_MAX_N];
    for (int i = 0; i < n; i++) {
        diagP[i] = rls->P[i + i * n];
        if (diagP[i] > RLS_COV_MAX)
            lam = 1.f + 0.1f * (1.f - rls->lambda);
    }

    /* P aT */
    float PaT[RLS_MAX_N];
    for (int i = 0; i < n; i++) {
        PaT[i] = 0.f;
        for (int k = 0; k < n; k++)
            PaT[i] += rls->P[k + i * n] * aT[k];
    }

    /* a P aT (scalar) */
    float aPaT = 0.f;
    for (int i = 0; i < n; i++)
        aPaT += aT[i] * PaT[i];

    /* prediction error */
    float eT[RLS_MAX_P];
    for (int j = 0; j < p; j++) {
        eT[j] = 0.f;
        for (int k = 0; k < n; k++)
            eT[j] += aT[k] * rls->X[k + j * n];
        eT[j] = yT[j] - eT[j];
    }

    /* gain */
    float isig = 1.f / (lam + aPaT);
    float k[RLS_MAX_N];
    for (int i = 0; i < n; i++)
        k[i] = PaT[i] * isig;

    /* order decrement limiting (per-diagonal) */
    float maxDiagRatio = 0.f;
    for (int i = 0; i < n; i++) {
        float dkap = k[i] * PaT[i];
        if (dkap > 1e-6f) {
            float ratio = diagP[i] / dkap;
            if (ratio > maxDiagRatio)
                maxDiagRatio = ratio;
        }
    }
    float KAPmult = 1.f;
    if (maxDiagRatio > RLS_COV_MIN)
        KAPmult = MIN((1.f - RLS_MAX_P_ORDER_DECREMENT) * maxDiagRatio, 1.f);

    /* covariance update: P = (KAPmult*P - k*PaT') / lambda */
    float ilam = 1.f / lam;
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            float elem = -KAPmult * rls->P[row + col * n] + k[row] * PaT[col];
            rls->P[row + col * n] = -ilam * elem;
        }
    }

    /* parameter update: X += k * eT' */
    for (int j = 0; j < p; j++)
        for (int i = 0; i < n; i++)
            rls->X[i + j * n] += k[i] * eT[j];
}

/* ── PRNG (xorshift32, matches Rust bench) ─────────────────────────── */

static uint32_t xor_state;
static float xor_next_f32(void) {
    xor_state ^= xor_state << 13;
    xor_state ^= xor_state >> 17;
    xor_state ^= xor_state << 5;
    return (float)xor_state / (float)UINT32_MAX * 2.0f - 1.0f;
}

/* ── Benchmark ─────────────────────────────────────────────────────── */

#define N_WARMUP  50
#define N_ITERS   10000
#define N_STEPS   100

static void bench(const char *label, int n, int p, float gamma, float lambda, uint32_t seed) {
    /* Generate data */
    xor_state = seed;
    float regressors[N_STEPS][RLS_MAX_N];
    float observations[N_STEPS][RLS_MAX_P];
    for (int s = 0; s < N_STEPS; s++) {
        for (int i = 0; i < n; i++)
            regressors[s][i] = xor_next_f32();
        for (int i = 0; i < p; i++)
            observations[s][i] = xor_next_f32();
    }

    /* Warmup */
    for (int w = 0; w < N_WARMUP; w++) {
        rls_parallel_t rls;
        rls_init(&rls, n, p, gamma, lambda);
        for (int s = 0; s < N_STEPS; s++)
            rls_update(&rls, regressors[s], observations[s]);
    }

    /* Timed runs */
    struct timespec t0, t1;
    float final_x00 = 0.f;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < N_ITERS; r++) {
        rls_parallel_t rls;
        rls_init(&rls, n, p, gamma, lambda);
        for (int s = 0; s < N_STEPS; s++)
            rls_update(&rls, regressors[s], observations[s]);
        if (r == N_ITERS - 1)
            final_x00 = rls.X[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double total_ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                      (double)(t1.tv_nsec - t0.tv_nsec);
    double ns_per_seq = total_ns / N_ITERS;
    double ns_per_step = ns_per_seq / N_STEPS;

    printf("%s,%d,c_standard,%.1f,%.1f,%.8e\n",
           label, N_STEPS, ns_per_seq, ns_per_step, final_x00);
}

int main(void) {
    printf("config,steps,solver,ns_per_sequence,ns_per_step,final_x00\n");
    bench("motor_n4p1", 4, 1, 100.f, 0.995f, 42);
    bench("g1g2_n8p3",  8, 3, 100.f, 0.995f, 123);
    bench("force_n4p3", 4, 3, 100.f, 0.995f, 77);
    return 0;
}
