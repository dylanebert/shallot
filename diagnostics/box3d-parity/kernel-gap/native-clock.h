#pragma once
#include <stdatomic.h>
#include <time.h>
#include <limits.h>
extern int diagnostic_timing;
extern unsigned long long diagnostic_bias;
extern _Atomic unsigned diagnostic_counts[19];
extern unsigned long long diagnostic_times[22][16];
extern unsigned long long diagnostic_self[22][16];
extern unsigned diagnostic_calls[22][16];
extern _Atomic unsigned diagnostic_next_thread;
#define DC(i,n) do { if (!diagnostic_timing) atomic_fetch_add_explicit(&diagnostic_counts[i], (n), memory_order_relaxed); } while (0)
static inline unsigned long long dn(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + t.tv_nsec;
}
typedef struct DiagnosticTimer {
    int i;
    unsigned long long begin, overhead, children;
    struct DiagnosticTimer* parent;
} DiagnosticTimer;
extern _Thread_local DiagnosticTimer* diagnostic_top;
extern _Thread_local unsigned diagnostic_thread;
extern _Thread_local unsigned long long diagnostic_overhead;
static inline void dt_start(DiagnosticTimer* t) {
    if (!diagnostic_timing) return;
    unsigned long long before = dn();
    if (diagnostic_thread == UINT_MAX) diagnostic_thread = atomic_fetch_add(&diagnostic_next_thread, 1);
    if (diagnostic_thread >= 16) __builtin_trap();
    t->parent = diagnostic_top;
    diagnostic_top = t;
    t->begin = dn();
    diagnostic_overhead += t->begin - before;
    t->overhead = diagnostic_overhead;
}
static inline void dt_end(DiagnosticTimer* t) {
    if (!diagnostic_timing) return;
    unsigned long long end = dn();
    unsigned long long elapsed = end - t->begin;
    unsigned long long overhead = diagnostic_overhead - t->overhead + diagnostic_bias;
    unsigned long long net = elapsed > overhead ? elapsed - overhead : 0;
    unsigned thread = diagnostic_thread;
    diagnostic_times[t->i][thread] += net;
    diagnostic_self[t->i][thread] += net > t->children ? net - t->children : 0;
    diagnostic_calls[t->i][thread] += 1;
    diagnostic_top = t->parent;
    if (t->parent) t->parent->children += net;
    diagnostic_overhead += dn() - end + diagnostic_bias;
}
#define DT(k) DiagnosticTimer dt __attribute__((cleanup(dt_end))) = {.i = k}; dt_start(&dt);
