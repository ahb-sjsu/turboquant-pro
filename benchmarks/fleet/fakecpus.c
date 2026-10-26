/* TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
 * Copyright (c) 2026 Andrew H. Bond
 * MIT License
 *
 * Report TQP_FAKE_CPUS CPUs from sched_getaffinity, so OpenBLAS sizes its thread pool as the
 * reference jobs did when a pod is pinned to one core. OpenBLAS takes the smaller of
 * OPENBLAS_NUM_THREADS and the CPUs in the affinity mask, and the thread count changes how it
 * splits a product and so the low bits of a score. A 1-CPU NRP pod is pinned to one core, so
 * a 6-thread search there ran at 1 thread and could not reproduce the scores of the 6-CPU
 * reference jobs. With this preloaded the threads still run on the one allowed core; only the
 * count OpenBLAS divides the work by changes. On Atlas, pinned to one core, it restores the
 * 6-thread score bits of server 12 shard 4803 (2026-10-26).
 *
 * Build: gcc -O2 -shared -fPIC -o fakecpus.so fakecpus.c
 * No libc number parsing, so the library needs nothing newer than glibc 2.34 (dlsym).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>
#include <stdlib.h>

static int fake_cpus(void) {
    const char *s = getenv("TQP_FAKE_CPUS");
    int n = 0;
    if (!s) return 0;
    for (; *s >= '0' && *s <= '9'; s++) n = n * 10 + (*s - '0');
    return *s ? 0 : n;
}

int sched_getaffinity(pid_t pid, size_t size, cpu_set_t *mask) {
    static int (*real)(pid_t, size_t, cpu_set_t *) = 0;
    if (!real) real = (int (*)(pid_t, size_t, cpu_set_t *))dlsym(RTLD_NEXT, "sched_getaffinity");
    int rc = real(pid, size, mask);
    int want = fake_cpus();
    if (rc == 0 && want > 0) {
        CPU_ZERO_S(size, mask);
        for (int i = 0; i < want && (size_t)i < size * 8; i++) CPU_SET_S(i, size, mask);
    }
    return rc;
}
