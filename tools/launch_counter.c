// launch_counter — LD_PRELOAD CUDA launch-API interposer (no privileges needed).
//
// Counts host-side kernel/graph launches and async mem ops; unlike CUPTI/nsys
// it works in containers where GPU profiling is administratively restricted.
//
// Build: gcc -shared -fPIC tools/launch_counter.c -ldl -o launch_counter.so
// Use:   LAUNCH_COUNTER_OUT=stats.json LD_PRELOAD=./launch_counter.so ./app
// Output: {"launches": N, "graph_launches": G, "memops": M}
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long g_launches = 0, g_graph_launches = 0, g_memops = 0;

int cudaLaunchKernel(const void* f, void* g, void* b, void** a, size_t s, void* st) {
    static int (*real)(const void*, void*, void*, void**, size_t, void*) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "cudaLaunchKernel");
    __atomic_add_fetch(&g_launches, 1, __ATOMIC_RELAXED);
    return real(f, g, b, a, s, st);
}

int cudaLaunchKernelExC(void* cfg, const void* f, void** a) {
    static int (*real)(void*, const void*, void**) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "cudaLaunchKernelExC");
    __atomic_add_fetch(&g_launches, 1, __ATOMIC_RELAXED);
    return real(cfg, f, a);
}

int cudaGraphLaunch(void* exec, void* st) {
    static int (*real)(void*, void*) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "cudaGraphLaunch");
    __atomic_add_fetch(&g_graph_launches, 1, __ATOMIC_RELAXED);
    return real(exec, st);
}

int cudaMemcpyAsync(void* dst, const void* src, size_t n, int kind, void* st) {
    static int (*real)(void*, const void*, size_t, int, void*) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    __atomic_add_fetch(&g_memops, 1, __ATOMIC_RELAXED);
    return real(dst, src, n, kind, st);
}

int cudaMemsetAsync(void* p, int v, size_t n, void* st) {
    static int (*real)(void*, int, size_t, void*) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "cudaMemsetAsync");
    __atomic_add_fetch(&g_memops, 1, __ATOMIC_RELAXED);
    return real(p, v, n, st);
}

__attribute__((destructor)) static void dump(void) {
    const char* path = getenv("LAUNCH_COUNTER_OUT");
    FILE* f = path ? fopen(path, "w") : stderr;
    if (!f) f = stderr;
    fprintf(f, "{\"launches\": %llu, \"graph_launches\": %llu, \"memops\": %llu}\n",
            g_launches, g_graph_launches, g_memops);
    if (path && f != stderr) fclose(f);
}
